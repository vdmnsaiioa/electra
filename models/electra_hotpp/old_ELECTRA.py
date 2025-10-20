from __future__ import annotations

import time

import wandb
import logging
from tools.atom_tools import valence_electrons
from ase.data import chemical_symbols
from utils.model_handling import ModelIO
from torch.utils.data import Dataset, DataLoader
import matgl
import numpy as np
import torch
import re
import plotly.graph_objs as go
import plotly.io as pio
from matgl.utils.io import IOMixIn
from torch import nn
import json

from models.dists import gto, get_full_grid_dens, get_n_points_dens, get_n_points_dens_single
from tools.ase_converter import Atoms2Graph
from modules.grid2pos import grid2pos
from tools.density_conversions import normalize_density, cd_to_chgcar, get_qm9_density
from tools.graph_tools import CollateFuncAtoms
from utils.train_helper_funcs import set_optimizer
from modules.loss import DensityLoss
import os
from tools.visualization import create_chg_delta
import lightning as L
from models.electra_hotpp.layer import AtomicEmbedding, BesselPoly, PolynomialCutoff, AufbauEmbedding
from models.electra_hotpp.model import MiaoNet, MiaoMiaoNet
from ase.data import atomic_numbers
from ase.neighborlist import neighbor_list

logger = logging.getLogger(__file__)


class ELECTRA(L.LightningModule, IOMixIn):
    __version__ = 1

    def __init__(
            self,
            train_files: list[str] = None,
            test_files: list[str] = None,
            validation_files: list[str] = None,
            model_handler: ModelIO = None,
            config: dict = None,
            **kwargs,
    ):
        r"""

        """
        super().__init__()

        self.save_args(locals(), kwargs)
        self.loss = DensityLoss(config=config,
                                device=self.device)
        self.activation_type = config['activation_type']
        self.gaus_per_electrons = config['gaus_per_electrons']
        self.units = self.gaus_per_electrons
        self.master_units = config['master_units']
        self.num_layers = config['hotpp_nlayers']
        self.hotpp_outdim = config['hotpp_outdim']
        self.r_max = config['r_max']
        self.n_max = config['n_max']
        self.out_max = config['out_max']
        self.cutoff = config['hotpp_cutoff']
        self.negative_contributions = config['negative_contributions']
        self.config = config
        self.train_files = train_files
        self.validation_files = validation_files
        self.test_files = test_files
        self.qm9_dens_path = config['qm9_dens_path']
        self.pred_dens_path_val = config['pred_dens_path_val']
        self.pred_dens_path_test = config['pred_dens_path_test']
        self.density_delta_path_val = config['density_delta_path_val']
        self.density_delta_path_test = config['density_delta_path_test']
        self.gaus_pos_path_val = config['gaus_pos_path_val']
        self.gaus_pos_path_test = config['gaus_pos_path_test']
        self.model_handler = model_handler
        self.best_val_loss = float("inf")
        self.collate_func = CollateFuncAtoms(self.cutoff)
        self.g_converter = Atoms2Graph(cutoff=self.cutoff, element_types=('H', 'O', 'C', 'N', 'F', 'Cl', 'Br', 'I'))
        self.val_error_dict = {}
        self.val_iter = 0
        for i, file in enumerate(self.validation_files):
            name = f"Val_file_{i}"
            self.val_error_dict[name] = float("inf")

        # self.device = device
        self.cutoff = config['hotpp_cutoff']
        self.element_types = ['H', 'O', 'C', 'N', 'F', 'Cl', 'Br', 'I']
        dtype = matgl.float_th

        self.init_hotpp_model()
        self.plot_gaus_pos = config['plot_gaus_pos']
        if self.config['hpc']:
            self.simple_gaus = True
        else:
            self.simple_gaus = False

        self.wsm_network = nn.Sequential(
            nn.Linear(4 * self.units, self.units * 3),
            nn.Mish(),
            nn.Linear(self.units * 3, 2*self.units)
        )
        self.pos_factors_network = nn.Sequential(
            nn.Linear(4 * self.units, self.units * 3),
            nn.Mish(),
            nn.Linear(self.units * 3, self.units * 3),
        )
        ## ATOMIC TRANSFORM
        self.matrix_factors_network = nn.Sequential(
            nn.Linear(4 * self.units, self.units * 3),
            nn.Mish(),
            nn.Linear(self.units * 3, self.units * 3),
        )

        ## ENERGY NETWORK
        if self.config['energy_readout_mode'] == "AW":
            units_to_use_for_readout = self.master_units
        else:
            units_to_use_for_readout = self.units
        self.phi_energy = nn.Sequential(
            nn.Linear(3 * units_to_use_for_readout, units_to_use_for_readout * 3),
            nn.Mish(),
            nn.Linear(units_to_use_for_readout * 3, units_to_use_for_readout * 3),
        )
        self.rho_energy = nn.Sequential(
            nn.Linear(3 * units_to_use_for_readout, units_to_use_for_readout * 3),
            nn.Mish(),
            nn.Linear(units_to_use_for_readout * 3, 1),
        )
        # i add this energy mlp
       # MLP that predicts the energy directly from the l=0 features of the
        # three heads. This network is optional and can be used instead of the
        # two‑stage energy readout above.
        self.energy_mlp = nn.Sequential(
            nn.Linear(3 * units_to_use_for_readout, units_to_use_for_readout * 3),
            nn.Mish(),
            nn.Linear(units_to_use_for_readout * 3, 1),
        )

        self.precision = torch.float32
        self.train_steps = 0


    def forward(self,
                atoms,
                state_attr: torch.Tensor | None = None,
                n_elec: int | None = None,
                rotation_matrix: torch.Tensor | None = None,
                orig_pos: torch.Tensor | None = None,
                **kwargs):
        """

        Args:
            g : DGLGraph for a batch of graphs.
            state_attr: State attrs for a batch of graphs.
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            output: output: Output property for a batch of graphs
        """
        start = time.time()
        # Obtain graph, with distances and relative position vectors
        idx_i, idx_j, offsets = neighbor_list("ijS", atoms, self.cutoff, self_interaction=False)
        offset = np.array(offsets) @ atoms.get_cell()
        data = {
            "atomic_number": torch.tensor(atoms.numbers, dtype=torch.long, device=self.device),
            "idx_i": torch.tensor(idx_i, dtype=torch.long, device=self.device),
            "idx_j": torch.tensor(idx_j, dtype=torch.long, device=self.device),
            "coordinate": torch.tensor(atoms.positions, dtype=self.precision, device=self.device),
            "n_atoms": torch.tensor([len(atoms)], dtype=torch.long, device=self.device),
            "offset": torch.zeros_like(torch.tensor(offset, dtype=self.precision, device=self.device)),
            "scaling": torch.eye(3, dtype=self.precision, device=self.device).view(1, 3, 3),
            "batch": torch.zeros(len(atoms), dtype=torch.long, device=self.device),
        }
        at_numbers = data['atomic_number']
        X_scal_full, X_vec_full, X_tens_full, init_emb = self.base_model(batch_data=data,
                                                                         properties=None,
                                                                         create_graph=False)

        X_scal_all = X_scal_full[0]
        X_scal_vec = X_scal_full[1]
        X_scal_tens = X_scal_full[2]

        X_vec_scal_all = X_vec_full[0]
        X_vec = X_vec_full[1]
        X_vec_tens = X_vec_full[2]

        X_tens_scal_all = X_tens_full[0]
        X_tens_vec = X_tens_full[1]
        X_tens = X_tens_full[2]

        valence_tensor = torch.tensor([valence_electrons(chemical_symbols[atom]) for atom in at_numbers], device=self.device)
        #valence_tensor = at_numbers

        # Compute the element-wise minimum between valence_tensor and fixed_value_tensor
        n_multiples = valence_tensor * self.units
        n_multiples = n_multiples.to(self.device)
        atom_embeds = init_emb.repeat_interleave(valence_tensor, dim=0)

        # Concatenate the selected slices along the first dimension
        n_atoms = torch.sum(valence_tensor)

        X_scal = self.slice(X_scal_all, n_multiples).view(n_atoms, self.units)
        X_scal_vec = self.slice(X_scal_vec, n_multiples).view(n_atoms, self.units, 3)
        X_scal_tens = self.slice(X_scal_tens, n_multiples).view(n_atoms, self.units, 3, 3)

        X_vec_scal = self.slice(X_vec_scal_all, n_multiples).view(n_atoms, self.units)
        X_vec = self.slice(X_vec, n_multiples).view(n_atoms, self.units, 3)
        X_vec_tens = self.slice(X_vec_tens, n_multiples).view(n_atoms, self.units, 3, 3)

        X_tens_scal = self.slice(X_tens_scal_all, n_multiples).view(n_atoms, self.units)
        X_tens_vec = self.slice(X_tens_vec, n_multiples).view(n_atoms, self.units, 3)
        X_tens = self.slice(X_tens, n_multiples).view(n_atoms, self.units, 3, 3)
        if self.config['remove_vector_components']:
            X_tens = self.remove_vector_component(X_tens)
            X_vec_tens = self.remove_vector_component(X_vec_tens)
            X_scal_tens = self.remove_vector_component(X_scal_tens)

        pos_disp = X_vec.view(-1, 3)
        pos_disp_2 = X_scal_vec.view(-1, 3)
        pos_disp_3 = X_tens_vec.view(-1, 3)

        wsm_ = torch.cat((atom_embeds[:, :self.units], X_scal, X_vec_scal, X_tens_scal), dim=-1).view(n_atoms, -1)
        pos_scalars = self.pos_factors_network(wsm_)
        matrix_scalars = self.matrix_factors_network(wsm_)
        wsm = self.wsm_network(wsm_)

        weights_sm = torch.softmax(wsm[:, :self.units].reshape(n_atoms * self.units), dim=0)
        scal_mults_th = wsm[:, self.units:self.units * 2].reshape(n_atoms * self.units)

        pos_factors = pos_scalars[:, :self.units].reshape(n_atoms * self.units, 1)
        pos_factors_2 = pos_scalars[:, self.units:self.units * 2].reshape(n_atoms * self.units, 1)
        pos_factors_3 = pos_scalars[:, self.units * 2:self.units * 3].reshape(n_atoms * self.units, 1)

        mat_factors = matrix_scalars[:, :self.units].reshape(n_atoms * self.units, 1)
        mat_factors_2 = matrix_scalars[:, self.units:self.units * 2].reshape(n_atoms * self.units, 1)
        mat_factors_3 = matrix_scalars[:, self.units * 2:self.units * 3].reshape(n_atoms * self.units, 1)

        scaling_factors = torch.softmax(torch.cat([mat_factors, mat_factors_2, mat_factors_3], dim=-1), dim=-1)
        S1, S2, S3 = scaling_factors[:, 0][:, None, None], scaling_factors[:, 1][:, None, None], scaling_factors[:, 2][:, None, None]

        X_tens = S1 * self.symm_tensor(X_tens).view(-1, 3, 3) + S2 * self.symm_tensor(X_vec_tens).view(-1, 3,3) + S3 * self.symm_tensor(X_scal_tens).view(-1, 3, 3)

        cov_final = X_tens.view(-1, 3, 3)
        cov_final = self.construct_pos_def(cov_final)

        if self.config['use_pos_disp_functions']:
            pos_disp_final = pos_disp*torch.exp(pos_factors) + pos_disp_2*torch.square(pos_factors_2) + pos_disp_3 * pos_factors_3
        else:
            pos_disp_final = pos_disp + pos_disp_2 + pos_disp_3
        if self.config['energy_readout_mode'] == "AW":
            en_input = torch.cat([X_scal_all, X_vec_scal_all, X_tens_scal_all], dim=-1)
        else:
            """
            en_input =  torch.cat((X_scal, X_vec_scal, X_tens_scal), dim=-1).view(n_atoms, -1)
        energy_nodewise = self.phi_energy(en_input)
        energy_n_summed = torch.sum(energy_nodewise, dim=0)
        energy = self.rho_energy(energy_n_summed)
            """
            en_input = torch.cat((X_scal, X_vec_scal, X_tens_scal), dim=-1).view(n_atoms, -1)

        energy_nodewise = self.energy_mlp(en_input)
        energy = energy_nodewise.sum(dim=0)

        result = {"cov": cov_final,
                  "pos_disp": pos_disp_final,
                  "weights": weights_sm,
                  "scal_mults": scal_mults_th,
                  "n_multiples": n_multiples,
                  "energy": energy}

        end = time.time()
        if self.config['wandb']:
            wandb.log({"Forward Time": end - start})
        else:
            print("Forward Time: ", end - start)
        return result

    def symm_tensor(self, matrix):
        #denom = torch.norm(matrix, dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
        #denom = torch.where(denom == 0, torch.tensor(1.0, device=denom.device), denom)
        symm_matrix = torch.matmul(matrix, matrix.transpose(-1, -2)) #/ denom
        return symm_matrix

    def normalize_vector(self, vector):
        vector = vector.reshape(-1, 3)
        denom = torch.norm(vector, dim=-1, keepdim=True)
        #denom = denom.mean()
        # Use torch.where to handle the zero values without breaking gradient flow
        denom = torch.where(denom == 0, torch.tensor(1.0, device=denom.device), denom)
        return vector / denom

    def get_shell_number(self, atom_numbers):
        orbital_map = {
            1: 1,  # H: 1s
            6: 2,  # C: 1s, 2s, 2px, 2py, 2pz
            7: 2,  # N: 1s, 2s, 2px, 2py, 2pz
            8: 2,  # O: 1s, 2s, 2px, 2py, 2pz
            9: 2,  # F: 1s, 2s, 2px, 2py, 2pz
        }
        return torch.tensor([orbital_map[int(atom)] for atom in atom_numbers], device=self.device)

    def normalize_matrix(self, matrix):
        orig_shape = matrix.shape
        matrix = matrix.view(-1, 3, 3)
        denom = torch.norm(matrix, dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
        denom = torch.where(denom == 0, torch.tensor(1.0, device=denom.device), denom)
        #denom = denom.mean()
        matrix = matrix / denom
        matrix = matrix.view(orig_shape)
        return matrix

    def remove_vector_component(self, matrix):
        vec_part = (1/2) * (matrix - matrix.transpose(-1, -2))
        return matrix - vec_part

    def slice(self, tensor, n_multiples):
        return torch.cat([tensor[i, torch.arange(tensor.size(1))[:num_selections]] for i, num_selections in enumerate(n_multiples)])

    def construct_pos_def(self, matrix: torch.tensor):
        """
        Make a non-positive definite matrix positive definite by adding a small epsilon to its diagonal.

        Args:
            matrix (torch.Tensor): Input matrix.

        Returns:
            torch.Tensor: Positive definite matrix.
        """
        eps = 1e-6  # Small epsilon
        matrix += torch.eye(3, device=self.device).unsqueeze(0).expand(matrix.shape) * eps
        return matrix


    def configure_optimizers(self):
        optimizer = set_optimizer(self, self.config)
        anneal_milestones = list(self.config['lr_dec_every'] * np.arange(1, 500))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                           milestones=anneal_milestones,
                                                          gamma=self.config['lr_gamma'])
        return [optimizer], [lr_scheduler]

    def training_step(self,
                      batch,
                      batch_idx: int):
        qm9_density, qm9_mol, qm9_n_elec, qm9_grid_dict, filename, qm9_energy = batch[0]
        qm9_mol.pbc = False
        pos_grid = grid2pos(qm9_mol, qm9_grid_dict)
        pos_grid = pos_grid.to(self.device)
        atom_positions = torch.tensor(qm9_mol.positions, device=self.device)
        start = time.time()
        result = self(qm9_mol, n_elec=qm9_n_elec)

        weights = result['weights']
        pos_disp = result['pos_disp']
        X = result['cov']
        scalar_mults = result['scal_mults']
        n_multiples = result['n_multiples']
        predicted_energy = result['energy']
        if not self.config['negative_contributions']:
            scalar_mults = None
        weights = weights.reshape(weights.shape[0], 1, 1)

        n_multiples = n_multiples.to(self.device)
        atom_positions = atom_positions.to(self.device)
        atom_positions = atom_positions.reshape(n_multiples.shape[0], -1)
        pos_tens_orig = atom_positions.repeat_interleave(n_multiples, 0).view(-1, 3)
        pos_tens_orig = pos_tens_orig.to(self.device)
        gaus_pos = pos_tens_orig + pos_disp
        if not self.config['energy_only']:
            sys_dist = gto(
                pos=atom_positions,
                pos_disp=pos_disp,
                cov=X,
                scales=weights,
                scalar_mults=scalar_mults,
                n_multiples=n_multiples,
                relu=self.config['relu'],
                cut_distance=self.config['inference_cutoff'],
                use_links=self.config['use_links'],
                device=self.device)
            if not self.config['train_sample_all']:
                density, sampled_points = get_n_points_dens(atom_dist=sys_dist,
                                                            pos_grid=pos_grid,
                                                            gaus_pos=gaus_pos,
                                                            n_multiples=result['n_multiples'],
                                                            n_points=self.config['n_train_points'],
                                                            sample_all=self.config['train_sample_all'])
            else:
                if pos_grid.flatten().size()[0] > self.config['max_grid_points']:
                    density, _ = get_n_points_dens(atom_dist=sys_dist,
                                                   pos_grid=pos_grid,
                                                   gaus_pos=gaus_pos,
                                                   n_multiples=result['n_multiples'],
                                                   n_points=self.config['n_split_points'],
                                                   sample_all=True)
                else:
                    density = get_full_grid_dens(atom_dist=sys_dist,
                                                 gaus_pos=gaus_pos,
                                                 n_multiples=result['n_multiples'],
                                                 pos_grid=pos_grid)
                sampled_points = None

            density, n_val_elec = normalize_density(n_elec=qm9_n_elec,
                                                    density=density,
                                                    qm9_density=qm9_density,
                                                    grid_dict=qm9_grid_dict,
                                                    sys=qm9_mol,
                                                    points=sampled_points)
        end = time.time()
        if self.config['wandb']:
            wandb.log({"Full Inference Time TRAIN": end - start})
        else:
            print("Full Inference Time TRAIN: ", end - start)
        if not self.config["energy_only"]:
            loss, dens_error = self.loss.compute_total_loss(
                sys=qm9_mol,
                pred_cd=density,
                true_cd_tens=qm9_density,
                grid_dict=qm9_grid_dict,
                n_valence_electrons=n_val_elec,
                sampled_points=sampled_points,
                training=True
            )
        else:
            loss = torch.tensor(0.0, device=self.device)
            dens_error = torch.tensor(0.0, device=self.device)
        if self.config["square_dens_loss"]:
            loss = torch.square(loss)
        if self.config['multimodal']:
            energy_loss, energy_error, energy_error_kcalmol = self.loss.compute_energy_loss(
                predicted_energy=predicted_energy,
                true_energy=qm9_energy,
                training=True)
            if self.config["square_energy_loss"]:
                energy_loss = torch.square(energy_loss)
            if torch.isnan(energy_loss):
                raise ValueError(f"Energy loss is NaN for molecule {qm9_mol.get_chemical_formula()}. "
                                 f"Predicted energy: {predicted_energy}, Target energy: {qm9_energy}")
            if torch.isnan(loss):
                raise ValueError(f"Density loss is NaN for molecule {qm9_mol.get_chemical_formula()}.")
            if self.config['energy_only']:
                loss = energy_loss
            else:
                if self.config["square_loss_together"]:
                    loss = torch.square(loss + energy_loss)
                else:
                    loss = self.config["dens_coeff"]*loss + self.config["energy_coeff"]*energy_loss
            self.log("TE [H]",
                     np.round(energy_error.detach().cpu().numpy(), 4),
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True,
                     batch_size=1)
            self.log("TE [kcal/mol]",
                     np.round(energy_error_kcalmol.detach().cpu().numpy(), 2),
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True,
                     batch_size=1)

        self.log("TD NMAE [%]",
                 np.round(100 * dens_error.detach().cpu().numpy(), 2),
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=1)
        self.log("T Loss",
                 np.round(loss.detach().cpu().numpy(), 3),
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=1)

        if self.config['hpc']:
            if self.config['multimodal']:
                wandb.log({"Train Density NMAE [%]": np.round(100 * dens_error.detach().cpu().numpy(), 2)})
                wandb.log({"Train Energy Err [H]": np.round(energy_error.detach().cpu().numpy(), 4)})
                wandb.log({"Training Combined Loss": np.round(loss.detach().cpu().numpy(), 3)})
                wandb.log({"Train Energy Err kcal/mol": np.round(energy_error_kcalmol.detach().cpu().numpy(), 2)})
            else:
                wandb.log({"Train Density NMAE [%]": np.round(100 * dens_error.detach().cpu().numpy(), 2)})
                wandb.log({"Training Loss": np.round(loss.detach().cpu().numpy(), 3)})
        if batch_idx < self.config['n_warmups']:
            self.optimizers().optimizer.param_groups[0]['lr'] = self.config['final_lr'] *((batch_idx+1)/self.config['n_warmups'])
        self.train_steps += 1
        if self.train_steps % self.config['save_every'] == 0:
            self.model_handler.save(self)
        return loss

    def validation_step(self,
                        batch,
                        batch_idx: int):
        qm9_density, qm9_mol, qm9_n_elec, qm9_grid_dict, mol_cd_file, qm9_energy = batch[0]
        qm9_mol.pbc = False
        pos_grid = grid2pos(qm9_mol, qm9_grid_dict)
        pos_grid = pos_grid.to(self.device)
        atom_positions = torch.tensor(qm9_mol.positions, device=self.device)
        start = time.time()
        result = self(qm9_mol, n_elec=qm9_n_elec)

        weights = result['weights']
        pos_disp = result['pos_disp']
        X = result['cov']
        scalar_mults = result['scal_mults']
        n_multiples = result['n_multiples']
        predicted_energy = result['energy']
        if not self.config['negative_contributions']:
            scalar_mults = None
        weights = weights.reshape(weights.shape[0], 1, 1)

        n_multiples = n_multiples.to(self.device)
        atom_positions = atom_positions.to(self.device)
        atom_positions = atom_positions.reshape(n_multiples.shape[0], -1)
        pos_tens_orig = atom_positions.repeat_interleave(n_multiples, 0).view(-1, 3)
        pos_tens_orig = pos_tens_orig.to(self.device)
        gaus_pos = pos_tens_orig + pos_disp
        if not self.config['energy_only']:
            sys_dist = gto(
                pos=atom_positions,
                pos_disp=pos_disp,
                cov=X,
                scales=weights,
                scalar_mults=scalar_mults,
                n_multiples=result['n_multiples'],
                relu=self.config['relu'],
                cut_distance=self.config['inference_cutoff'],
                use_links=self.config['use_links'],
                device=self.device)
            if pos_grid.flatten().size()[0] > self.config['max_grid_points']:
                density, _ = get_n_points_dens_single(atom_dist=sys_dist,
                                               pos_grid=pos_grid,
                                               gaus_pos=gaus_pos,
                                               n_multiples=result['n_multiples'],
                                               n_points=self.config['n_split_points_single'],
                                               sample_all=True)
            else:
                density = get_full_grid_dens(atom_dist=sys_dist,
                                             gaus_pos=gaus_pos,
                                             n_multiples=result['n_multiples'],
                                             pos_grid=pos_grid)
            sampled_points = None

            density, n_val_elec = normalize_density(n_elec=qm9_n_elec,
                                                    density=density,
                                                    qm9_density=qm9_density,
                                                    grid_dict=qm9_grid_dict,
                                                    sys=qm9_mol,
                                                    points=sampled_points)
        end = time.time()
        if self.config['wandb']:
            if wandb.run is not None:
                wandb.log({"Full Inference Time VAL": end - start})
        else:
            print("Full Inference Time VAL:", end - start)
        if not self.config["energy_only"]:
            loss, density_error = self.loss.compute_total_loss(
                sys=qm9_mol,
                pred_cd=density,
                true_cd_tens=qm9_density,
                grid_dict=qm9_grid_dict,
                n_valence_electrons=n_val_elec,
                sampled_points=sampled_points,
                training=True
            )
        else:
            loss = torch.tensor(0.0, device=self.device)
            density_error = torch.tensor(0.0, device=self.device)
        if self.config["square_dens_loss"]:
            loss = torch.square(loss)
        if self.config['multimodal']:
            energy_loss, energy_error, energy_error_kcalmol = self.loss.compute_energy_loss(predicted_energy=predicted_energy,
                                                                      true_energy=qm9_energy, training=False)
            if self.config["square_energy_loss"]:
                energy_loss = torch.square(energy_loss)
            if self.config['energy_only']:
                loss = energy_loss
            else:
                if self.config["square_loss_together"]:
                    loss = torch.square(loss + energy_loss)
                else:
                    loss = self.config["dens_coeff"]*loss + self.config["energy_coeff"]*energy_loss
            self.log("VE [H]",
                     np.round(energy_error.detach().cpu().numpy(), 4),
                     on_step=True,
                     on_epoch=True,
                     prog_bar=False,
                     logger=True,
                     batch_size=1)
            self.log("Validation Energy Err kcal/mol",
                     np.round(energy_error_kcalmol.detach().cpu().numpy(), 2),
                     on_step=True,
                     on_epoch=True,
                     prog_bar=False,
                     logger=True,
                     batch_size=1)
        self.log("VD NMAE [%]",
                 np.round(100 * density_error.detach().cpu().numpy(), 2),
                 on_step=True,
                 on_epoch=True,
                 prog_bar=False,
                 logger=True,
                 batch_size=1)

        self.log("V Loss",
                 np.round(loss.detach().cpu().numpy(), 3),
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=1)
        if self.config['hpc']:
            if self.config['multimodal']:
                wandb.log({"Validation Density NMAE [%]": np.round(100 * density_error.detach().cpu().numpy(), 2)})
                wandb.log({"Validation Energy Err [H]": np.round(energy_error.detach().cpu().numpy(), 4)})
                wandb.log({"Validation Combined Loss": np.round(loss.detach().cpu().numpy(), 3)})
                wandb.log({"Validation Energy Err kcal/mol": np.round(energy_error_kcalmol.detach().cpu().numpy(), 2)})
            else:
                wandb.log({"Validation Density NMAE [%]": np.round(100 * density_error.detach().cpu().numpy(), 2)})
                wandb.log({"Validation Loss": np.round(loss.detach().cpu().numpy(), 3)})
        if batch_idx == 0 and self.config['save_model']:
            self.best_val_loss = density_error
            self.model_handler.save(self)

        if self.config['construct_val_cd']:
            if density_error < self.val_error_dict[f"Val_file_{batch_idx}"]:
                qm9_path = os.path.join(self.qm9_dens_path, mol_cd_file)
                cd_to_chgcar(atoms=qm9_mol,
                             cd=density.detach().cpu().numpy(),
                             filename=f'{self.pred_dens_path_val}{mol_cd_file[-17:-11]}_{qm9_mol.get_chemical_formula()}_val_{batch_idx}.CHGCAR')
                create_chg_delta(
                    pred_dens_file=f'{self.pred_dens_path_val}{mol_cd_file[-17:-11]}_{qm9_mol.get_chemical_formula()}_val_{batch_idx}.CHGCAR',
                    true_dens_file=qm9_path,
                    delta_folder=self.density_delta_path_val,
                    name_iter_str=f'{mol_cd_file[-17:-11]}_{qm9_mol.get_chemical_formula()}_test_{batch_idx}'
                )
        if batch_idx == 0:
            self.val_iter += 1
        if self.plot_gaus_pos:
            pos = atom_positions.reshape(result['n_multiples'].shape[0], -1)
            pos = pos.repeat_interleave(result['n_multiples'], 0).view(-1, 3)
            gaus_pos = pos_disp + pos
            origin_atoms = qm9_mol.get_chemical_symbols()
            origin_atoms = np.repeat(origin_atoms, result['n_multiples'].detach().cpu().numpy())
            self.plot_gaussian_positions(atom_types=qm9_mol.get_chemical_symbols(),
                                         scal_mults=scalar_mults,
                                         weights=weights,
                                         atom_positions=atom_positions,
                                        gaus_positions=gaus_pos,
                                         origin_atoms=origin_atoms,
                                         origin_pos=pos,
                                         covariance_matrices=X,
                                     filename=f'{self.gaus_pos_path_val}_iter_{self.val_iter}_{mol_cd_file[-17:-11]}_{qm9_mol.get_chemical_formula()}.html',
                                         simple=self.simple_gaus
                                         )

        return loss

    def test_step(self,
                        batch,
                        batch_idx: int):
        qm9_density, qm9_mol, qm9_n_elec, qm9_grid_dict, mol_cd_file, qm9_energy = batch[0]
        qm9_mol.pbc = False
        pos_grid = grid2pos(qm9_mol, qm9_grid_dict)
        pos_grid = pos_grid.to(self.device)
        atom_positions = torch.tensor(qm9_mol.positions, device=self.device)
        start = time.time()
        result = self(qm9_mol, n_elec=qm9_n_elec)

        weights = result['weights']
        pos_disp = result['pos_disp']
        X = result['cov']
        scalar_mults = result['scal_mults']
        n_multiples = result['n_multiples']
        predicted_energy = result['energy']
        if not self.config['negative_contributions']:
            scalar_mults = None
        weights = weights.reshape(weights.shape[0], 1, 1)

        n_multiples = n_multiples.to(self.device)
        atom_positions = atom_positions.to(self.device)
        atom_positions = atom_positions.reshape(n_multiples.shape[0], -1)
        pos_tens_orig = atom_positions.repeat_interleave(n_multiples, 0).view(-1, 3)
        pos_tens_orig = pos_tens_orig.to(self.device)
        gaus_pos = pos_tens_orig + pos_disp
        if not self.config['energy_only']:
            sys_dist = gto(
                pos=atom_positions,
                pos_disp=pos_disp,
                cov=X,
                scales=weights,
                scalar_mults=scalar_mults if self.config['negative_contributions'] else None,
                n_multiples=result['n_multiples'],
                relu=self.config['relu'],
                cut_distance=self.config['inference_cutoff'],
                use_links=self.config['use_links'],
                device=self.device)

            density, _ = get_n_points_dens_single(atom_dist=sys_dist,
                                           pos_grid=pos_grid,
                                           gaus_pos=gaus_pos,
                                           n_multiples=result['n_multiples'],
                                           n_points=self.config['n_split_points_single'],
                                           sample_all=True)

            sampled_points = None

            density, n_val_elec = normalize_density(n_elec=qm9_n_elec,
                                                    density=density,
                                                    qm9_density=qm9_density,
                                                    grid_dict=qm9_grid_dict,
                                                    sys=qm9_mol,
                                                    points=sampled_points)
        end = time.time()

        if self.config['wandb']:
            wandb.log({"Full Inference Time TEST": end - start})
        else:
            print("Full Inference Time TEST: ", end - start)
        if not self.config["energy_only"]:
            loss, density_error = self.loss.compute_total_loss(
                sys=qm9_mol,
                pred_cd=density,
                true_cd_tens=qm9_density,
                grid_dict=qm9_grid_dict,
                n_valence_electrons=n_val_elec,
                sampled_points=sampled_points,
                training=True
            )
        else:
            loss = torch.tensor(0.0, device=self.device)
            density_error = torch.tensor(0.0, device=self.device)
        if self.config["square_dens_loss"]:
            loss = torch.square(loss)
        if self.config['multimodal']:
            energy_loss, energy_error, energy_error_kcalmol = self.loss.compute_energy_loss(
                predicted_energy=predicted_energy,
                true_energy=qm9_energy,
                training=False)
            if self.config["square_energy_loss"]:
                energy_loss = torch.square(energy_loss)
            if self.config['energy_only']:
                loss = energy_loss
            else:
                if self.config["square_loss_together"]:
                    loss = torch.square(loss + energy_loss)
                else:
                    loss = self.config["dens_coeff"]*loss + self.config["energy_coeff"]*energy_loss

            self.log("Test Energy Err [H]",
                     np.round(energy_error.detach().cpu().numpy(), 4),
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True,
                     batch_size=1)
            self.log("Test Energy Err [kcal/mol]",
                     np.round(energy_error_kcalmol.detach().cpu().numpy(), 2),
                     on_step=True,
                     on_epoch=True,
                     prog_bar=False,
                     logger=True,
                     batch_size=1)
            self.log("Test Combined Loss",
                     np.round(loss.detach().cpu().numpy(), 3),
                     on_step=True,
                     on_epoch=True,
                     prog_bar=False,
                     logger=True,
                     batch_size=1)
        self.log("Test Density NMAE [%]",
                 np.round(100 * density_error.detach().cpu().numpy(), 2),
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=1)

        if self.config['hpc']:
            if self.config['multimodal']:
                wandb.log({"Test Density NMAE [%]": np.round(100 * density_error.detach().cpu().numpy(), 2)})
                wandb.log({"Test Energy Err [H]": np.round(energy_error.detach().cpu().numpy(), 4)})
                wandb.log({"Test Combined Loss": np.round(loss.detach().cpu().numpy(), 3)})
                wandb.log({"Test Energy Err [kcal/mol]": np.round(energy_error_kcalmol.detach().cpu().numpy(), 2)})
            else:
                wandb.log({"Test Density NMAE [%]": np.round(100 * density_error.detach().cpu().numpy(), 2)})
                wandb.log({"Test Loss": np.round(loss.detach().cpu().numpy(), 3)})

        if self.config['construct_test_cd'] and batch_idx % 100 == 0:
            qm9_path = os.path.join(self.qm9_dens_path, mol_cd_file)
            cd_to_chgcar(atoms=qm9_mol,
                         cd=density.detach().cpu().numpy(),
                         filename=f'{self.pred_dens_path_test}{mol_cd_file[-17:-11]}_{qm9_mol.get_chemical_formula()}_test_{batch_idx}.CHGCAR')
            create_chg_delta(
                pred_dens_file=f'{self.pred_dens_path_test}{mol_cd_file[-17:-11]}_{qm9_mol.get_chemical_formula()}_test_{batch_idx}.CHGCAR',
                true_dens_file=qm9_path,
                delta_folder=self.density_delta_path_test,
                name_iter_str=f'{mol_cd_file[-17:-11]}_{qm9_mol.get_chemical_formula()}_test_{batch_idx}'
            )
        if self.plot_gaus_pos:
            pos = atom_positions.reshape(result['n_multiples'].shape[0], -1)
            pos = pos.repeat_interleave(result['n_multiples'], 0).view(-1, 3)
            gaus_pos = pos_disp + pos
            origin_atoms = qm9_mol.get_chemical_symbols()
            origin_atoms = np.repeat(origin_atoms, result['n_multiples'].detach().cpu().numpy())
            self.plot_gaussian_positions(atom_types=qm9_mol.get_chemical_symbols(),
                                         scal_mults=scalar_mults if self.config['negative_contributions'] else None,
                                         weights=weights,
                                         atom_positions=atom_positions,
                                         gaus_positions=gaus_pos,
                                         origin_atoms=origin_atoms,
                                         origin_pos=pos,
                                         covariance_matrices=X,
                                     filename=f'{self.gaus_pos_path_test}{mol_cd_file[-17:-11]}_{qm9_mol.get_chemical_formula()}_batchidx_{batch_idx}.html',
                                         simple=self.simple_gaus
                                         )
        return loss

    def eq_check_prediction(self,
                batch,
                 r_mat = None,
                translation = None,
                inversion = False):

        qm9_density, qm9_mol, qm9_n_elec, qm9_grid_dict, filename = batch[0]
        qm9_mol.pbc = False
        pos_grid = grid2pos(qm9_mol, qm9_grid_dict).to(self.device)
        grid_nonrot = torch.tensor(pos_grid)
        atom_positions_original = torch.tensor(qm9_mol.positions.copy(), device=self.device, dtype=torch.float)

        result_nontransformed = self(qm9_mol.copy(), n_elec=qm9_n_elec, rotation_matrix=r_mat)

        weights_nontransformed = result_nontransformed['weights']
        pos_disp_nontransformed = result_nontransformed['pos_disp']
        X_nontransformed = result_nontransformed['cov']
        scalar_mults_nontransformed = result_nontransformed['scal_mults']

        n_multiples_original = result_nontransformed['n_multiples']
        n_multiples_original = n_multiples_original.to(self.device)
        atom_positions_original = atom_positions_original.to(self.device)
        atom_positions_original = atom_positions_original.reshape(n_multiples_original.shape[0], -1)
        pos_tens_orig_original = atom_positions_original.repeat_interleave(n_multiples_original, 0).view(-1, 3)
        pos_tens_orig_original = pos_tens_orig_original.to(self.device)
        gaus_pos_original = pos_tens_orig_original + pos_disp_nontransformed

        if r_mat is not None:
            positions_original = torch.tensor(atom_positions_original)
            positions_original = positions_original.to(self.device)
            atom_positions = (r_mat @ positions_original.T).T
            qm9_mol.positions = atom_positions.cpu().numpy()
            grid_nonrot = torch.tensor(pos_grid)
            pos_grid = self.rotate_vector(pos_grid.view(-1, 3), r_mat).view(pos_grid.shape)
            result_transformed = self(atoms=qm9_mol,
                                      n_elec=qm9_n_elec,
                                      rotation_matrix=r_mat,
                                      orig_pos=positions_original)
        if translation is not None:
            positions_original = torch.tensor(atom_positions_original)
            positions_original = positions_original.to(self.device)
            atom_positions = positions_original + translation
            qm9_mol.positions = atom_positions.cpu().numpy()
            pos_grid = pos_grid + translation
            result_transformed = self(atoms=qm9_mol,
                                      n_elec=qm9_n_elec,
                                      rotation_matrix=r_mat,
                                      orig_pos=positions_original)

        if inversion:
            positions_original = torch.tensor(atom_positions_original)
            positions_original = positions_original.to(self.device)
            atom_positions = -positions_original
            qm9_mol.positions = atom_positions.cpu().numpy()
            pos_grid = -pos_grid
            result_transformed = self(atoms=qm9_mol,
                                      n_elec=qm9_n_elec,
                                      rotation_matrix=r_mat,
                                      orig_pos=positions_original)
        weights_transformed = result_transformed['weights']
        pos_disp_transformed = result_transformed['pos_disp']
        X_transformed = result_transformed['cov']
        scalar_mults_transformed = result_transformed['scal_mults']
        n_multiples_transformed  = result_nontransformed['n_multiples']
        n_multiples_transformed = n_multiples_transformed.to(self.device)
        atom_positions = atom_positions.to(self.device)
        atom_positions = atom_positions.reshape(n_multiples_transformed.shape[0], -1)
        pos_tens_orig_transformed = atom_positions.repeat_interleave(n_multiples_transformed, 0).view(-1, 3)
        pos_tens_orig_transformed = pos_tens_orig_transformed.to(self.device)
        gaus_pos_transformed = pos_tens_orig_transformed + pos_disp_transformed

        if r_mat is not None or inversion or translation is not None:
            self.check_equivariance_scalar(scalar_transformed=weights_transformed, orig_scalar=weights_nontransformed)
            self.check_equivariance_scalar(scalar_transformed=scalar_mults_transformed, orig_scalar=scalar_mults_nontransformed)
            self.check_equivariance_vector(vector_transformed=pos_disp_transformed, orig_vector=pos_disp_nontransformed,
                                           r_mat=r_mat if r_mat is not None else None,
                                           translation=translation if translation is not None else None,
                                           inversion=inversion)
            self.check_equivariance_matrix(matrix_transformed=X_transformed, orig_matrix=X_nontransformed,
                                           r_mat=r_mat if r_mat is not None else None,
                                           translation=translation if translation is not None else None,
                                           inversion=inversion)


        sys_dist = gto(
            pos=atom_positions_original,
            pos_disp=pos_disp_nontransformed,
            cov=X_nontransformed,
            scales=weights_nontransformed.reshape(weights_nontransformed.shape[0], 1, 1),
            scalar_mults=scalar_mults_nontransformed if self.config['negative_contributions'] else None,
            n_multiples=result_nontransformed['n_multiples'],
            relu=self.config['relu'],
            cut_distance=self.config['inference_cutoff'],
            use_links=self.config['use_links'],
            device=self.device)

        density_original, _ = get_n_points_dens(atom_dist=sys_dist,
                                       pos_grid=grid_nonrot,
                                       gaus_pos=gaus_pos_original,
                                       n_multiples=result_nontransformed['n_multiples'],
                                       n_points=self.config['n_split_points'],
                                       sample_all=True,
                                       r_mat=r_mat,
                                       grid_nonrot=grid_nonrot)

        density_original, n_val_elec = normalize_density(n_elec=qm9_n_elec,
                                                density=density_original,
                                                qm9_density=qm9_density,
                                                grid_dict=qm9_grid_dict,
                                                sys=qm9_mol,
                                                points=None)

        if r_mat is not None or inversion or translation is not None:
            sys_dist = gto(
                pos=atom_positions,
                pos_disp=pos_disp_transformed,
                cov=X_transformed,
                scales=weights_transformed.reshape(weights_transformed.shape[0], 1, 1),
                scalar_mults=scalar_mults_transformed if self.config['negative_contributions'] else None,
                n_multiples=result_transformed['n_multiples'],
                relu=self.config['relu'],
                cut_distance=self.config['inference_cutoff'],
                use_links=self.config['use_links'],
                device=self.device)

            density_transformed, _ = get_n_points_dens(atom_dist=sys_dist,
                                                    pos_grid=pos_grid,
                                                    gaus_pos=gaus_pos_transformed,
                                                    n_multiples=result_transformed['n_multiples'],
                                                    n_points=self.config['n_split_points'],
                                                    sample_all=True)

            density_transformed, n_val_elec = normalize_density(n_elec=qm9_n_elec,
                                                             density=density_transformed,
                                                             qm9_density=qm9_density,
                                                             grid_dict=qm9_grid_dict,
                                                             sys=qm9_mol,
                                                             points=None)

            return density_original, density_transformed
        else:
            return density_original
    def rotate_vector(self, vector, r_mat):
        rotated = (r_mat @ vector.T).T
        return rotated

    def rotate_matrix(self, matrix, r_mat):
        rotation_matrix_inv_expanded = torch.inverse(r_mat).unsqueeze(0).unsqueeze(0)
        rotation_matrix_expanded = r_mat.unsqueeze(0).unsqueeze(0)
        intermediate_tensor = torch.matmul(rotation_matrix_expanded, matrix)
        rotated_tensor = torch.matmul(intermediate_tensor, rotation_matrix_inv_expanded).squeeze(0)
        return rotated_tensor

    def check_equivariance_vector(self, vector_transformed, orig_vector, r_mat, translation, inversion):
        if r_mat is not None:
            transformed_original_vector = self.rotate_vector(orig_vector, r_mat)
        elif inversion:
            transformed_original_vector = -orig_vector
        elif translation is not None:
            transformed_original_vector = orig_vector
        equivariant = torch.allclose(transformed_original_vector, vector_transformed, atol=1e-2)
        print(f"Vector equivariance check passed: {equivariant}")

    def check_equivariance_matrix(self, matrix_transformed, orig_matrix, r_mat, translation, inversion):
        if r_mat is not None:
            transformed_original_matrix = self.rotate_matrix(orig_matrix, r_mat)
        elif inversion:
            transformed_original_matrix = orig_matrix
        elif translation is not None:
            transformed_original_matrix = orig_matrix
        equivariant = torch.allclose(transformed_original_matrix, matrix_transformed, atol=1e-2)
        print(f"Matrix equivariance check passed: {equivariant}")

    def check_equivariance_scalar(self, scalar_transformed, orig_scalar):
        equivariant = torch.allclose(orig_scalar, scalar_transformed, atol=1e-2)
        print(f"Scalar equivariance check passed: {equivariant}")

    def init_hotpp_model(self):
        elements = ['H', 'O', 'C', 'N', 'F', 'Cl', 'Br', 'I']
        element_numbers = []
        for element in elements:
            element_numbers.append(atomic_numbers[element])
        if self.config['embed_type'] == 'aufbau':
            emb_layer = AufbauEmbedding(element_numbers, self.config['master_units'], device=self.device)
        elif self.config['embed_type'] == 'atomic':
            emb_layer = AtomicEmbedding(element_numbers, self.config['master_units'])
        emb_layer.to(self.device)
        self.emb_layer = emb_layer
        cut_fn = PolynomialCutoff(cutoff=self.config['hotpp_cutoff'], p=13)
        radial_fn = BesselPoly(r_max=self.config['hotpp_cutoff'], n_max=self.config['master_units'], cutoff_fn=cut_fn)
        nlayers = self.config['hotpp_nlayers']
        if self.config['miaonet'] == "miaomiao":
            self.base_model = MiaoMiaoNet(embedding_layer=emb_layer,
                                      radial_fn=radial_fn,
                                      n_layers=nlayers,
                                      max_n_body=[self.config['n_max']]*nlayers,
                                      max_r_way=[self.config['r_max']] * nlayers,
                                      max_out_way=[self.config['out_max']] * nlayers,
                                      max_out_heads=self.config['max_out_heads'],
                                      output_dim=[self.config['hotpp_outdim']] * nlayers,
                                      activate_fn=self.config['activation_type'],
                                      head_activate_list=self.config['head_activate_list'],
                                      mean=0.,
                                      std=1.,
                                      norm_factor=1.,
                                      bilinear=False,
                                      norm_heads=self.config['norm_heads'],
                                     norm_blocks=self.config['norm_blocks'],
                                      conv_mode=self.config['conv_mode'],
                                      update_edge=self.config['update_edge'],
                                     prune=self.config['prune'])
        else:
            self.base_model = MiaoNet(embedding_layer=emb_layer,
                              radial_fn=radial_fn,
                              n_layers=nlayers,
                              max_r_way=[self.config['r_max']]*nlayers,
                              max_out_way=[self.config['out_max']]*nlayers,
                              max_out_heads=self.config['max_out_heads'],
                              output_dim=[self.config['hotpp_outdim']]*nlayers,
                              activate_fn=self.config['activation_type'],
                              head_activate_list=self.config['head_activate_list'],
                              mean=0.,
                              std=1.,
                              norm_factor=1.,
                              bilinear=False,
                              conv_mode=self.config['conv_mode'],
                              update_edge=self.config['update_edge'])

    def train_dataloader(self):
        # Create dataset and dataloader
        return DataLoader(QM9Dataset(self.train_files, self.qm9_dens_path, self.config),
                          batch_size=1,
                          shuffle=False,
                          collate_fn=collate_fn,
                          pin_memory=True,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(QM9Dataset(self.validation_files, self.qm9_dens_path, self.config),
                          batch_size=1,
                          shuffle=False,
                          collate_fn=collate_fn,
                          pin_memory=True,
                          num_workers=0)

    def test_dataloader(self):
        return DataLoader(QM9Dataset(self.test_files, self.qm9_dens_path, self.config),
                          batch_size=1,
                          shuffle=False,
                          collate_fn=collate_fn,
                          pin_memory=True,
                          num_workers=0)

    def plot_gaussian_positions(self,
                                atom_types,
                                scal_mults,
                                weights,
                                atom_positions,
                                gaus_positions,
                                covariance_matrices,
                                origin_atoms,
                                origin_pos,
                                filename,
                                simple=True,
                                ):
        # Convert tensors to numpy if needed
        if isinstance(atom_positions, torch.Tensor):
            atom_positions = atom_positions.detach().cpu().numpy()
        if isinstance(gaus_positions, torch.Tensor):
            gaus_positions = gaus_positions.detach().cpu().numpy()
        if isinstance(atom_types, torch.Tensor):
            atom_types = atom_types.detach().cpu().numpy()
        if isinstance(scal_mults, torch.Tensor):
            scal_mults = scal_mults.detach().cpu().numpy()
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        if isinstance(covariance_matrices, torch.Tensor):
            covariance_matrices = covariance_matrices.detach().cpu().numpy()

        # Define a mapping for atom types to colors
        atom_color_map = {
            'H': 'white',  # Hydrogen
            'C': 'black',  # Carbon
            'N': 'green',  # Nitrogen
            'O': 'red',  # Oxygen
            'F': 'brown',  # Fluorine
            'Cl': 'yellow',  # Chlorine
            'Br': 'brown',  # Bromine
            'I': 'purple'  # Iodine
        }
        atom_size_map = {
            'H': 20,  # Hydrogen
            'C': 40,  # Carbon
            'N': 40,  # Nitrogen
            'O': 45,  # Oxygen
            'F': 50,  # Fluorine
            'Cl': 55,  # Chlorine
            'Br': 60,  # Bromine
            'I': 65  # Iodine
        }

        # Get colors for atom types based on atomic numbers
        atom_colors = [atom_color_map.get(atom, 'gray') for atom in atom_types]
        atom_sizes = [atom_size_map.get(atom, 12) for atom in atom_types]  # Default size = 12 if atom type not in map
        hover_text_atoms = [
            f"Atom: {atom}<br>Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            for atom, pos in zip(atom_types, atom_positions)
        ]
        # Create the 3D scatter plot for atom positions
        trace_atoms = go.Scatter3d(
            x=atom_positions[:, 0],
            y=atom_positions[:, 1],
            z=atom_positions[:, 2],
            mode='markers',
            marker=dict(size=atom_sizes, color=atom_colors, line=dict(color='black', width=5), opacity=1.0),
            showlegend=False,
            hoverinfo='text',
            text=hover_text_atoms
        )

        # Determine colors based on sign of scal_mults
        if scal_mults is None:
            scal_mults = np.ones_like(weights)
        gaussian_colors = ['blue' if scal < 0 else 'red' for scal in scal_mults]

        # Gaussian hover text (position and origin info)
        hover_text_gaussians = [
            f"Position: ({gp[0]:.2f}, {gp[1]:.2f}, {gp[2]:.2f})<br>" +
            f"Origin: {origin_atoms[i]} at ({op[0]:.2f}, {op[1]:.2f}, {op[2]:.2f})"
            for i, (gp, op) in enumerate(zip(gaus_positions, origin_pos))
        ]

        # Plot each Gaussian as an ellipsoid
        if simple:
            # Create the 3D scatter plot for gaussian positions
            # Compute sizes for Gaussian points using absolute product of scal_mults and weights
            gaussian_sizes = abs(scal_mults * weights)  # Adjust size scaling factor if needed
            gaussian_sizes = (np.log(gaussian_sizes / (gaussian_sizes.min() +1)) + 1) * 3
            trace_gaussians = go.Scatter3d(
                x=gaus_positions[:, 0],
                y=gaus_positions[:, 1],
                z=gaus_positions[:, 2],
                mode='markers',
                marker=dict(size=gaussian_sizes, color=gaussian_colors, opacity=0.5),
                showlegend=False,
                text=hover_text_gaussians,
                hoverinfo='text',  # Set hoverinfo to show custom text]
            )
            trace_gaussians = [trace_gaussians]
        else:
            trace_gaussians = []
            for i, (center, cov_matrix) in enumerate(zip(gaus_positions, covariance_matrices)):
                # Get eigenvalues and eigenvectors for the covariance matrix
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                radii = np.sqrt(eigenvalues)  # Scale factors for each axis

                # Create a mesh grid of points for the ellipsoid in its local coordinates
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 10)
                x = radii[0] * np.outer(np.cos(u), np.sin(v))
                y = radii[1] * np.outer(np.sin(u), np.sin(v))
                z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

                # Transform points to align with eigenvectors and center at Gaussian position
                scale = 0.1
                ellipsoid_points = 0.1 * np.dot(eigenvectors, np.array([x.flatten(), y.flatten(), z.flatten()]))
                x_ellipsoid = ellipsoid_points[0].reshape(x.shape) + center[0]
                y_ellipsoid = ellipsoid_points[1].reshape(y.shape) + center[1]
                z_ellipsoid = ellipsoid_points[2].reshape(z.shape) + center[2]

                # Add ellipsoid as a surface plot to represent the Gaussian shape
                trace_gaussians.append(
                    go.Surface(
                        x=x_ellipsoid, y=y_ellipsoid, z=z_ellipsoid,
                        opacity=1.0, colorscale=[[0, gaussian_colors[i]], [1, gaussian_colors[i]]],
                        showscale=False, showlegend=False, hoverinfo='text',
                        text=hover_text_gaussians[i]
                    )
                )

        # Add a dummy trace for the atom color legend
        atom_legend_traces = []
        for atom, color in atom_color_map.items():
            if atom in atom_types:
                atom_legend_traces.append(
                    go.Scatter3d(
                        x=[None], y=[None], z=[None],
                        mode='markers',
                        marker=dict(size=16, color=color, line=dict(color='black', width=5)),
                        showlegend=True,
                        name=f'{atom}'
                    )
                )

        # Add dummy traces for gaussian sign and size legends
        gaussian_sign_legend = [
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=16, color='blue', opacity=0.5),
                showlegend=True,
                name="Negative Gaussians"
            ),
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=16, color='red', opacity=0.5),
                showlegend=True,
                name="Positive Gaussians"
            ),
        ]

        # Define layout
        layout = go.Layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            legend=dict(
                title="Legend",
                itemsizing='constant'
            )
        )

        # Create the figure
        fig = go.Figure(data=[trace_atoms] + atom_legend_traces + gaussian_sign_legend + trace_gaussians, layout=layout)

        # Save the interactive plot to an HTML file
        pio.write_html(fig, file=filename, auto_open=False)


def collate_fn(batch):
    return batch


class QM9Dataset(Dataset):
    def __init__(self, file_list, qm9_dens_path, config):
        self.file_list = file_list
        self.qm9_dens_path = qm9_dens_path
        with open(config['qm9_energy_path'], 'r') as f:
            self.energy_dict = json.load(f)
        self.cache = {}
        self.energy_only = config['energy_only']

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        valid = False
        while not valid:
            try:
                mol_cd_file = self.file_list[idx]
                qm9_path = os.path.join(self.qm9_dens_path, mol_cd_file)
                qm9idx = extract_index(qm9_path)
                if mol_cd_file in self.cache:
                    return self.cache[mol_cd_file]
                qm9_density, qm9_mol, qm9_n_elec, qm9_grid_dict = get_qm9_density(qm9_path)
                energy = self.energy_dict[str(qm9idx)]
                if self.energy_only:
                    self.cache[mol_cd_file] = (None, qm9_mol, qm9_n_elec, qm9_grid_dict, mol_cd_file, energy)
                else:
                    self.cache[mol_cd_file] = (qm9_density, qm9_mol, qm9_n_elec, qm9_grid_dict, mol_cd_file, energy)
                valid = True
            except Exception as e:
                idx = np.random.randint(0, len(self.file_list)-1)
        return self.cache[mol_cd_file]

def extract_index(path):
    match = re.search(r'/(\d+)\.CHGCAR\.lz4$', path)
    if match:
        return int(match.group(1))  # Convert to int to remove leading zeros
    return None  # Return None if no match is found



