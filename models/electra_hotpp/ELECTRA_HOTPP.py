from __future__ import annotations

import time

import wandb
import logging
from tools.atom_tools import valence_electrons
from ase.data import chemical_symbols
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from utils.model_handling import ModelIO, get_tag
from typing import Literal
from torch.utils.data import Dataset, DataLoader
import gc
import dgl
import matgl
import numpy as np
import torch
from models.ELECTRA.custom_cartesian_tensor import CartesianTensor
from e3nn.o3 import FullyConnectedTensorProduct
import plotly.graph_objs as go
import plotly.io as pio
from matgl.config import DEFAULT_ELEMENTS
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
)
from matgl.layers import (
    ActivationFunction,
    BondExpansion,
)
from models.ELECTRA.custom_graph_convolution import TensorNetInteraction
from matgl.utils.io import IOMixIn
from torch import nn

from models.dists import gto, get_full_grid_dens, get_n_points_dens
from models.ELECTRA._embedding import TensorEmbedding
from models.ELECTRA.ase_converter import Atoms2Graph
from modules.grid2pos import grid2pos
from tools.density_conversions import normalize_density, cd_to_chgcar, get_qm9_density
from tools.graph_tools import CollateFuncAtoms
from utils.train_helper_funcs import set_optimizer
from modules.loss import DensityLoss
import os
from tools.visualization import create_chg_delta
import lightning as L
from models.electra_hotpp.layer import AtomicEmbedding, BesselPoly, PolynomialCutoff, AufbauEmbedding, MLPPoly, \
    CosineCutoff, ChebyshevPoly
from models.electra_hotpp.model import MiaoNet, MiaoMiaoNet
from ase.data import atomic_numbers
from ase.neighborlist import neighbor_list
import torch.nn.functional as F

logger = logging.getLogger(__file__)


class ELECTRA_hotpp(L.LightningModule, IOMixIn):
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
        self.use_graph_directly = config['use_graph_directly']
        self.collate_func = CollateFuncAtoms(self.cutoff)
        self.g_converter = Atoms2Graph(cutoff=self.cutoff, element_types=('H', 'O', 'C', 'N', 'F', 'Cl', 'Br', 'I'))
        self.val_error_dict = {}
        self.val_iter = 0
        for i, file in enumerate(self.validation_files):
            name = f"Val_file_{i}"
            self.val_error_dict[name] = float("inf")

        self.weigh_com_by_mass = config['weigh_com_by_mass']
        # self.device = device
        self.cutoff = config['hotpp_cutoff']
        self.element_types = ['H', 'O', 'C', 'N', 'F', 'Cl', 'Br', 'I']
        dtype = matgl.float_th

        self.init_hotpp_model()
        self.plot_gaus_pos = config['plot_gaus_pos']

        irreps_out = f"{self.units}x0e + {self.units}x2e + {self.units}x0e + {self.units}x0e + {self.units}x1o"

        ## FIRST FCTP
        if not self.config['use_graph_directly']:
            self.fctp = FullyConnectedTensorProduct(
                irreps_in1=f"{self.units}x0e + {self.units}x2e",
                irreps_in2=f"{2 * self.units}x0e + {self.units}x1o",
                irreps_out=irreps_out,
                shared_weights=False)
            self.fctp_weights = torch.nn.Sequential(
                torch.nn.Linear(4 * self.units, 4 * self.units),
                torch.nn.ReLU(),
                torch.nn.Linear(4 * self.units, int(2 * self.units / self.config['reduce_weight_by'])),
                torch.nn.ReLU(),
                torch.nn.Linear(int(2 * self.units / self.config['reduce_weight_by']), self.fctp.weight_numel),
            )

        self.scal_mult_mlp_direct = nn.Sequential(
            nn.Linear(4 * self.units, self.units * 7),
            nn.ReLU(),
            nn.Linear(self.units * 7, self.units * 11)
        )
        ## ATOMIC TRANSFORM
        self.atomic_transform = nn.Sequential(
            nn.Linear(4 * self.units, self.units * 2),
            nn.ReLU(),
            nn.Linear(self.units * 2, self.units * 2),
        )


        self.tens_irrep = CartesianTensor("ij")
        self.sym_tens_irrep = CartesianTensor("ij=ji")
        self.vec_irrep = CartesianTensor("i")
        self.precision = torch.float32


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
        start = time.process_time()
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

        X_scal = X_scal_full[0]
        X_scal_vec = X_scal_full[1]
        X_scal_tens = X_scal_full[2]

        X_vec_scal = X_vec_full[0]
        X_vec = X_vec_full[1]
        X_vec_tens = X_vec_full[2]

        X_tens_scal = X_tens_full[0]
        X_tens_vec = X_tens_full[1]
        X_tens = X_tens_full[2]

        valence_tensor = torch.tensor([valence_electrons(chemical_symbols[atom]) for atom in at_numbers],
                                      device=self.device)

        # Compute the element-wise minimum between valence_tensor and fixed_value_tensor
        n_multiples = valence_tensor * self.units
        n_multiples = n_multiples.to(self.device)
        atom_embeds = init_emb.repeat_interleave(valence_tensor, dim=0)

        # Concatenate the selected slices along the first dimension
        n_atoms = torch.sum(valence_tensor)

        X_scal = self.slice(X_scal, n_multiples).view(n_atoms, self.units)
        X_scal_vec = self.slice(X_scal_vec, n_multiples).view(n_atoms, self.units, 3)
        X_scal_tens = self.slice(X_scal_tens, n_multiples).view(n_atoms, self.units, 3, 3)

        X_vec_scal = self.slice(X_vec_scal, n_multiples).view(n_atoms, self.units)
        X_vec = self.slice(X_vec, n_multiples).view(n_atoms, self.units, 3)
        X_vec_tens = self.slice(X_vec_tens, n_multiples).view(n_atoms, self.units, 3, 3)

        X_tens_scal = self.slice(X_tens_scal, n_multiples).view(n_atoms, self.units)
        X_tens_vec = self.slice(X_tens_vec, n_multiples).view(n_atoms, self.units, 3)
        X_tens = self.slice(X_tens, n_multiples).view(n_atoms, self.units, 3, 3)
        if self.config['remove_vector_components']:
            X_tens = self.remove_vector_component(X_tens)
            X_vec_tens = self.remove_vector_component(X_vec_tens)
            X_scal_tens = self.remove_vector_component(X_scal_tens)

        X_vec_scal_2 = X_vec_scal
        X_tens_scal_2 = X_tens_scal
        VMF_Vec = X_tens_vec.view(n_atoms * self.units, 3)
        pos_disp = X_vec.view(-1, 3)
        pos_disp_2 = X_scal_vec.view(-1, 3)

        #pos_disp = self.normalize_vector(pos_disp)
        #pos_disp_2 = self.normalize_vector(pos_disp_2)
        #VMF_Vec = self.normalize_vector(VMF_Vec)

        wsm_ = torch.cat((atom_embeds[:, :self.units], X_scal, X_vec_scal_2, X_tens_scal_2), dim=-1).view(n_atoms, -1)
        wsm = self.scal_mult_mlp_direct(wsm_)
        weights_sm = torch.softmax(wsm[:, :self.units].reshape(n_atoms * self.units), dim=0)
        scal_mults_th = wsm[:, self.units:self.units * 2].reshape(n_atoms * self.units)
        pos_factors = wsm[:, self.units * 2:self.units * 3].reshape(n_atoms * self.units, 1)
        pos_factors_2 = wsm[:, self.units * 3:self.units * 4].reshape(n_atoms * self.units, 1)
        scaling_factors_1 =  wsm[:, self.units * 4:self.units * 5].reshape(n_atoms * self.units, 1)
        scaling_factors_2 =  wsm[:, self.units * 5:self.units * 6].reshape(n_atoms * self.units, 1)
        scaling_factors_3 =  torch.abs(wsm[:, self.units * 6:self.units * 7].reshape(n_atoms * self.units, 1))
        kappa = wsm[:, self.units * 7:self.units * 8].reshape(n_atoms * self.units, 1)
        kappa2 = torch.nn.functional.softplus(wsm[:, self.units * 8:self.units * 9].reshape(n_atoms * self.units, 1))
        VMF_factors = wsm[:, self.units * 9:self.units * 10].reshape(n_atoms * self.units, 1)
        final_scaling = torch.exp(wsm[:, self.units * 10:self.units * 11].reshape(n_atoms * self.units, 1))
        scaling_factors = torch.softmax(torch.cat([scaling_factors_1, scaling_factors_2, scaling_factors_3], dim=-1), dim=-1)
        S1, S2, S3 = scaling_factors[:, 0][:, None, None], scaling_factors[:, 1][:, None, None], scaling_factors[:, 2][:, None, None]
        #S1, S2, S3 = torch.exp(scaling_factors_1)[:, None], scaling_factors_2[:, None], torch.log(torch.abs(scaling_factors_3))[:, None]
        X_tens = S1*self.normalize_matrix(self.symm_tensor(X_tens)).view(-1, 3, 3) + S2*self.normalize_matrix(self.symm_tensor(X_vec_tens)).view(-1, 3, 3) + S3*self.normalize_matrix(self.symm_tensor(X_scal_tens)).view(-1, 3, 3)

        cov_final = X_tens.view(n_atoms * self.units, 3, 3)
        cov_final = cov_final * final_scaling.view(-1, 1, 1)
        cov_final = self.construct_pos_def(cov_final)

        pos_disp_final = pos_disp*torch.exp(pos_factors) + pos_disp_2*pos_factors_2 + VMF_Vec * (VMF_factors**2)
        VMF_Vec = VMF_Vec * VMF_factors

        result = {"cov": cov_final,
                  "pos_disp": pos_disp_final,
                  "weights": weights_sm,
                  "scal_mults": scal_mults_th,
                  "n_multiples": n_multiples,
                  "kappa": kappa,
                  "kappa2": kappa2,
                  "VMF_vec": VMF_Vec.view(-1, 3)}

        end = time.process_time()
        if self.config['wandb']:
            wandb.log({"Forward Time": end - start})
        else:
            print("Forward Time: ", end - start)
        return result

    def symm_tensor(self, matrix):
        denom = torch.norm(matrix, dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
        denom = torch.where(denom == 0, torch.tensor(1.0, device=denom.device), denom)
        symm_matrix = torch.matmul(matrix, matrix.transpose(-1, -2)) / denom
        return symm_matrix

    def normalize_vector(self, vector):
        vector = vector.reshape(-1, 3)
        denom = torch.norm(vector, dim=-1, keepdim=True)
        #denom = denom.mean()
        # Use torch.where to handle the zero values without breaking gradient flow
        denom = torch.where(denom == 0, torch.tensor(1.0, device=denom.device), denom)
        return vector / denom

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
        start = time.process_time()
        qm9_density, qm9_mol, qm9_n_elec, qm9_grid_dict, filename = batch[0]
        qm9_mol.pbc = False
        pos_grid = grid2pos(qm9_mol, qm9_grid_dict).to(self.device)
        atom_positions = torch.tensor(qm9_mol.positions, device=self.device)

        result = self(qm9_mol, n_elec=qm9_n_elec)

        weights = result['weights']
        pos_disp = result['pos_disp']
        X = result['cov']
        scalar_mults = result['scal_mults']
        kappa = result['kappa']
        kappa2 = result['kappa2']
        VMF_vec = result['VMF_vec']
        n_multiples = result['n_multiples']
        if not self.config['negative_contributions']:
            scalar_mults = None
        scales_all_atoms = weights.reshape(weights.shape[0], 1, 1)

        n_multiples = n_multiples.to(self.device)
        atom_positions = atom_positions.to(self.device)
        atom_positions = atom_positions.reshape(n_multiples.shape[0], -1)
        pos_tens_orig = atom_positions.repeat_interleave(n_multiples, 0).view(-1, 3)
        pos_tens_orig = pos_tens_orig.to(self.device)
        gaus_pos = pos_tens_orig + pos_disp

        sys_dist = gto(
            pos=atom_positions,
            pos_disp=pos_disp,
            cov=X,
            kappa=kappa,
            kappa2=kappa2,
            VMF_vec=VMF_vec,
            use_vmf=self.config['use_vmf'],
            scales=scales_all_atoms,
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

        loss, dens_error = self.loss.compute_total_loss(
            sys=qm9_mol,
            pred_cd=density,
            true_cd_tens=qm9_density,
            grid_dict=qm9_grid_dict,
            n_valence_electrons=n_val_elec,
            sampled_points=sampled_points,
            training=True
        )
        end = time.process_time()
        if self.config['wandb']:
            wandb.log({"Full Inference Time": end - start})
        else:
            print("Full Inference Time: ", end - start)

        self.log("Train Density Err %",
                 np.round(100 * dens_error.detach().cpu().numpy(), 2),
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=1)
        if self.config['niflheim']:
            wandb.log({"Train Density Err %": np.round(100 * dens_error.detach().cpu().numpy(), 2)})
        if batch_idx < self.config['n_warmups']:
            self.optimizers().optimizer.param_groups[0]['lr'] = self.config['final_lr'] *((batch_idx+1)/self.config['n_warmups'])
        return loss

    def validation_step(self,
                        batch,
                        batch_idx: int):
        qm9_density, qm9_mol, qm9_n_elec, qm9_grid_dict, mol_cd_file = batch[0]
        qm9_mol.pbc = False
        pos_grid = grid2pos(qm9_mol, qm9_grid_dict).to(self.device)
        atom_positions = torch.tensor(qm9_mol.positions, device=self.device)

        result = self(qm9_mol, n_elec=qm9_n_elec)

        weights = result['weights']
        pos_disp = result['pos_disp']
        X = result['cov']
        scalar_mults = result['scal_mults']
        kappa = result['kappa']
        kappa2 = result['kappa2']
        VMF_vec = result['VMF_vec']
        n_multiples = result['n_multiples']
        if not self.config['negative_contributions']:
            scalar_mults = None
        scales_all_atoms = weights.reshape(weights.shape[0], 1, 1)

        n_multiples = n_multiples.to(self.device)
        atom_positions = atom_positions.to(self.device)
        atom_positions = atom_positions.reshape(n_multiples.shape[0], -1)
        pos_tens_orig = atom_positions.repeat_interleave(n_multiples, 0).view(-1, 3)
        pos_tens_orig = pos_tens_orig.to(self.device)
        gaus_pos = pos_tens_orig + pos_disp

        sys_dist = gto(
            pos=atom_positions,
            pos_disp=pos_disp,
            cov=X,
            kappa=kappa,
            kappa2=kappa2,
            VMF_vec=VMF_vec,
            use_vmf=self.config['use_vmf'],
            scales=scales_all_atoms,
            scalar_mults=scalar_mults,
            n_multiples=result['n_multiples'],
            relu=self.config['relu'],
            cut_distance=self.config['inference_cutoff'],
            use_links=self.config['use_links'],
            device=self.device)
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

        loss, density_error = self.loss.compute_total_loss(
            sys=qm9_mol,
            pred_cd=density,
            true_cd_tens=qm9_density,
            grid_dict=qm9_grid_dict,
            n_valence_electrons=n_val_elec,
            sampled_points=sampled_points,
            training=False
        )

        self.log("Validation Density Err %",
                 np.round(100 * density_error.detach().cpu().numpy(), 2),
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=1)
        if self.config['niflheim']:
            wandb.log({"Validation Density Err %": np.round(100 * density_error.detach().cpu().numpy(), 2)})

        if batch_idx == 0 and self.config['save_model']:
            if density_error < self.best_val_loss:
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
                                     filename=f'{self.gaus_pos_path_val}_iter_{self.val_iter}_{mol_cd_file[-17:-11]}_{qm9_mol.get_chemical_formula()}.html')

        return loss

    def predict_step(self,
                     batch,
                     batch_idx: int,
                     atom_types: list[int] = None,
                     atom_idxs: list[int] = None):
        """atom_types can be a list of atom types to predict the density for, e.g. [1,6] if only Hydrogen and Carbon.
        If None, all atoms are considered.
        atom_idxs can be a list of atom indices to predict the density for, e.g. [0,1,2] if only the first three atoms."""
        qm9_density, qm9_mol, qm9_n_elec, qm9_grid_dict, mol_cd_file = batch[0]
        qm9_mol.pbc = False
        pos_grid = grid2pos(qm9_mol, qm9_grid_dict).to(self.device)

        atoms_list = self.collate_func([qm9_mol])
        n_atoms = len(atoms_list['atom_xyz'][0])

        g, lat, state_attr = self.g_converter.get_graph(qm9_mol)

        g.ndata["pos"] = atoms_list['atom_xyz'][0]
        g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
        g.ndata['atom_numbers'] = atoms_list['nodes'][0]
        atom_positions = g.ndata["pos"]
        atom_positions = atom_positions.to(self.device)
        # g = g.to(self.device)

        result = self(qm9_mol, n_elec=qm9_n_elec)

        weights = result['weights']
        pos_disp = result['pos_disp']
        X = result['cov']
        scalar_mults = result['scal_mults']
        kappa = result['kappa']
        kappa2 = result['kappa2']
        VMF_vec = result['VMF_vec']
        if not self.config['negative_contributions']:
            scalar_mults = None
        scales_all_atoms = weights.reshape(weights.shape[0], 1, 1)

        if atom_types is not None:
            mask = torch.zeros_like(n_atoms * self.units, dtype=torch.bool)
            for i, type in enumerate(g.ndata['atom_numbers']):
                if type in atom_types:
                    mask[i * self.units:(i + 1) * self.units] = True
        elif atom_idxs is not None:
            mask = torch.zeros_like(n_atoms * self.units, dtype=torch.bool)
            for i in range(n_atoms):
                if i in atom_idxs:
                    mask[i * self.units:(i + 1) * self.units] = True
        else:
            mask = None

        sys_dist = gto(
            pos=atoms_list['atom_xyz'],
            pos_disp=pos_disp,
            cov=X,
            kappa=kappa,
            kappa2=kappa2,
            VMF_vec=VMF_vec,
            use_vmf=self.config['use_vmf'],
            scales=scales_all_atoms,
            scalar_mults=scalar_mults,
            n_multiples=result['n_multiples'],
            device=self.device,
            cut_distance=self.config['inference_cutoff'],
            relu=self.config['relu'],
            use_links=self.config['use_links'],
            mask=mask)

        if pos_grid.flatten().size()[0] > self.config['max_grid_points']:
            density, _ = get_n_points_dens(atom_dist=sys_dist,
                                           pos_grid=pos_grid,
                                           atom_positions=atom_positions,
                                           n_multiples=result['n_multiples'],
                                           n_points=self.config['n_split_points'],
                                           sample_all=True)
        else:
            density = get_full_grid_dens(atom_dist=sys_dist,
                                         atom_positions=atom_positions,
                                         n_multiples=result['n_multiples'],
                                         pos_grid=pos_grid)
        sampled_points = None

        density, n_val_elec = normalize_density(n_elec=qm9_n_elec,
                                                density=density,
                                                qm9_density=qm9_density,
                                                grid_dict=qm9_grid_dict,
                                                sys=qm9_mol,
                                                points=sampled_points)

        return density

    def test_step(self,
                        batch,
                        batch_idx: int):
        qm9_density, qm9_mol, qm9_n_elec, qm9_grid_dict, mol_cd_file = batch[0]
        qm9_mol.pbc = False
        pos_grid = grid2pos(qm9_mol, qm9_grid_dict).to(self.device)
        atom_positions = torch.tensor(qm9_mol.positions, device=self.device)

        result = self(qm9_mol, n_elec=qm9_n_elec)

        weights = result['weights']
        pos_disp = result['pos_disp']
        X = result['cov']
        scalar_mults = result['scal_mults']
        kappa = result['kappa']
        kappa2 = result['kappa2']
        VMF_vec = result['VMF_vec']
        n_multiples = result['n_multiples']
        if not self.config['negative_contributions']:
            scalar_mults = None
        scales_all_atoms = weights.reshape(weights.shape[0], 1, 1)

        n_multiples = n_multiples.to(self.device)
        atom_positions = atom_positions.to(self.device)
        atom_positions = atom_positions.reshape(n_multiples.shape[0], -1)
        pos_tens_orig = atom_positions.repeat_interleave(n_multiples, 0).view(-1, 3)
        pos_tens_orig = pos_tens_orig.to(self.device)
        gaus_pos = pos_tens_orig + pos_disp

        sys_dist = gto(
            pos=atom_positions,
            pos_disp=pos_disp,
            cov=X,
            kappa=kappa,
            kappa2=kappa2,
            VMF_vec=VMF_vec,
            use_vmf=self.config['use_vmf'],
            scales=scales_all_atoms,
            scalar_mults=scalar_mults,
            n_multiples=result['n_multiples'],
            relu=self.config['relu'],
            cut_distance=self.config['inference_cutoff'],
            use_links=self.config['use_links'],
            device=self.device)

        if pos_grid.flatten().size()[0] > self.config['max_grid_points']:
            density, _ = get_n_points_dens(atom_dist=sys_dist,
                                           pos_grid=pos_grid,
                                           gaus_pos=gaus_pos,
                                           n_multiples=result['n_multiples'],
                                           n_points=self.config['n_split_points'],
                                           sample_all=True)
        else:
            density = get_full_grid_dens(atom_dist=sys_dist,
                                         n_multiples=result['n_multiples'],
                                         pos_grid=pos_grid)
        sampled_points = None

        density, n_val_elec = normalize_density(n_elec=qm9_n_elec,
                                                density=density,
                                                qm9_density=qm9_density,
                                                grid_dict=qm9_grid_dict,
                                                sys=qm9_mol,
                                                points=sampled_points)

        loss, density_error = self.loss.compute_total_loss(
            sys=qm9_mol,
            pred_cd=density,
            true_cd_tens=qm9_density,
            grid_dict=qm9_grid_dict,
            n_valence_electrons=n_val_elec,
            sampled_points=sampled_points,
            training=False
        )

        self.log("Test Density Err %",
                 np.round(100 * density_error.detach().cpu().numpy(), 2),
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=1)

        if self.config['niflheim']:
            wandb.log({"Test Density Err %": np.round(100 * density_error.detach().cpu().numpy(), 2)})


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
                                         scal_mults=scalar_mults,
                                         weights=weights,
                                         atom_positions=atom_positions,
                                         gaus_positions=gaus_pos,
                                         origin_atoms=origin_atoms,
                                         origin_pos=pos,
                                         covariance_matrices=X,
                                     filename=f'{self.gaus_pos_path_test}{mol_cd_file[-17:-11]}_{qm9_mol.get_chemical_formula()}_batchidx_{batch_idx}.html')
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
        kappa_nontransformed = result_nontransformed['kappa']
        kappa2_nontransformed = result_nontransformed['kappa2']
        VMF_vec_nontransformed = result_nontransformed['VMF_vec']

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
        kappa_transformed = result_transformed['kappa']
        kappa2_transformed = result_transformed['kappa2']
        VMF_vec_transformed = result_transformed['VMF_vec']
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
            self.check_equivariance_vector(vector_transformed=VMF_vec_transformed, orig_vector=VMF_vec_nontransformed,
                                           r_mat=r_mat if r_mat is not None else None,
                                           translation=translation if translation is not None else None,
                                           inversion=inversion)
            self.check_equivariance_vector(vector_transformed=pos_disp_transformed, orig_vector=pos_disp_nontransformed,
                                           r_mat=r_mat if r_mat is not None else None,
                                           translation=translation if translation is not None else None,
                                           inversion=inversion)
            self.check_equivariance_matrix(matrix_transformed=X_transformed, orig_matrix=X_nontransformed,
                                           r_mat=r_mat if r_mat is not None else None,
                                           translation=translation if translation is not None else None,
                                           inversion=inversion)
            self.check_equivariance_scalar(scalar_transformed=kappa_transformed, orig_scalar=kappa_nontransformed)
            self.check_equivariance_scalar(scalar_transformed=kappa2_transformed, orig_scalar=kappa2_nontransformed)


        sys_dist = gto(
            pos=atom_positions_original,
            pos_disp=pos_disp_nontransformed,
            cov=X_nontransformed,
            kappa=kappa_nontransformed,
            kappa2=kappa2_nontransformed,
            VMF_vec=VMF_vec_nontransformed,
            use_vmf=self.config['use_vmf'],
            scales=weights_nontransformed.reshape(weights_nontransformed.shape[0], 1, 1),
            scalar_mults=scalar_mults_nontransformed,
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
                kappa=kappa_transformed,
                kappa2=kappa2_transformed,
                VMF_vec=VMF_vec_transformed,
                use_vmf=self.config['use_vmf'],
                scales=weights_transformed.reshape(weights_transformed.shape[0], 1, 1),
                scalar_mults=scalar_mults_transformed,
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
                                      conv_mode=self.config['conv_mode'],
                                      update_edge=self.config['update_edge'])
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
                                filename):
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
        gaussian_colors = ['blue' if scal < 0 else 'red' for scal in scal_mults]

        # Gaussian hover text (position and origin info)
        hover_text_gaussians = [
            f"Position: ({gp[0]:.2f}, {gp[1]:.2f}, {gp[2]:.2f})<br>" +
            f"Origin: {origin_atoms[i]} at ({op[0]:.2f}, {op[1]:.2f}, {op[2]:.2f})"
            for i, (gp, op) in enumerate(zip(gaus_positions, origin_pos))
        ]

        # Plot each Gaussian as an ellipsoid
        ellipsoids = []
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
            ellipsoids.append(
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
        fig = go.Figure(data=[trace_atoms] + atom_legend_traces + gaussian_sign_legend + ellipsoids, layout=layout)

        # Save the interactive plot to an HTML file
        pio.write_html(fig, file=filename, auto_open=False)


def collate_fn(batch):
    return batch


class QM9Dataset(Dataset):
    def __init__(self, file_list, qm9_dens_path, config):
        self.file_list = file_list
        self.qm9_dens_path = qm9_dens_path
        self.config = config
        self.cache = {}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        valid = False
        while not valid:
            try:
                mol_cd_file = self.file_list[idx]
                qm9_path = os.path.join(self.qm9_dens_path, mol_cd_file)
                if mol_cd_file in self.cache:
                    return self.cache[mol_cd_file]
                qm9_density, qm9_mol, qm9_n_elec, qm9_grid_dict = get_qm9_density(qm9_path)
                self.cache[mol_cd_file] = (qm9_density, qm9_mol, qm9_n_elec, qm9_grid_dict, mol_cd_file)
                valid = True
            except Exception as e:
                idx = np.random.randint(0, len(self.file_list)-1)
        return self.cache[mol_cd_file]


