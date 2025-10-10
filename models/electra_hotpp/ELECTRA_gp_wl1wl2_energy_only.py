"""Energy-only variant of ELECTRA_gp_wl1wl2.

This module removes every component that was previously used for density
prediction.  Only the scalar (l=0), vector (l=1) and tensor (l=2) features are
employed to build an energy readout network.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from collections import deque
from typing import Any, Dict, Iterable, Optional

import lightning as L
import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.io.vasp import read_vasp
from ase.neighborlist import neighbor_list
import lz4.frame
from matgl.utils.io import IOMixIn
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.electra_hotpp.layer import (
    AtomicEmbedding,
    AufbauEmbedding,
    BesselPoly,
    PolynomialCutoff,
)
from models.electra_hotpp.model import MiaoMiaoNet, MiaoNet
from tools.atom_tools import valence_electrons
from utils.model_handling import ModelIO
from utils.train_helper_funcs import load_csv_to_dict, set_optimizer

logger = logging.getLogger(__name__)

def _load_qm9_structure(qm9_path: str) -> Atoms:
    with lz4.frame.open(qm9_path, mode="rb") as handle:
        contents = handle.read()

    tmp_fd, tmp_path = tempfile.mkstemp(prefix="qm9_chgcar_", suffix=".chgcar")
    try:
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            tmp_file.write(contents)

        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as file_obj:
            atoms = read_vasp(file_obj)
    finally:
        os.remove(tmp_path)

    if atoms is None:  # pragma: no cover - defensive branch
        raise ValueError(f"Unable to parse atomic structure from {qm9_path}")
    return atoms


def collate_fn(batch: Iterable[Any]) -> Iterable[Any]:
    return batch


class QM9Dataset(Dataset):
    def __init__(self, file_list, qm9_dens_path, config):
        self.file_list = file_list
        self.qm9_dens_path = qm9_dens_path
        self.config = config
        self.cache: Dict[str, Any] = {}

        energy_csv = config.get("energy_csv")
        try:
            self.energy_dict = load_csv_to_dict(energy_csv, "file", "energy")
            self.atom_count_dict = load_csv_to_dict(energy_csv, "file", "atom_count")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unable to load energy metadata: %s", exc)
            self.energy_dict = None
            self.atom_count_dict = None

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        valid = False
        while not valid:
            mol_cd_file = self.file_list[idx]
            qm9_path = os.path.join(self.qm9_dens_path, mol_cd_file)
            try:
                if mol_cd_file in self.cache:
                    return self.cache[mol_cd_file]
                qm9_mol = _load_qm9_structure(qm9_path)
                qm9_n_elec = valence_electrons(qm9_mol.get_chemical_formula())
                valid = True
            except Exception as exc:  # pragma: no cover - data robustness
                logger.warning("Error loading %s: %s", mol_cd_file, exc)
                idx = np.random.randint(0, len(self.file_list) - 1)

        energy = None
        atom_count = None
        if self.energy_dict is not None and self.atom_count_dict is not None:
            key = os.path.basename(mol_cd_file).split(".")[0]
            key_no_zeros = key.lstrip("0") or "0"
            try:
                energy = float(self.energy_dict[key_no_zeros])
                atom_count = float(self.atom_count_dict[key_no_zeros])
            except Exception as exc:  # pragma: no cover - inconsistent csv
                logger.debug("Missing energy metadata for %s: %s", key_no_zeros, exc)

        sample = (
            None,
            qm9_mol,
            qm9_n_elec,
            None,
            mol_cd_file,
            None,
            energy,
            atom_count,
        )
        if not self.config.get("save_memory", False):
            self.cache[mol_cd_file] = sample
        return sample


class SmallDensityDataset(Dataset):
    def __init__(self, root, mol_name, split):
        super().__init__()
        assert mol_name in (
            "benzene",
            "ethanol",
            "phenol",
            "resorcinol",
            "ethane",
            "malonaldehyde",
        )
        self.root = root
        self.mol_name = mol_name
        self.split = split
        if split == "validation":
            split = "test"

        self.ATOM_TYPES = {
            "benzene": ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
            "ethanol": ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
            "phenol": ["C", "C", "C", "C", "C", "C", "O", "H", "H", "H", "H", "H", "H"],
            "resorcinol": ["C", "C", "C", "C", "C", "C", "O", "H", "O", "H", "H", "H", "H", "H"],
            "ethane": ["C", "C", "H", "H", "H", "H", "H", "H"],
            "malonaldehyde": ["O", "C", "C", "C", "O", "H", "H", "H", "H"],
        }

        self.n_grid = 50
        self.grid_size = 20.0
        self.data_path = os.path.join(root, mol_name, f"{mol_name}_{split}")

        self.atom_type = self.ATOM_TYPES[mol_name]
        self.atom_coords = np.load(os.path.join(self.data_path, "structures.npy"))
        energy_file = os.path.join(self.data_path, "energies.npy")
        self.energies = np.load(energy_file) if os.path.isfile(energy_file) else None

    def __getitem__(self, item):
        mol = Atoms("".join(self.atom_type), positions=self.atom_coords[item])
        n_elec = valence_electrons(mol.get_chemical_formula())

        energy = float(self.energies[item]) if self.energies is not None else None

        return None, mol, n_elec, None, None, None, energy

    def __len__(self):
        return self.atom_coords.shape[0]


class ELECTRAEnergyOnly(L.LightningModule, IOMixIn):
    __version__ = 1

    def __init__(
        self,
        train_files: Optional[list[str]] = None,
        test_files: Optional[list[str]] = None,
        validation_files: Optional[list[str]] = None,
        model_handler: Optional[ModelIO] = None,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if config is None:
            raise ValueError("Configuration dictionary must be provided")

        self.save_args(locals(), kwargs)
        self.config = config
        self.train_files = train_files or []
        self.validation_files = validation_files or []
        self.test_files = test_files or []
        self.model_handler = model_handler

        self.activation_type = config["activation_type"]
        self.gaus_per_electrons = config["gaus_per_electrons"]
        self.units = self.gaus_per_electrons
        self.master_units = config["master_units"]
        self.num_layers = config["hotpp_nlayers"]
        self.hotpp_outdim = config["hotpp_outdim"]
        self.r_max = config["r_max"]
        self.n_max = config["n_max"]
        self.out_max = config["out_max"]
        self.cutoff = config["hotpp_cutoff"]
        self.energy_loss_fn = nn.MSELoss()
        self.energy_loss_coef = config.get("energy_loss_coef", 1.0)
        self.energy_loss_history: deque[dict[str, Any]] = deque(maxlen=20)
        self.energy_nan_log_path = self._create_nan_log_path(
            config.get("energy_nan_log_path", "energy_nan_debug.txt")
        )
        self._last_energy_nan_signature: Optional[tuple[str, int, int]] = None

        self.qm9_dens_path = config.get("qm9_dens_path")
        self.precision = torch.float32

        self.init_hotpp_model()

        self.energy_input_dim = 10 * self.units
        self.phi_network = nn.Sequential(
            nn.Linear(self.energy_input_dim, 3 * self.units),
            nn.Tanh(),
            nn.Linear(3 * self.units, self.units),
        )
        self.rho_network = nn.Sequential(
            nn.Linear(self.units, self.units),
            nn.Tanh(),
            nn.Linear(self.units, 1),
        )

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _create_nan_log_path(self, base_path: str) -> str:
        log_dir, log_filename = os.path.split(base_path)
        name, ext = os.path.splitext(log_filename)
        unique_suffix = uuid.uuid4().hex[:8]
        new_filename = f"{name}_{unique_suffix}{ext}" if name else f"{unique_suffix}{ext}"
        return os.path.join(log_dir, new_filename) if log_dir else new_filename

    @staticmethod
    def _serialize_energy(value: torch.Tensor | float | int | None) -> Optional[float | list[float]]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            flat = value.detach().cpu().reshape(-1)
            if flat.numel() == 1:
                return float(flat.item())
            return flat.tolist()
        return float(value)

    def _write_energy_nan_history(self, stage: str) -> None:
        if not self.energy_loss_history:
            return
        try:
            with open(self.energy_nan_log_path, "a", encoding="utf-8") as handle:
                json.dump(list(self.energy_loss_history), handle)
                handle.write("\n")
        except OSError as exc:  # pragma: no cover - diagnostic utility
            logger.error("Failed to write NaN history (%s): %s", stage, exc)

    def _track_energy_information(
        self,
        *,
        stage: str,
        identifier: Optional[str],
        batch_idx: int,
        qm9_mol: Optional[Atoms],
        predicted_energy: Optional[torch.Tensor],
        target_energy: Optional[torch.Tensor],
        energy_loss: Optional[torch.Tensor],
    ) -> None:
        if predicted_energy is None or target_energy is None:
            return
        record = {
            "stage": stage,
            "global_step": int(self.global_step),
            "batch_index": int(batch_idx),
            "index": self._get_entry_identifier(identifier, batch_idx),
            "molecular_formula": qm9_mol.get_chemical_formula() if qm9_mol is not None else None,
            "predicted_energy": self._serialize_energy(predicted_energy),
            "target_energy": self._serialize_energy(target_energy),
        }
        self.energy_loss_history.append(record)
        if energy_loss is None:
            return
        energy_loss_detached = energy_loss.detach()
        if not torch.isnan(energy_loss_detached).any().item():
            return
        signature = (stage, int(self.global_step), int(batch_idx))
        if self._last_energy_nan_signature == signature:
            return
        self._write_energy_nan_history(stage)
        self._last_energy_nan_signature = signature

    def _get_entry_identifier(self, identifier: Optional[str], batch_idx: int) -> str:
        if identifier:
            base = os.path.basename(identifier)
            return base.split(".")[0]
        return f"batch_{batch_idx}"

    @staticmethod
    def slice(self_tensor: torch.Tensor, n_multiples: torch.Tensor) -> torch.Tensor:
        slices = []
        for i, num_selections in enumerate(n_multiples):
            count = int(num_selections.item())
            if count == 0:
                continue
            slices.append(self_tensor[i, torch.arange(self_tensor.size(1), device=self_tensor.device)[:count]])
        if slices:
            return torch.cat(slices, dim=0)
        shape = (0,) + tuple(self_tensor.shape[2:])
        return torch.empty(*shape, device=self_tensor.device, dtype=self_tensor.dtype)

    # ------------------------------------------------------------------
    # Model definition
    # ------------------------------------------------------------------
    def init_hotpp_model(self) -> None:
        elements = ["H", "O", "C", "N", "F", "Cl", "Br", "I"]
        element_numbers = [atomic_numbers[element] for element in elements]
        embed_type = self.config.get("embed_type", "aufbau")
        if embed_type == "aufbau":
            emb_layer = AufbauEmbedding(element_numbers, self.config["master_units"], device=self.device)
        else:
            emb_layer = AtomicEmbedding(element_numbers, self.config["master_units"])
        emb_layer.to(self.device)
        self.emb_layer = emb_layer
        cut_fn = PolynomialCutoff(cutoff=self.config["hotpp_cutoff"], p=self.config["poly_order"])
        radial_fn = BesselPoly(
            r_max=self.config["hotpp_cutoff"],
            n_max=self.config["master_units"],
            cutoff_fn=cut_fn,
        )
        nlayers = self.config["hotpp_nlayers"]
        if self.config.get("miaonet") == "miaomiao":
            self.base_model = MiaoMiaoNet(
                embedding_layer=emb_layer,
                radial_fn=radial_fn,
                n_layers=nlayers,
                max_n_body=[self.config["n_max"]] * nlayers,
                max_r_way=[self.config["r_max"]] * nlayers,
                max_out_way=[self.config["out_max"]] * nlayers,
                max_out_heads=self.config["max_out_heads"],
                output_dim=[self.config["hotpp_outdim"]] * nlayers,
                activate_fn=self.config["activation_type"],
                head_activate_list=self.config["head_activate_list"],
                mean=0.0,
                std=1.0,
                norm_factor=1.0,
                bilinear=False,
                norm_heads=self.config["norm_heads"],
                norm_blocks=self.config["norm_blocks"],
                conv_mode=self.config["conv_mode"],
                update_edge=self.config["update_edge"],
                prune=self.config["prune"],
            )
        else:
            self.base_model = MiaoNet(
                embedding_layer=emb_layer,
                radial_fn=radial_fn,
                n_layers=nlayers,
                max_r_way=[self.config["r_max"]] * nlayers,
                max_out_way=[self.config["out_max"]] * nlayers,
                max_out_heads=self.config["max_out_heads"],
                output_dim=[self.config["hotpp_outdim"]] * nlayers,
                activate_fn=self.config["activation_type"],
                head_activate_list=self.config["head_activate_list"],
                mean=0.0,
                std=1.0,
                norm_factor=1.0,
                bilinear=False,
                conv_mode=self.config["conv_mode"],
                update_edge=self.config["update_edge"],
            )

    # ------------------------------------------------------------------
    # Forward pass and feature construction
    # ------------------------------------------------------------------
    def forward(
        self,
        atoms: Atoms,
        state_attr: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        start = time.time()
        atoms.pbc = self.config.get("pbc", False)
        idx_i, idx_j, offsets = neighbor_list("ijS", atoms, self.cutoff, self_interaction=False)
        offset = np.array(offsets) @ atoms.get_cell()
        if not atoms.pbc.all():
            offset = torch.zeros_like(torch.tensor(offset, dtype=self.precision, device=self.device))
        data = {
            "atomic_number": torch.tensor(atoms.numbers, dtype=torch.long, device=self.device),
            "idx_i": torch.tensor(idx_i, dtype=torch.long, device=self.device),
            "idx_j": torch.tensor(idx_j, dtype=torch.long, device=self.device),
            "coordinate": torch.tensor(atoms.positions, dtype=self.precision, device=self.device),
            "n_atoms": torch.tensor([len(atoms)], dtype=torch.long, device=self.device),
            "offset": torch.tensor(offset, dtype=self.precision, device=self.device),
            "scaling": torch.eye(3, dtype=self.precision, device=self.device).view(1, 3, 3),
            "batch": torch.zeros(len(atoms), dtype=torch.long, device=self.device),
            "cell": torch.tensor(atoms.cell[:], dtype=self.precision, device=self.device),
        }

        X_scal_full, X_vec_full, X_tens_full, init_emb = self.base_model(
            batch_data=data,
            properties=None,
            create_graph=False,
        )

        valence_tensor = torch.tensor(
            [valence_electrons(chemical_symbols[atom]) for atom in data["atomic_number"]],
            dtype=torch.long,
            device=self.device,
        )
        n_multiples = (valence_tensor * self.units).to(self.device)
        atom_embeds = init_emb.repeat_interleave(valence_tensor, dim=0)
        n_atoms = int(torch.sum(valence_tensor).item())

        X_scal = self.slice(X_scal_full[0], n_multiples).view(n_atoms, self.units)
        X_scal_vec = self.slice(X_scal_full[1], n_multiples).view(n_atoms, self.units, 3)
        X_scal_tens = self.slice(X_scal_full[2], n_multiples).view(n_atoms, self.units, 3, 3)

        X_vec_scal = self.slice(X_vec_full[0], n_multiples).view(n_atoms, self.units)
        X_vec = self.slice(X_vec_full[1], n_multiples).view(n_atoms, self.units, 3)
        X_vec_tens = self.slice(X_vec_full[2], n_multiples).view(n_atoms, self.units, 3, 3)

        X_tens_scal = self.slice(X_tens_full[0], n_multiples).view(n_atoms, self.units)
        X_tens_vec = self.slice(X_tens_full[1], n_multiples).view(n_atoms, self.units, 3)
        X_tens = self.slice(X_tens_full[2], n_multiples).view(n_atoms, self.units, 3, 3)

        l0_features = torch.cat(
            (
                atom_embeds[:, : self.units],
                X_scal,
                X_vec_scal,
                X_tens_scal,
            ),
            dim=-1,
        )

        l1_features = torch.cat(
            (
                torch.linalg.norm(X_vec, dim=-1),
                torch.linalg.norm(X_scal_vec, dim=-1),
                torch.linalg.norm(X_tens_vec, dim=-1),
            ),
            dim=-1,
        )

        det_scal = torch.det(X_scal_tens.view(-1, 3, 3)).view(n_atoms, self.units)
        det_vec = torch.det(X_vec_tens.view(-1, 3, 3)).view(n_atoms, self.units)
        det_tens = torch.det(X_tens.view(-1, 3, 3)).view(n_atoms, self.units)
        l2_features = torch.cat((det_scal, det_vec, det_tens), dim=-1)

        energy_features = torch.cat((l0_features, l1_features, l2_features), dim=-1)
        phi_out = self.phi_network(energy_features)
        pooled = phi_out.sum(dim=0)
        energy = self.rho_network(pooled).squeeze(-1)

        end = time.time()
        if self.config.get("wandb"):
            import wandb

            wandb.log({"Forward Time": end - start})
        else:
            logger.debug("Forward Time: %.4f", end - start)

        return {"energy": energy}

    # ------------------------------------------------------------------
    # Optimizer configuration
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = set_optimizer(self, self.config)
        anneal_milestones = list(self.config["lr_dec_every"] * np.arange(1, 500))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=anneal_milestones,
            gamma=self.config["lr_gamma"],
        )
        return [optimizer], [lr_scheduler]

    # ------------------------------------------------------------------
    # Training / validation / test steps
    # ------------------------------------------------------------------
    def _unpack_batch(self, batch):
        density, qm9_mol, _, _, identifier, _, *rest = batch[0]
        energy = rest[0] if rest else None
        atom_count = rest[1] if len(rest) > 1 else None
        return density, qm9_mol, identifier, energy, atom_count

    def _compute_energy_metrics(self, predicted, energy, atom_count):
        if energy is None:
            raise ValueError("Energy value must be provided for training")
        target_energy = torch.tensor(energy, dtype=predicted.dtype, device=predicted.device)
        energy_loss = self.energy_loss_fn(predicted, target_energy)
        rmse = torch.sqrt(energy_loss)
        metrics = {
            "energy_loss": energy_loss,
            "rmse": rmse,
            "nl1": torch.abs(predicted - target_energy),
        }
        if atom_count is not None:
            atom_count_t = torch.tensor(atom_count, dtype=predicted.dtype, device=predicted.device)
            metrics["normalized_rmse"] = rmse / atom_count_t
            metrics["normalized_l1"] = metrics["nl1"] / atom_count_t
        return energy_loss, metrics, target_energy

    def _log_energy_metrics(
        self,
        metrics: dict[str, torch.Tensor],
        stage: str,
        batch_size: int | None = None,
    ) -> None:
        log_batch_size = (
            batch_size
            or self.config.get("real_batch_size")
            or self.config.get("batch_size")
            or 1
        )
        self.log(
            f"{stage.capitalize()} Energy Loss",
            metrics["energy_loss"].detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=log_batch_size,
        )
        if "rmse" in metrics:
            self.log(
                f"{stage.capitalize()} Energy RMSE",
                metrics["rmse"].detach(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=log_batch_size,
            )
        if "normalized_rmse" in metrics:
            self.log(
                f"{stage.capitalize()} Energy NRMSE",
                metrics["normalized_rmse"].detach(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=log_batch_size,
            )
        if "normalized_l1" in metrics:
            self.log(
                f"{stage.capitalize()} Energy NL1",
                metrics["normalized_l1"].detach(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=log_batch_size,
            )

    def training_step(self, batch, batch_idx: int):
        _, qm9_mol, identifier, energy, atom_count = self._unpack_batch(batch)
        result = self(qm9_mol)
        energy_loss, metrics, target_energy = self._compute_energy_metrics(result["energy"], energy, atom_count)
        scaled_loss = self.energy_loss_coef * energy_loss
        self._log_energy_metrics(metrics, "train")
        self._track_energy_information(
            stage="train",
            identifier=identifier,
            batch_idx=batch_idx,
            qm9_mol=qm9_mol,
            predicted_energy=result["energy"],
            target_energy=target_energy,
            energy_loss=energy_loss,
        )
        return scaled_loss

    def validation_step(self, batch, batch_idx: int):
        _, qm9_mol, identifier, energy, atom_count = self._unpack_batch(batch)
        result = self(qm9_mol)
        energy_loss, metrics, target_energy = self._compute_energy_metrics(result["energy"], energy, atom_count)
        scaled_loss = self.energy_loss_coef * energy_loss
        self._log_energy_metrics(metrics, "val")
        self._track_energy_information(
            stage="val",
            identifier=identifier,
            batch_idx=batch_idx,
            qm9_mol=qm9_mol,
            predicted_energy=result["energy"],
            target_energy=target_energy,
            energy_loss=energy_loss,
        )
        return scaled_loss

    def test_step(self, batch, batch_idx: int):
        _, qm9_mol, identifier, energy, atom_count = self._unpack_batch(batch)
        result = self(qm9_mol)
        energy_loss, metrics, target_energy = self._compute_energy_metrics(result["energy"], energy, atom_count)
        scaled_loss = self.energy_loss_coef * energy_loss
        self._log_energy_metrics(metrics, "test")
        self._track_energy_information(
            stage="test",
            identifier=identifier,
            batch_idx=batch_idx,
            qm9_mol=qm9_mol,
            predicted_energy=result["energy"],
            target_energy=target_energy,
            energy_loss=energy_loss,
        )
        return scaled_loss

    # ------------------------------------------------------------------
    # Data interfaces
    # ------------------------------------------------------------------
    def train_dataloader(self):
        if self.config.get("MD_Data"):
            return DataLoader(
                SmallDensityDataset(self.config["MD_ROOT"], self.config["MD_MOL"], "train"),
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=True,
                num_workers=0,
            )
        return DataLoader(
            QM9Dataset(self.train_files, self.qm9_dens_path, self.config),
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=0,
        )

    def val_dataloader(self):
        if self.config.get("MD_Data"):
            return DataLoader(
                SmallDensityDataset(self.config["MD_ROOT"], self.config["MD_MOL"], "validation"),
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=True,
                num_workers=0,
            )
        return DataLoader(
            QM9Dataset(self.validation_files, self.qm9_dens_path, self.config),
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=0,
        )

    def test_dataloader(self):
        if self.config.get("MD_Data"):
            return DataLoader(
                SmallDensityDataset(self.config["MD_ROOT"], self.config["MD_MOL"], "test"),
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=True,
                num_workers=0,
            )
        return DataLoader(
            QM9Dataset(self.test_files, self.qm9_dens_path, self.config),
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=0,
        )


__all__ = ["ELECTRAEnergyOnly", "QM9Dataset", "SmallDensityDataset", "collate_fn"]
