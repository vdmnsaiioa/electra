#!/usr/bin/env python3
"""Train a one-layer energy readout on frozen ELECTRA_mm base-model features.

The script loads an already-trained ELECTRA_mm checkpoint from
``load_model_path`` in the YAML config, keeps only the ``base_model``
(MiaoMiaoNet/MiaoNet) parameters, freezes them, and trains a single linear
layer on per-molecule features derived from the three base-model output heads and the frozen embedding output.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import os
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ase.neighborlist import neighbor_list
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.train_helper_funcs import set_all_seeds


def load_electra_mm_class():
    """Load the extension-less ELECTRA_mm module and return its ELECTRA class."""
    module_path = Path(__file__).resolve().parents[1]/ "ELECTRA" / "models" / "electra_hotpp" / "ELECTRA_mm.py"
    loader = SourceFileLoader("electra_mm_module", str(module_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ImportError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    if not hasattr(module, "ELECTRA"):
        raise ImportError(f"No ELECTRA class found in {module_path}")
    return module.ELECTRA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("hpc_conf_frozen.yaml"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging even if wandb is true in config.")
    parser.add_argument("--save_path", type=Path, default=None, help="Optional path for the trained linear readout state dict.")
    parser.add_argument("--feature_cache", type=Path, default=None, help="Optional torch feature cache produced by extract_frozen_features.py.")
    return parser.parse_args()


def load_energy_map(csv_path: Path) -> dict[str, float]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        energy_map: dict[str, float] = {}
        for row in reader:
            key = (row.get("file") or "").strip().lstrip("0") or "0"
            value = row.get("energy")
            if value not in (None, "", "None"):
                energy_map[key] = float(value)
        return energy_map


def density_filename(config: dict[str, Any], split_entry: Any) -> str:
    split_name = config["data_split"]
    if split_name not in {"ECD", "ECD_test", "nmc"}:
        return f"{int(split_entry):06}.CHGCAR.lz4"
    if split_name in {"ECD", "ECD_test"}:
        return f"{split_entry}.chgcar.lz4"
    return f"{split_entry}.CHGCAR.lz4"


def split_key(split_entry: Any) -> str:
    return str(split_entry).strip().lstrip("0") or "0"


def resolve_split_json(config: dict[str, Any]) -> Path:
    return Path(config["data_split_path"]) / f"datasplits_{config['data_split']}.json"


def resolve_density_root(config: dict[str, Any]) -> Path:
    return Path(config.get("mp_dens_path") or config.get("dens_path") or config["qm9_dens_path"])


def load_atoms_from_chgcar_lz4(path: Path):
    """Load only the ASE Atoms header from a compressed CHGCAR file.

    The frozen readout only needs atom types and coordinates for the base-model
    graph, so this avoids constructing the full charge-density grid.
    """
    import ase.io.vasp as aiv
    import lz4.frame

    with lz4.frame.open(path, mode="rb") as fp:
        filecontent = fp.read()
    return aiv.read_vasp(io.StringIO(filecontent.decode("utf-8")))


class EnergySplitDataset(Dataset):
    """Dataset of CHGCAR structures and scalar energy targets for one split."""

    def __init__(self, config: dict[str, Any], split: str, energy_map: dict[str, float]):
        split_data = json.loads(resolve_split_json(config).read_text(encoding="utf-8"))
        self.root = resolve_density_root(config)
        self.samples: list[tuple[Path, str, float]] = []
        for entry in split_data[split]:
            key = split_key(entry)
            if key not in energy_map:
                continue
            self.samples.append((self.root / density_filename(config, entry), key, energy_map[key]))
        if not self.samples:
            raise RuntimeError(f"No usable samples found for split '{split}'.")
        self._bad_keys: set[str] = set()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        first_error: Exception | None = None
        for offset in range(len(self.samples)):
            sample_index = (index + offset) % len(self.samples)
            path, key, energy = self.samples[sample_index]
            try:
                atoms = load_atoms_from_chgcar_lz4(path)
                return atoms, torch.tensor([energy], dtype=torch.float32), key
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                if key not in self._bad_keys:
                    print(f"[skip] Failed to load {path}: {exc}", flush=True)
                    self._bad_keys.add(key)

        raise RuntimeError(
            "Could not load any molecule from this split; first error was: "
            f"{first_error}"
        )


class CachedFeatureDataset(Dataset):
    """Dataset backed by precomputed frozen-base molecule features."""

    def __init__(self, cache: dict[str, Any], split: str):
        self.features = cache["features"]
        self.samples = [sample for sample in cache["samples"] if sample["split"] == split]
        if not self.samples:
            raise RuntimeError(f"No cached samples found for split '{split}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        feature_index = int(sample["feature_index"])
        features = self.features[feature_index].to(torch.float32)
        if sample["energy"] is None:
            raise RuntimeError(f"Cached sample {sample['key']} in split {sample['split']} does not have an energy target.")
        target = torch.tensor([float(sample["energy"])], dtype=torch.float32)
        return features, target, sample["key"]


def collate_cached(batch):
    features, targets, keys = zip(*batch)
    return torch.stack(features, dim=0), torch.stack(targets, dim=0), list(keys)


def collate_single(batch):
    if len(batch) != 1:
        raise ValueError("This script expects batch_size=1.")
    return batch[0]


def extract_base_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a PyTorch state dict.")
    base_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned = key.removeprefix("model.").removeprefix("electra.")
        if cleaned.startswith("base_model."):
            base_state[cleaned.removeprefix("base_model.")] = value
    if not base_state:
        raise RuntimeError("No base_model.* parameters found in checkpoint.")
    return base_state


def build_graph(atoms, config: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    atoms = atoms.copy()
    atoms.pbc = config.get("pbc", False)
    cutoff = float(config["hotpp_cutoff"])
    idx_i, idx_j, offsets = neighbor_list("ijS", atoms, cutoff, self_interaction=False)
    offset = np.array(offsets) @ atoms.get_cell()
    offset_tensor = torch.tensor(offset, dtype=dtype, device=device)
    if not atoms.pbc.all():
        offset_tensor = torch.zeros_like(offset_tensor)
    return {
        "atomic_number": torch.tensor(atoms.numbers, dtype=torch.long, device=device),
        "idx_i": torch.tensor(idx_i, dtype=torch.long, device=device),
        "idx_j": torch.tensor(idx_j, dtype=torch.long, device=device),
        "coordinate": torch.tensor(atoms.positions, dtype=dtype, device=device),
        "n_atoms": torch.tensor([len(atoms)], dtype=torch.long, device=device),
        "offset": offset_tensor,
        "scaling": torch.eye(3, dtype=dtype, device=device).view(1, 3, 3),
        "batch": torch.zeros(len(atoms), dtype=torch.long, device=device),
        "cell": torch.tensor(atoms.cell[:], dtype=dtype, device=device),
    }


@torch.no_grad()
def molecule_features_all(base_model: nn.Module, atoms, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    dtype = torch.float32
    graph = build_graph(atoms, config, device, dtype)
    base_outputs = base_model(batch_data=graph, properties=None, create_graph=False)
    output_heads = base_outputs[:3]
    init_embedding = base_outputs[3].reshape(len(atoms), -1)
    natoms = len(atoms)
    per_head_features = []
    for head in output_heads:
        scalar = head[0].reshape(natoms, -1)
        vector_norm = torch.linalg.norm(head[1].reshape(natoms, -1, 3), dim=-1)
        tensor_norm = torch.linalg.matrix_norm(head[2].reshape(natoms, -1, 3, 3), ord="fro", dim=(-2, -1))
        per_head_features.extend([scalar, vector_norm, tensor_norm])
    atom_features = torch.cat([*per_head_features, init_embedding], dim=-1)
    return atom_features.sum(dim=0)

@torch.no_grad()
def molecule_features(base_model: nn.Module, atoms, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    dtype = torch.float32
    graph = build_graph(atoms, config, device, dtype)

    base_outputs = base_model(batch_data=graph, properties=None, create_graph=False)

    # first output head only
    first_head = base_outputs[0]

    natoms = len(atoms)

    # scalar features only: shape (natoms, 750)
    scalar = first_head[0].reshape(natoms, -1)

    # sum-pool over atoms: shape (750,)
    mol_features = scalar.sum(dim=0)

    return mol_features



def evaluate_mae(base_model: nn.Module, readout: nn.Module, loader: DataLoader, config: dict[str, Any], device: torch.device) -> float:
    readout.eval()
    errors: list[torch.Tensor] = []
    with torch.no_grad():
        for atoms, target, _key in loader:
            features = molecule_features(base_model, atoms, config, device).unsqueeze(0)
            pred = readout(features).cpu()
            errors.append(torch.abs(pred.view_as(target) - target))
    return torch.cat(errors).mean().item()


def evaluate_cached_mae(readout: nn.Module, loader: DataLoader, device: torch.device) -> float:
    readout.eval()
    errors: list[torch.Tensor] = []
    with torch.no_grad():
        for features, target, _keys in loader:
            features = features.to(device)
            target = target.to(device)
            pred = readout(features)
            errors.append(torch.abs(pred.view_as(target) - target).cpu())
    return torch.cat(errors).mean().item()


def load_frozen_base_model(config: dict[str, Any], device: torch.device):
    if not config.get("load_model_path"):
        raise ValueError("Config must define load_model_path for the trained ELECTRA_mm checkpoint.")

    ELECTRA = load_electra_mm_class()
    model = ELECTRA(train_files=[], test_files=[], validation_files=[], model_handler=None, config=config).to(device)

    checkpoint = torch.load(config["load_model_path"], map_location=device)
    missing, unexpected = model.base_model.load_state_dict(extract_base_state_dict(checkpoint), strict=False)
    print(f"Loaded frozen base_model from {config['load_model_path']}")
    print(f"base_model missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    model.base_model.eval()
    for parameter in model.base_model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model, "emb_layer"):
        model.emb_layer.eval()
        for parameter in model.emb_layer.parameters():
            parameter.requires_grad_(False)
    return model.base_model


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    set_all_seeds(int(config.get("seed", 0)))
    device = torch.device(args.device)
    base_model = None
    cached_training = args.feature_cache is not None

    if cached_training:
        cache = torch.load(args.feature_cache, map_location="cpu")
        print(f"Loaded frozen feature cache from {args.feature_cache}")
        print(f"Cached samples: {len(cache['samples'])} | feature_dim={cache['feature_dim']}")
        train_dataset = CachedFeatureDataset(cache, "train")
        val_dataset = CachedFeatureDataset(cache, "validation")
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=collate_cached)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_cached)
        feature_dim = int(cache["feature_dim"])
    else:
        base_model = load_frozen_base_model(config, device)
        energy_map = load_energy_map(Path(config["energy_csv"]))
        train_dataset = EnergySplitDataset(config, "train", energy_map)
        val_dataset = EnergySplitDataset(config, "validation", energy_map)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=collate_single)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_single)

        first_atoms, _target, _key = train_dataset[0]
        feature_dim = molecule_features(base_model, first_atoms, config, device).numel()

    readout = nn.Sequential(
        nn.Linear(feature_dim, 128),
        nn.SiLU(),
        nn.Linear(128, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    use_wandb = bool(config.get("wandb", False)) and not args.no_wandb
    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(project=config.get("project_name", "ELECTRA_mm_energy_readout"), config={**config, "readout_epochs": args.epochs, "readout_lr": args.lr, "feature_cache": str(args.feature_cache) if args.feature_cache else None})

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        readout.train()
        abs_errors: list[float] = []

        for step, batch in enumerate(train_loader):
            if cached_training:
                features, target, _keys = batch
                features = features.to(device)
            else:
                atoms, target, _key = batch
                if base_model is None:
                    raise RuntimeError("base_model must be loaded when not using --feature_cache.")
                features = molecule_features(base_model, atoms, config, device).unsqueeze(0)
            target = target.to(device)
            pred = readout(features)

            loss = loss_fn(pred.view_as(target), target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_mae = torch.abs(pred.detach().view_as(target) - target).mean().item()
            abs_errors.append(batch_mae)

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step_loss": loss.item(),
                        "step_mae": batch_mae,
                        "epoch": epoch,
                        "step_in_epoch": step,
                    },
                    step=global_step,
                )

            if step % 10 == 0:
                print(
                    f"epoch={epoch:04d} step={step:05d}/{len(train_loader)} "
                    f"loss={loss.item():.8f} mae={batch_mae:.8f}",
                    flush=True,
                )

            global_step += 1

        train_mae = float(np.mean(abs_errors))
        print(f"epoch={epoch:04d} train_mae={train_mae:.8f}", flush=True)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train_mae_epoch": train_mae,
                    "epoch": epoch,
                },
                step=global_step,
            )

    if cached_training:
        val_mae = evaluate_cached_mae(readout, val_loader, device)
    else:
        if base_model is None:
            raise RuntimeError("base_model must be loaded when not using --feature_cache.")
        val_mae = evaluate_mae(base_model, readout, val_loader, config, device)
    print(f"validation_mae={val_mae:.8f}", flush=True)
    if wandb_run is not None:
        wandb_run.log({"validation_mae": val_mae})
        wandb_run.finish()

    save_path = args.save_path or Path(config.get("model_dir", ".")) / "frozen_base_energy_readout.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"readout_state_dict": readout.state_dict(), "feature_dim": feature_dim, "config_path": str(args.config), "feature_cache": str(args.feature_cache) if args.feature_cache else None}, save_path)
    print(f"Saved readout to {save_path}")

if __name__ == "__main__":
    main()
