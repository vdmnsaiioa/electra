#!/usr/bin/env python3
"""Precompute frozen ELECTRA_mm molecule features for a data split.

This script loops through every molecule listed in the configured data-split JSON,
runs the frozen ELECTRA_mm base model once, and saves the resulting molecule
features plus split/key/energy metadata to a single ``torch.save`` cache file.

The cache is intended for ``train_frozen_mlp2.py --feature_cache ...`` so the
readout training loop can load tensors directly instead of rebuilding graphs and
rerunning the frozen base model on every epoch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from train_frozen_mlp2 import (
    density_filename,
    load_atoms_from_chgcar_lz4,
    load_energy_map,
    load_frozen_base_model,
    molecule_features,
    resolve_density_root,
    resolve_split_json,
    split_key,
)
from utils.train_helper_funcs import set_all_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("hpc_conf_frozen.yaml"))
    parser.add_argument("--output", type=Path, default=None, help="Output .pt feature-cache path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "validation", "test"),
        help="Split names to precompute from the data-split JSON.",
    )
    return parser.parse_args()


def default_output_path(config: dict[str, Any], config_path: Path) -> Path:
    split_name = config.get("data_split", config_path.stem)
    model_dir = Path(config.get("model_dir", "."))
    return model_dir / f"frozen_base_features_{split_name}.pt"


def iter_split_samples(config: dict[str, Any], split: str, energy_map: dict[str, float]):
    split_json = resolve_split_json(config)
    split_data = json.loads(split_json.read_text(encoding="utf-8"))
    if split not in split_data:
        print(f"[skip] Split '{split}' was not found in {split_json}", flush=True)
        return

    density_root = resolve_density_root(config)
    for entry in split_data[split]:
        key = split_key(entry)
        yield {
            "split": split,
            "key": key,
            "energy": energy_map.get(key),
            "path": density_root / density_filename(config, entry),
        }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_all_seeds(int(config.get("seed", 0)))
    device = torch.device(args.device)
    output_path = args.output or default_output_path(config, args.config)

    energy_map = load_energy_map(Path(config["energy_csv"]))
    base_model = load_frozen_base_model(config, device)

    samples: list[dict[str, Any]] = []
    feature_tensors: list[torch.Tensor] = []
    feature_dim: int | None = None

    for split in args.splits:
        split_samples = list(iter_split_samples(config, split, energy_map))
        print(f"Precomputing {len(split_samples)} samples for split '{split}'", flush=True)

        for index, sample in enumerate(split_samples):
            try:
                atoms = load_atoms_from_chgcar_lz4(sample["path"])
            except Exception as exc:
                print(f"[skip] Failed to load {sample['path']}: {exc}", flush=True)
                continue

            features = molecule_features(base_model, atoms, config, device).detach().cpu().to(torch.float32)
            if feature_dim is None:
                feature_dim = features.numel()
            elif features.numel() != feature_dim:
                raise RuntimeError(f"Feature dimension changed from {feature_dim} to {features.numel()} for key {sample['key']}.")

            samples.append(
                {
                    "split": sample["split"],
                    "key": sample["key"],
                    "energy": sample["energy"],
                    "path": str(sample["path"]),
                    "feature_index": len(feature_tensors),
                }
            )
            feature_tensors.append(features.reshape(-1))

            if index % 100 == 0:
                print(f"  split={split} index={index:06d}/{len(split_samples)} key={sample['key']}", flush=True)

    if not feature_tensors:
        raise RuntimeError("No features were extracted.")

    cache = {
        "features": torch.stack(feature_tensors, dim=0),
        "samples": samples,
        "feature_dim": feature_dim,
        "config_path": str(args.config),
        "data_split": config.get("data_split"),
        "split_json": str(resolve_split_json(config)),
        "load_model_path": config.get("load_model_path"),
        "feature_extractor": "train_frozen_mlp2.molecule_features:first_head_scalar_sum",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    print(f"Saved {len(samples)} feature rows with dim={feature_dim} to {output_path}", flush=True)


if __name__ == "__main__":
    main()
