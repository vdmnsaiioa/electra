#!/usr/bin/env python3
"""Sample random density/structure/energy ground-truth triplets.

This script picks N random density files, loads:
  - structure (ASE Atoms) embedded in CHGCAR(.lz4)
  - ground-truth density tensor
  - ground-truth energy from CSV (if available)

and writes a small report + XYZ files.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import yaml
from ase.io import write

from tools.density_conversions import chgcar_to_cd


def load_csv_map(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out: dict[str, dict[str, str]] = {}
        for row in reader:
            key = (row.get("file") or "").strip()
            if not key:
                continue
            out[key] = row
    return out


def to_energy_key_from_density_file(cd_file: str) -> str:
    stem = Path(cd_file).name.split(".")[0]
    return stem.lstrip("0") or "0"


def resolve_density_path(root: Path | None, cd_file: str) -> Path:
    p = Path(cd_file)
    if p.is_absolute():
        return p
    if root is None:
        return p
    return root / p


def infer_file_list_from_split_json(split_json: Path, dens_root: Path, split_name: str) -> list[str]:
    data = json.loads(split_json.read_text(encoding="utf-8"))
    files: list[str] = []
    for section in ("train", "validation", "test"):
        for entry in data.get(section, []):
            if split_name not in {"ECD", "ECD_test", "nmc"}:
                fn = f"{int(entry):06}.CHGCAR.lz4"
            elif split_name in {"ECD", "ECD_test"}:
                fn = f"{entry}.chgcar.lz4"
            else:
                fn = f"{entry}.CHGCAR.lz4"
            files.append(str(dens_root / fn))
    return files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("hpc_conf_mol_mm.yaml"), help="YAML config file")
    p.add_argument("--num", type=int, default=3, help="number of random structures to sample")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/ground_truth_samples"))
    p.add_argument(
        "--density-files",
        type=Path,
        help=(
            "Optional text file with one density path per line. If omitted, uses data_split_path + data_split from config."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    dens_root = Path(cfg.get("mp_dens_path") or cfg.get("dens_path") or cfg["qm9_dens_path"])
    energy_csv = Path(cfg["energy_csv"])

    if args.density_files:
        density_files = [line.strip() for line in args.density_files.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        split_name = cfg["data_split"]
        split_json = Path(cfg["data_split_path"]) / f"datasplits_{split_name}.json"
        density_files = infer_file_list_from_split_json(split_json, dens_root, split_name)

    if not density_files:
        raise RuntimeError("No density files found to sample.")

    energy_map = load_csv_map(energy_csv)
    rng = random.Random(args.seed)
    chosen = rng.sample(density_files, k=min(args.num, len(density_files)))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for i, cd_file in enumerate(chosen, start=1):
        dens_path = resolve_density_path(dens_root, cd_file)
        density, atoms = chgcar_to_cd(str(dens_path))

        key = to_energy_key_from_density_file(cd_file)
        row = energy_map.get(key, {})
        energy = row.get("energy")
        atom_count = row.get("atom_count")

        xyz_path = args.output_dir / f"sample_{i}_{atoms.get_chemical_formula()}.xyz"
        write(xyz_path.as_posix(), atoms)

        rec = {
            "sample_index": i,
            "density_file": str(dens_path),
            "energy_key": key,
            "formula": atoms.get_chemical_formula(),
            "num_atoms": len(atoms),
            "density_shape": list(density.shape),
            "energy_ground_truth": None if energy in (None, "", "None") else float(energy),
            "atom_count_ground_truth": None if atom_count in (None, "", "None") else float(atom_count),
            "xyz_file": str(xyz_path),
        }
        results.append(rec)

    out_json = args.output_dir / "samples.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Wrote {len(results)} samples to {out_json}")
    for rec in results:
        print(
            f"[{rec['sample_index']}] {rec['formula']} | dens={rec['density_file']} "
            f"| shape={tuple(rec['density_shape'])} | E={rec['energy_ground_truth']} | xyz={rec['xyz_file']}"
        )


if __name__ == "__main__":
    main()
