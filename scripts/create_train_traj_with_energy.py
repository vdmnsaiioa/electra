#!/usr/bin/env python3
"""Create an ASE .traj from the train split using structures from density files.

For each index listed in `train` of a datasplit JSON file, this script:
1) loads the corresponding density/CHGCAR file,
2) extracts only the atomic structure (no density grid),
3) attaches the matching energy from a CSV file as `atoms.info['energy']`,
4) writes all structures to a single `.traj` file.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import lz4.frame
from ase.io import read, write


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--split-json",
        type=Path,
        default=Path("data_splits/datasplits_mpfull2025.json"),
        help="Path to datasplit JSON containing a 'train' list.",
    )
    p.add_argument(
        "--energies-csv",
        type=Path,
        default=Path("data_splits/energies.csv"),
        help="CSV with columns: file, energy.",
    )
    p.add_argument(
        "--density-dir",
        type=Path,
        required=True,
        help="Directory containing density files.",
    )
    p.add_argument(
        "--density-pattern",
        default="{index:06}.CHGCAR.lz4",
        help="Filename pattern relative to --density-dir; uses {index} placeholder.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("train_structures_with_energy.traj"),
        help="Output ASE trajectory file path.",
    )
    return p.parse_args()


def load_train_indices(split_json: Path) -> list[int]:
    with split_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "train" not in data:
        raise KeyError(f"Missing 'train' key in split file: {split_json}")
    return [int(x) for x in data["train"]]


def load_energies(energies_csv: Path) -> dict[int, float]:
    energies: dict[int, float] = {}
    with energies_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"file", "energy"} - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"Missing required column(s) in {energies_csv}: {sorted(missing)}")

        for row in reader:
            idx = int(row["file"])
            energies[idx] = float(row["energy"])
    return energies


def load_structure_from_density(path: Path):
    with lz4.frame.open(path, mode="rb") as fp:
        chgcar_bytes = fp.read()
    return read(io.StringIO(chgcar_bytes.decode("utf-8")), format="vasp")


def main() -> None:
    args = parse_args()

    train_indices = load_train_indices(args.split_json)
    energies = load_energies(args.energies_csv)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    missing_density = 0
    missing_energy = 0
    written = 0

    frames = []
    for idx in train_indices:
        density_path = args.density_dir / args.density_pattern.format(index=idx)
        if not density_path.exists():
            missing_density += 1
            continue

        energy = energies.get(idx)
        if energy is None:
            missing_energy += 1
            continue

        atoms = load_structure_from_density(density_path)
        atoms.info["index"] = idx
        atoms.info["energy"] = energy
        frames.append(atoms)
        written += 1

    if frames:
        write(args.output, frames, format="traj")

    print(f"Train indices: {len(train_indices)}")
    print(f"Wrote frames: {written}")
    print(f"Missing density files: {missing_density}")
    print(f"Missing energies: {missing_energy}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
