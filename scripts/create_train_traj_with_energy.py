#!/usr/bin/env python3
"""Create an ASE .traj from the train split using structures from density files.

For each index listed in `train` of a datasplit JSON file, this script:
1) loads the corresponding density/CHGCAR file,
2) extracts only the atomic structure (no density grid),
3) attaches the matching energy from a CSV file as `atoms.info['energy']`,
4) streams structures into a single `.traj` file.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path

import lz4.frame
from ase.io import read
from ase.io.trajectory import Trajectory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-json", type=Path, default=Path("data_splits/datasplits_mpfull2025.json"))
    p.add_argument("--energies-csv", type=Path, default=Path("data_splits/energies.csv"))
    p.add_argument("--density-dir", type=Path, required=True)
    p.add_argument("--density-pattern", default="{index:06}.CHGCAR.lz4")
    p.add_argument("--output", type=Path, default=Path("train_structures_with_energy.traj"))
    p.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N train indices.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of train indices for quick testing.",
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
            energies[int(row["file"])] = float(row["energy"])
    return energies


def load_structure_from_density(path: Path):
    with lz4.frame.open(path, mode="rb") as fp:
        chgcar_bytes = fp.read()
    return read(io.StringIO(chgcar_bytes.decode("utf-8")), format="vasp")


def main() -> None:
    args = parse_args()
    train_indices = load_train_indices(args.split_json)
    if args.limit is not None:
        train_indices = train_indices[: args.limit]
    energies = load_energies(args.energies_csv)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    missing_density = 0
    missing_energy = 0
    written = 0
    total = len(train_indices)
    start = time.time()

    with Trajectory(args.output, mode="w") as traj:
        for i, idx in enumerate(train_indices, start=1):
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
            traj.write(atoms)
            written += 1

            if args.progress_every > 0 and i % args.progress_every == 0:
                elapsed = time.time() - start
                print(
                    f"Processed {i}/{total} | written={written} | "
                    f"missing_density={missing_density} | missing_energy={missing_energy} | "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    elapsed = time.time() - start
    print(f"Train indices: {total}")
    print(f"Wrote frames: {written}")
    print(f"Missing density files: {missing_density}")
    print(f"Missing energies: {missing_energy}")
    print(f"Elapsed seconds: {elapsed:.1f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
