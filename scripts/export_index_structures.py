#!/usr/bin/env python3
"""Export dataset index -> molecular structure mappings without density values.

Reads split indices from a datasplit JSON and corresponding CHGCAR(.lz4) files,
extracts only the embedded atomic structure (ASE Atoms), and writes one JSON
record per molecule.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import lz4.frame
from ase.io import read, write


def _build_density_path(qm9_dens_path: str, split_name: str, sample_id: int | str) -> str:
    if split_name not in {"ECD", "ECD_test", "nmc"}:
        return f"{qm9_dens_path}/{int(sample_id):06}.CHGCAR.lz4"
    if split_name in {"ECD", "ECD_test"}:
        return f"{qm9_dens_path}/{sample_id}.chgcar.lz4"
    return f"{qm9_dens_path}/{sample_id}.CHGCAR.lz4"


def _load_atoms_from_lz4_chgcar(path: str):
    with lz4.frame.open(path, mode="rb") as fp:
        chgcar_bytes = fp.read()
    return read(io.StringIO(chgcar_bytes.decode("utf-8")), format="vasp")


def _atoms_to_xyz_string(atoms) -> str:
    buf = io.StringIO()
    write(buf, atoms, format="xyz")
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--subset",
        choices=["train", "validation", "test", "all"],
        default="all",
        help="Which split subset to export.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file path.")
    args = parser.parse_args()

    import yaml

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    split_name = config["data_split"]
    data_split_file = Path(config["data_split_path"]) / f"datasplits_{split_name}.json"
    with data_split_file.open("r", encoding="utf-8") as f:
        split_data = json.load(f)

    subset_keys = ["train", "validation", "test"] if args.subset == "all" else [args.subset]
    output_records = []

    for subset_key in subset_keys:
        sample_ids = split_data[subset_key]
        for dataset_index, sample_id in enumerate(sample_ids):
            density_path = _build_density_path(config["qm9_dens_path"], split_name, sample_id)
            atoms = _load_atoms_from_lz4_chgcar(density_path)
            output_records.append(
                {
                    "subset": subset_key,
                    "dataset_index": dataset_index,
                    "sample_id": sample_id,
                    "density_file": density_path,
                    "formula": atoms.get_chemical_formula(),
                    "structure_xyz": _atoms_to_xyz_string(atoms),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(output_records)} structures to {args.output}")


if __name__ == "__main__":
    main()
