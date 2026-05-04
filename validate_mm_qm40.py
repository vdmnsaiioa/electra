#!/usr/bin/env python3
import argparse
import importlib.util
import math
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pandas as pd
import torch
import yaml
from ase.io import read

from tools.atom_tools import valence_electrons
from utils.train_helper_funcs import get_files, set_all_paths, set_all_seeds


def load_electra_mm_class(repo_root: Path):
    module_root = repo_root / "models" / "electra_hotpp"
    module_path = module_root / "ELECTRA_mm"
    if not module_path.exists():
        module_path = module_root / "ELECTRA_mm.py"

    loader = SourceFileLoader("electra_mm_eval_module", str(module_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ImportError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)

    if not hasattr(module, "ELECTRA"):
        raise ImportError(f"No ELECTRA class found in {module_path}")
    return module.ELECTRA


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ELECTRA_mm MAE on XYZ + energy CSV dataset.")
    p.add_argument("--config", default="/home/energy/s234633/ELECTRA/hpc_conf_mol_mm.yaml", help="Path to model config YAML (must include load_model/load_model_path).")
    p.add_argument("--xyz_dir", default="/home/energy/s234633/ELECTRA/for_qm40/xyz_out", help="Directory containing .xyz files named like ZINC....xyz")
    p.add_argument("--energy_csv", default="/home/energy/s234633/ELECTRA/data_splits/energies_qm40.csv", help="CSV with at least id and energy columns.")
    p.add_argument("--id_col", default="id", help="ID column in energy CSV.")
    p.add_argument("--energy_col", default="formation_energy", help="Energy column in energy CSV.")
    p.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")
    p.add_argument("--limit", type=int, default=None, help="Optional max number of molecules to evaluate.")
    return p.parse_args()


def load_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["wandb"] = False
    cfg["save_model"] = False
    return cfg


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    xyz_dir = Path(args.xyz_dir)
    energy_csv_path = Path(args.energy_csv)

    if not xyz_dir.exists():
        raise FileNotFoundError(f"XYZ directory not found: {xyz_dir}")
    if not energy_csv_path.exists():
        raise FileNotFoundError(f"Energy CSV not found: {energy_csv_path}")

    # Load energy mapping
    df_energy = pd.read_csv(energy_csv_path)
    if args.id_col not in df_energy.columns or args.energy_col not in df_energy.columns:
        raise ValueError(
            f"Energy CSV must contain columns '{args.id_col}' and '{args.energy_col}'. "
            f"Found: {list(df_energy.columns)}"
        )

    energy_map = {}
    for _, row in df_energy.iterrows():
        mol_id = str(row[args.id_col])
        try:
            energy_map[mol_id] = float(row[args.energy_col])
        except Exception:
            continue

    # Collect XYZ files and keep only those with known target energy
    xyz_files = sorted(xyz_dir.glob("*.xyz"))
    if not xyz_files:
        raise ValueError(f"No .xyz files found in {xyz_dir}")

    samples = []
    for p in xyz_files:
        mol_id = p.stem
        if mol_id in energy_map:
            samples.append((mol_id, p, energy_map[mol_id]))

    if args.limit is not None:
        samples = samples[: args.limit]

    if not samples:
        raise ValueError("No overlapping IDs between xyz files and energy CSV.")

    # Build/load model exactly like repo validation flow
    cfg = load_config(Path(args.config))
    if not cfg.get("load_model", False):
        raise ValueError("Config must have load_model: True.")
    if "load_model_path" not in cfg:
        raise ValueError("Config must define load_model_path.")

    set_all_seeds(cfg["seed"])
    cfg = set_all_paths(cfg, wb_run_name=None)
    train_files, test_files, validation_files = get_files(cfg)  # not used for this custom loop, but keeps init compatible

    ELECTRA = load_electra_mm_class(repo_root)
    model = ELECTRA(
        train_files=train_files,
        test_files=test_files,
        validation_files=validation_files,
        model_handler=None,
        config=cfg,
    )

    print(f"Loading model from {cfg['load_model_path']}")
    state_dict = torch.load(cfg["load_model_path"], map_location="cpu")
    model.load_state_dict(state_dict)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = model.to(device)
    model.eval()

    abs_errors = []
    sq_errors = []

    with torch.no_grad():
        for i, (mol_id, xyz_path, target_energy) in enumerate(samples):
            atoms = read(str(xyz_path))  # ASE Atoms
            n_elec = valence_electrons(atoms.get_chemical_formula())

            out = model(atoms, n_elec=n_elec)  # same call pattern as repo
            pred = out["energy"]
            target = torch.as_tensor(target_energy, dtype=pred.dtype, device=pred.device)

            err = target - pred
            ae = torch.abs(err).item()
            se = torch.square(err).item()

            abs_errors.append(ae)
            sq_errors.append(se)

            print(f"[{i+1}/{len(samples)}] {mol_id}: pred={pred.item():.8f} target={target_energy:.8f} abs_err={ae:.8f}")

    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(sq_errors) / len(sq_errors))

    print("\n=== Final Metrics ===")
    print(f"N: {len(samples)}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")


if __name__ == "__main__":
    main()
