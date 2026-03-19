import argparse
import importlib.util
from importlib.machinery import SourceFileLoader
import os
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch import seed_everything

from utils.train_helper_funcs import get_files, set_all_paths, set_all_seeds


def load_electra_mm_class():
    """Load the ELECTRA class implementation from models/electra_hotpp/ELECTRA_mm."""
    module_root = Path(__file__).parent / "models" / "electra_hotpp"
    module_path = module_root / "ELECTRA_mm"
    if not module_path.exists():
        module_path = module_root / "ELECTRA_mm.py"
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
    parser = argparse.ArgumentParser(
        description="Run validation-only for the multimodal ELECTRA checkpoint."
    )
    parser.add_argument(
        "--config",
        default="hpc_conf_mol_mm.yaml",
        help="Path to YAML config containing multimodal checkpoint settings.",
    )
    parser.add_argument(
        "--metrics_file",
        default=None,
        help=(
            "Optional explicit path for a metrics log file. If omitted and running "
            "under Slurm, metrics are also appended to logs2/electra_MM_<jobid>.log."
        ),
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU validation.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logger (disabled by default for validation-only runs).",
    )
    return parser.parse_args()


def default_slurm_metrics_file() -> Path | None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    return Path("logs2") / f"electra_MM_{job_id}.log"


def emit_metrics(metrics: dict, metrics_file: Path | None) -> None:
    lines = ["Validation metrics:"]
    for key in sorted(metrics.keys()):
        lines.append(f"{key}: {metrics[key]}")
    content = "\n".join(lines)

    # Always print to stdout/stderr so Slurm --output captures everything.
    print(content, flush=True)

    if metrics_file is not None:
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with metrics_file.open("a", encoding="utf-8") as f:
            f.write(content + "\n")


def load_validation_config(config_path: str | os.PathLike[str], *, wandb: bool) -> dict:
    """Load and normalize config for validation-only checkpoint evaluation."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not wandb:
        config["wandb"] = False
    config["save_model"] = False
    return config


def evaluate_loaded_model_on_validation(
    config: dict,
    *,
    force_cpu: bool = False,
) -> dict:
    """Evaluate a loaded ELECTRA_mm checkpoint on the validation split.

    The validation molecules are resolved from the datasplit JSON referenced by
    ``config['data_split_path']`` and ``config['data_split']`` via ``get_files``.
    The checkpoint itself is loaded from ``config['load_model_path']`` and must
    be enabled with ``config['load_model'] = True``.
    """
    ELECTRA = load_electra_mm_class()

    set_all_seeds(config["seed"])
    config = set_all_paths(config, wb_run_name=None)
    train_files, test_files, validation_files = get_files(config)

    model = ELECTRA(
        train_files=train_files,
        test_files=test_files,
        validation_files=validation_files,
        model_handler=None,
        config=config,
    )

    if not config.get("load_model", False):
        raise ValueError(
            "Config must have load_model: True for validation-only checkpoint runs."
        )

    if not validation_files:
        raise ValueError(
            "No validation molecules were resolved from the datasplit JSON in config."
        )

    print(f"Loading model from {config['load_model_path']}", flush=True)
    state_dict = torch.load(config["load_model_path"], map_location="cpu")
    model.load_state_dict(state_dict)

    seed_everything(config["seed"], workers=True)
    accelerator = "cpu" if force_cpu else "auto"
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_progress_bar=False,
    )

    val_loader = model.val_dataloader()
    results = trainer.validate(model=model, dataloaders=val_loader)
    metrics = dict(results[0]) if results else {}

    summary_metrics = {
        key: metrics[key]
        for key in (
            "Validation Energy RMSE",
            "Validation Energy RMSE (meV)",
            "Validation Energy MAE",
            "Validation Energy MAE (meV)",
        )
        if key in metrics
    }
    if summary_metrics:
        print("Validation energy summary:", flush=True)
        for key in sorted(summary_metrics):
            print(f"{key}: {summary_metrics[key]}", flush=True)

    return metrics


def run() -> None:
    args = parse_args()
    config = load_validation_config(args.config, wandb=args.wandb)
    metrics = evaluate_loaded_model_on_validation(config, force_cpu=args.cpu)

    metrics_path = Path(args.metrics_file) if args.metrics_file else default_slurm_metrics_file()
    emit_metrics(metrics, metrics_path)


if __name__ == "__main__":
    run()
