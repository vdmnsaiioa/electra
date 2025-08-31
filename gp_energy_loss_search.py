"""Gaussian process optimization for ELECTRA energy_loss_coef.

This script tunes the `energy_loss_coef` parameter in `hpc_conf_mol.yaml`
using Gaussian process-based Bayesian optimization. The goal is to minimize
NRMSE of the energy prediction. Each Gaussian process epoch launches a short
ELECTRA training run and evaluates the resulting NRMSE.

The search uses 20 evaluations (`n_calls`) and, for every candidate
`energy_loss_coef`, runs the ELECTRA model for 4000 training steps.
"""

from __future__ import annotations

from functools import partial
from typing import Dict

import numpy as np

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
import wandb

from train import load_base_config
from utils.train_helper_funcs import set_all_seeds, set_all_paths, get_files
from models.electra_hotpp.ELECTRA import ELECTRA


# ---------------------------------------------------------------------------
# Helper function to evaluate one energy_loss_coef value
# ---------------------------------------------------------------------------

def _evaluate(coef: float) -> float:
    """Train and evaluate ELECTRA with a given energy loss coefficient.

    Parameters
    ----------
    coef: float
        Energy loss coefficient to apply in the configuration.

    Returns
    -------
    float
        Validation energy NRMSE after 4000 training steps.
    """

    # Load base configuration and update the energy loss coefficient
    config: Dict = load_base_config()
    config["energy_loss_coef"] = float(coef)
    config["wandb"] = True  # enable wandb logging

    # Ensure deterministic behaviour
    set_all_seeds(config["seed"])

    # Set up WandB logger
    logger = WandbLogger(
        config=config,
        project=config["project_name"],
        log_model=False,
        group=f"Split_{config['data_split']}",
    )
    run_name = logger.experiment.name

    # Prepare model and data loaders
    config = set_all_paths(config, run_name=run_name)
    with open(".wandbignore", "w") as f:
        f.write(f"{config['model_dir']}/\n*.pth\n")

    # Prepare Lightning trainer for a short 4000-step run
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_steps=4000,
        logger=logger,
        enable_checkpointing=False,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
    )
    train_files, test_files, val_files = get_files(config)
    electra = ELECTRA(
        train_files=train_files,
        test_files=test_files,
        validation_files=val_files,
        model_handler=None,
        config=config,
    )

    train_loader = electra.train_dataloader()
    val_loader = electra.val_dataloader()
    seed_everything(config["seed"], workers=True)

    # Run training and validation
    trainer.fit(model=electra, train_dataloaders=train_loader, val_dataloaders=val_loader)
    metrics = trainer.validate(model=electra, dataloaders=val_loader, verbose=False)
    wandb.finish()

    # Extract the validation NRMSE; default to a large value if missing
    nrmse = metrics[0].get("Validation Energy NRMSE", np.inf)
    return float(nrmse)


# ---------------------------------------------------------------------------
# Gaussian process optimization routine
# ---------------------------------------------------------------------------

space = [Real(1e-4, 1.0, name="energy_loss_coef")]


@use_named_args(space)
def objective(**params):
    return _evaluate(params["energy_loss_coef"])


def run_search():
    """Execute the Gaussian process search for 20 epochs."""
    result = gp_minimize(objective, space, n_calls=20, random_state=0)
    print("Best energy_loss_coef:", result.x[0])
    print("Corresponding NRMSE:", result.fun)


if __name__ == "__main__":
    run_search()

