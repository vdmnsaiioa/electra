import argparse
import warnings
import yaml
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything

from utils.model_handling import ModelIO, get_tag
from utils.train_helper_funcs import set_all_seeds, set_all_paths, get_files
from utils.eq_check import equivariance_check
from models.electra_hotpp.ELECTRA import ELECTRA

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    """Parse CLI arguments allowing selective overrides of the YAML config."""
    parser = argparse.ArgumentParser(
        description="Run ELECTRA training with optional config overrides.")

    # Boolean flag for pruning
    prune_group = parser.add_mutually_exclusive_group()
    prune_group.add_argument("--prune", dest="prune", action="store_true",
                              help="Enable pruning (overrides config).")
    prune_group.add_argument("--no_prune", dest="prune", action="store_false",
                              help="Disable pruning (overrides config).")
    parser.set_defaults(prune=None)

    # Float overrides for learning rates
    parser.add_argument("--initial_lr", type=float, default=None,
                        help="Override the initial learning rate.")
    parser.add_argument("--final_lr", type=float, default=None,
                        help="Override the final learning rate.")
    parser.add_argument("--lr_gamma", type=float, default=None,
                        help="Override the lr gamma")

    load_model_group = parser.add_mutually_exclusive_group()
    load_model_group.add_argument("--load_model", dest="load_model", action="store_true",
                              help="Enable model loading (overrides config).")
    load_model_group.add_argument("--no_load_model", dest="load_model", action="store_false",
                              help="Disable model loading (overrides config).")
    parser.set_defaults(load_model=None)

    # Boolean flag for positional displacement functions
    pos_group = parser.add_mutually_exclusive_group()
    pos_group.add_argument("--use_pos_disp_functions", dest="use_pos_disp_functions", action="store_true",
                           help="Enable positional displacement functions (overrides config).")
    pos_group.add_argument("--no_use_pos_disp_functions", dest="use_pos_disp_functions", action="store_false",
                           help="Disable positional displacement functions (overrides config).")
    parser.set_defaults(use_pos_disp_functions=None)

    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Return a new config with CLI overrides applied when provided."""
    if args.prune is not None:
        config["prune"] = args.prune

    if args.initial_lr is not None:
        config["initial_lr"] = args.initial_lr

    if args.final_lr is not None:
        config["final_lr"] = args.final_lr

    if args.use_pos_disp_functions is not None:
        config["use_pos_disp_functions"] = args.use_pos_disp_functions
    if args.lr_gamma is not None:
        config["lr_gamma"] = args.lr_gamma
    if args.load_model is not None:
        config["load_model"] = args.load_model

    return config


def load_base_config() -> dict:
    """Load the base YAML configuration depending on GPU availability and flags inside the YAML."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        base_path = "/home/energy/jels/ELECTRA_CLEAN"
        config = yaml.safe_load(open(f"{base_path}/hpc_conf_mol.yaml"))
        if config['crystal']:
            sub = 'crystal_nmc' if config.get('nmc', False) else 'crystal'
            config = yaml.safe_load(open(f"{base_path}/hpc_conf_{sub}.yaml"))
        elif config.get('MD', False):
            config = yaml.safe_load(open(f"{base_path}/hpc_conf_md.yaml"))
    else:
        torch.set_num_threads(1)
        config = yaml.safe_load(open("config_mol.yaml"))
        if config['crystal']:
            sub = 'crystal_nmc' if config.get('data_split') == 'nmc_full' else 'crystal'
            config = yaml.safe_load(open(f"config_{sub}.yaml"))
        elif config.get('MD', False):
            config = yaml.safe_load(open("config_md.yaml"))
    return config


def run():
    args = parse_args()
    config = load_base_config()
    config = apply_overrides(config, args)

    # Seed alignment
    set_all_seeds(config['seed'])

    # Lightning Trainer defaults: auto strategy is fine for single-GPU
    accel = 'auto'
    devices = 1

    # Short test timeout
    max_time = "0:02:30:00" if config['data_split'] == 'test' else config['max_time']

    trainer = L.Trainer(
        accelerator=accel,
        devices=devices,
        logger=WandbLogger(
            config=config,
            project=config["project_name"],
            log_model=True,
            group=f"Split_{config['data_split']}"
        ) if config.get('wandb', False) else None,
        check_val_every_n_epoch=config['eval_every'],
        log_every_n_steps=1,
        max_epochs=config['max_epochs'],
        gradient_clip_val=(config['gradient_clip_value'] if config.get('clip_grad', False) else None),
        gradient_clip_algorithm='value',
        max_time=max_time,
    )

    # WandB naming and paths
    if config.get('wandb', False):
        wb_name = trainer.logger.experiment.name
        tag = get_tag(wb_name)
    else:
        wb_name = None
        tag = get_tag("test")

    config = set_all_paths(config, wb_name)
    with open(".wandbignore", "w") as f:
        f.write(f"{config['model_dir']}/\n*.pth\n")

    train_files, test_files, validation_files = get_files(config)
    model_handler = ModelIO(directory=config['model_dir'], tag=tag) if config.get('save_model', False) else None

    electra = ELECTRA(
        train_files=train_files,
        test_files=test_files,
        validation_files=validation_files,
        model_handler=model_handler,
        config=config,
    )

    if config.get("load_model", False):
        print(f"Loading model from {config['load_model_path']}")
        electra.load_state_dict(torch.load(config['load_model_path']))

    # DataLoaders and seed workers
    train_loader = electra.train_dataloader()
    val_loader = electra.val_dataloader()
    test_loader = electra.test_dataloader()
    seed_everything(config['seed'], workers=True)

    if config.get('eq_check', False):
        assert equivariance_check(electra, train_loader)

    # Run training and testing
    trainer.fit(model=electra, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(dataloaders=test_loader)


if __name__ == "__main__":
    run()

