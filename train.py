import os

import yaml
import torch

from utils.model_handling import ModelIO, get_tag
from utils.train_helper_funcs import set_optimizer, set_all_seeds, set_all_paths, get_files, load_csv_to_dict
import warnings
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
#from models.ELECTRA.ELECTRA_SOTA import ELECTRA
from models.electra_hotpp.ELECTRA_HOTPP import ELECTRA_hotpp
from ase.data import atomic_numbers
from utils.eq_check import equivariance_check

warnings.filterwarnings("ignore", category=UserWarning)


def run():
    if torch.cuda.is_available():
        stream = open("/home/energy/jels/electra_v2/niflheim_config.yaml")
        torch.set_float32_matmul_precision('highest')
    else:
        stream = open("config.yaml")
        torch.set_float32_matmul_precision('highest')
    config = yaml.safe_load(stream)

    set_all_seeds(config['seed'])

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc = "auto" if torch.cuda.is_available() else "cpu"
    strategy = "fsdp" if torch.cuda.is_available() else "auto"
    if config['data_split'] == 'test':
        max_time = "0:08:30:00"
    else:
        max_time = "6:16:00:00"

    trainer = L.Trainer(accelerator=acc,
                        gradient_clip_val=config['gradient_clip_value'] if config['clip_grad'] else None,
                        devices=1,
                        strategy=strategy,
                        logger=WandbLogger(config=config,
                                           project=config["project_name"],
                                           log_model=True,
                                           group=f"Split_{config['data_split']}") if config['wandb'] else None,
                        check_val_every_n_epoch=config['eval_every'],
                        log_every_n_steps=1,
                        #profiler="advanced",
                        max_epochs=250,
                        gradient_clip_algorithm='value',
                        max_time=max_time)
    if config['wandb']:
        wb_name = trainer.logger.experiment.name
        tag = get_tag(wb_name)
    else:
        wb_name = None
        tag = get_tag("test")
    config = set_all_paths(config, wb_name)
    with open(".wandbignore", "w") as f:
        f.write(f"{config['model_dir']}/\n*.pth\n")
    train_files, test_files, validation_files = get_files(config)

    if config['save_model']:
        model_handler = ModelIO(directory=config['model_dir'], tag=tag)
    else:
        model_handler = None

    if config['model_type'] == "hotpp":
        electra = ELECTRA_hotpp(
                          train_files=train_files,
                          test_files=test_files,
                          validation_files=validation_files,
                          model_handler=model_handler,
                          config=config)
    if config["load_model"]:
        print(f"Loading model from {config['load_model_path']}")
        electra.load_state_dict(torch.load(config['load_model_path']))

    train_loader = electra.train_dataloader()
    val_loader = electra.val_dataloader()
    seed_everything(config['seed'], workers=True)
    if config['eq_check']:
       assert(equivariance_check(electra, train_loader))
    trainer.fit(model=electra,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    trainer.test(dataloaders=electra.test_dataloader())

if __name__ == "__main__":
    run()
