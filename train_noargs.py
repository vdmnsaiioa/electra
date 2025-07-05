import yaml
import torch

from utils.model_handling import ModelIO, get_tag
from utils.train_helper_funcs import set_all_seeds, set_all_paths, get_files
import warnings
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from utils.eq_check import equivariance_check
from models.electra_hotpp.ELECTRA import ELECTRA

warnings.filterwarnings("ignore", category=UserWarning)


def run():
    if torch.cuda.is_available():
        stream = open("/home/energy/jels/ELECTRA_CLEAN/hpc_conf_mol.yaml")
        torch.set_float32_matmul_precision('high')
        config = yaml.safe_load(stream)
        if config['crystal']:
            if config['nmc']:
                stream = open("/home/energy/jels/ELECTRA_CLEAN/hpc_conf_crystal_nmc.yaml")
                config = yaml.safe_load(stream)
            else:
                stream = open("/home/energy/jels/ELECTRA_CLEAN/hpc_conf_crystal.yaml")
                config = yaml.safe_load(stream)
        elif config['MD']:
            stream = open("/home/energy/jels/ELECTRA_CLEAN/hpc_conf_md.yaml")
            config = yaml.safe_load(stream)
    else:
        torch.set_num_threads(1)
        stream = open("config_mol.yaml")
        torch.set_float32_matmul_precision('high')
        config = yaml.safe_load(stream)
        if config['crystal']:
            if config['data_split'] == "nmc_full":
                stream = open("config_crystal_nmc.yaml")
                config = yaml.safe_load(stream)
            else:
                stream = open("config_crystal.yaml")
                config = yaml.safe_load(stream)
        elif config['MD']:
            stream = open("config_md.yaml")
            config = yaml.safe_load(stream)
    set_all_seeds(config['seed'])
    acc = "gpu" if torch.cuda.is_available() else "cpu"
    strategy = "auto"
    if config['data_split'] == 'test':
        max_time = "0:02:30:00"
    else:
        max_time = config['max_time']

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
                        max_epochs=config['max_epochs'],
                        gradient_clip_algorithm='value',
                        max_time=max_time,
                        #detect_anomaly=True
                        )
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

    electra = ELECTRA(
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
    test_loader = electra.test_dataloader()
    seed_everything(config['seed'], workers=True)
    if config['eq_check']:
       assert(equivariance_check(electra, train_loader))
    trainer.fit(model=electra,
                train_dataloaders=train_loader,
                val_dataloaders=test_loader)

    trainer.test(dataloaders=electra.test_dataloader())

if __name__ == "__main__":
    run()

