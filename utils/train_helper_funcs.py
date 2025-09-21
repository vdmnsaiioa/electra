from torch.optim import Adam, AdamW, SGD
import torch
import numpy as np
import random
import os
import csv
import json


def set_optimizer(tens_net, config):
    """Instantiate the optimizer configured for ``tens_net``.

    If ``split_params`` is enabled in the configuration and the network
    exposes an ``energy_network`` submodule, a dedicated parameter group is
    created so a custom learning rate or weight decay can be configured via
    ``energy_initial_lr`` and ``energy_weight_decay``.  Otherwise all
    parameters share the base optimiser settings.
    """

    optimizer_name = config['optimizer']
    base_lr = config['initial_lr']
    base_weight_decay = config.get('weight_decay', 0.0)

    param_groups = []
    all_params = list(tens_net.parameters())

    energy_params = []
    if hasattr(tens_net, 'energy_network') and tens_net.energy_network is not None:
        energy_params = list(tens_net.energy_network.parameters())

    split_params = config.get('split_params', False)

    if split_params and energy_params:
        energy_param_ids = {id(p) for p in energy_params}
        other_params = [p for p in all_params if id(p) not in energy_param_ids]
    else:
        other_params = all_params
        energy_params = []

    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': base_weight_decay,
        })

    if energy_params:
        energy_lr = config.get('energy_initial_lr', base_lr)
        energy_weight_decay = config.get('energy_weight_decay', base_weight_decay)
        param_groups.append({
            'params': energy_params,
            'lr': energy_lr,
            'weight_decay': energy_weight_decay,
        })
        print(energy_lr,'cacucacu')

    optimizer_kwargs = {}
    if optimizer_name == 'adam':
        optimizer_cls = Adam
        optimizer_kwargs['amsgrad'] = True
    elif optimizer_name == 'adamw':
        optimizer_cls = AdamW
        optimizer_kwargs['amsgrad'] = True
    elif optimizer_name == 'sgd':
        optimizer_cls = SGD
        optimizer_kwargs['momentum'] = config['momentum']
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_name}")

    if not param_groups:
        # Fallback to the full parameter list if no groups were constructed.
        param_groups = all_params

    optimizer = optimizer_cls(param_groups,
                              lr=base_lr,
                              weight_decay=base_weight_decay,
                              **optimizer_kwargs)
    return optimizer


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_all_paths(config, wb_run_name=None):
    pred_dens_path_val = config['pred_dens_path_val']
    pred_dens_path_test = config['pred_dens_path_test']
    density_delta_path_val = config['density_delta_path_val']
    density_delta_path_test = config['density_delta_path_test']
    gaus_pos_path_val = config['gaus_pos_path_val']
    gaus_pos_path_test = config['gaus_pos_path_test']
    if config['niflheim']:
        pred_dens_path_val = config['niflheim_base_path'] + f"split_{config['data_split']}/" + pred_dens_path_val
        pred_dens_path_test = config['niflheim_base_path'] + f"split_{config['data_split']}/" + pred_dens_path_test
        density_delta_path_val = config['niflheim_base_path'] + f"split_{config['data_split']}/" + density_delta_path_val
        density_delta_path_test = config['niflheim_base_path'] + f"split_{config['data_split']}/" + density_delta_path_test
        gaus_pos_path_val = config['niflheim_base_path'] + f"split_{config['data_split']}/" + config['gaus_pos_path_val']
        gaus_pos_path_test = config['niflheim_base_path'] + f"split_{config['data_split']}/" + config['gaus_pos_path_test']
    if config['wandb']:
        pred_dens_path_val = pred_dens_path_val + f"/{wb_run_name}/"
        pred_dens_path_test = pred_dens_path_test + f"/{wb_run_name}/"
        density_delta_path_val = density_delta_path_val + f"/{wb_run_name}/"
        density_delta_path_test = density_delta_path_test + f"/{wb_run_name}/"
        gaus_pos_path_val = gaus_pos_path_val + f"/{wb_run_name}/"
        gaus_pos_path_test = gaus_pos_path_test + f"/{wb_run_name}/"
    config['pred_dens_path_val'] = pred_dens_path_val
    config['pred_dens_path_test'] = pred_dens_path_test
    config['density_delta_path_val'] = density_delta_path_val
    config['density_delta_path_test'] = density_delta_path_test
    config['gaus_pos_path_val'] = gaus_pos_path_val
    config['gaus_pos_path_test'] = gaus_pos_path_test
    os.makedirs(pred_dens_path_val, exist_ok=True)
    os.makedirs(pred_dens_path_test, exist_ok=True)
    os.makedirs(density_delta_path_val, exist_ok=True)
    os.makedirs(density_delta_path_test, exist_ok=True)
    os.makedirs(gaus_pos_path_val, exist_ok=True)
    os.makedirs(gaus_pos_path_test, exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    return config

def get_files(config):
    file_split = config['data_split']
    file_split_path = config['data_split_path'] + f"datasplits_{file_split}.json"
    f = open(file_split_path)
    data = json.load(f)
    train_files_indices = data['train']
    test_files_indices = data['test']
    validation_files_indices = data['validation']
    if file_split != "ECD" and file_split != "ECD_test" and file_split != "nmc":
        train_files = [f"{config['qm9_dens_path']}/{num:06}.CHGCAR.lz4" for num in train_files_indices]
        test_files = [f"{config['qm9_dens_path']}/{num:06}.CHGCAR.lz4" for num in test_files_indices]
        validation_files = [f"{config['qm9_dens_path']}/{num:06}.CHGCAR.lz4" for num in validation_files_indices]
    elif file_split == "ECD" or file_split == "ECD_test":
        train_files = [f"{config['qm9_dens_path']}/{num}.chgcar.lz4" for num in train_files_indices]
        test_files = [f"{config['qm9_dens_path']}/{num}.chgcar.lz4" for num in test_files_indices]
        validation_files = [f"{config['qm9_dens_path']}/{num}.chgcar.lz4" for num in validation_files_indices]
    else:
        train_files = [f"{config['qm9_dens_path']}/{num}.CHGCAR.lz4" for num in train_files_indices]
        test_files = [f"{config['qm9_dens_path']}/{num}.CHGCAR.lz4" for num in test_files_indices]
        validation_files = [f"{config['qm9_dens_path']}/{num}.CHGCAR.lz4" for num in validation_files_indices]
    random.shuffle(train_files)
    random.shuffle(test_files)
    random.shuffle(validation_files)

    return train_files, test_files, validation_files

def load_csv_to_dict(file_path, key_column, value_column):
    result_dict = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row[key_column]
            value = row[value_column]
            result_dict[key] = value
    return result_dict
