from torch.optim import Adam, AdamW, SGD
import torch
import numpy as np
import random
import os
import csv
import json


def set_optimizer(tens_net, config):
    if config['optimizer'] == 'adam':
        optimizer = Adam(tens_net.parameters(),
                         lr=config['initial_lr'],
                         amsgrad=True,
                         weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = AdamW(tens_net.parameters(),
                          lr=config['initial_lr'],
                          amsgrad=True,
                          weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = SGD(tens_net.parameters(),
                        lr=config['initial_lr'],
                        momentum=config['momentum'],
                        weight_decay=config['weight_decay'])
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

    train_files = [f"{config['qm9_dens_path']}/{num:06}.CHGCAR.lz4" for num in train_files_indices]
    test_files = [f"{config['qm9_dens_path']}/{num:06}.CHGCAR.lz4" for num in test_files_indices]
    validation_files = [f"{config['qm9_dens_path']}/{num:06}.CHGCAR.lz4" for num in validation_files_indices]
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