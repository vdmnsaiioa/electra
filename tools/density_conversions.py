#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
from ase import Atoms
from ase.calculators.vasp import VaspChargeDensity
from utils.custom_vasp_loader import CustomVaspChargeDensity
import torch
from tools.atom_tools import valence_electrons
import time
import os
import tempfile
import lz4

def convert_to_unnormalized_density(orig_density: np.array,
                                    normalized_transported_density: np.array):
    unnormalized_transported_density = normalized_transported_density * orig_density.sum() / 10
    return unnormalized_transported_density

def normalize_density(n_elec: int,
                          density: torch.tensor,
                          qm9_density: torch.tensor,
                          grid_dict: dict,
                          sys: Atoms,
                         points: torch.tensor,
                        mol_volume: float = None):
    if mol_volume is not None:
        dv = mol_volume / (grid_dict['nx'] * grid_dict['ny'] * grid_dict['nz'])
    else:
        dv = sys.get_volume() / (grid_dict['nx'] * grid_dict['ny'] * grid_dict['nz'])
    if points is None:
        n_elec = n_elec
    else:
        n_elec = qm9_density.flatten()[points].sum() * dv
    integrated_num_electrons = density.sum() * dv
    factor = n_elec / integrated_num_electrons
    cd_sum_to_num_electrons = density * factor
    return cd_sum_to_num_electrons, n_elec


def get_electron_normalized_density(sys: Atoms,
                                    gridrefinement: int,
                                    cd: np.array):
    # This function returns a charge density that actually sums to the required number of electrons by just summing naively (with no integration)
    # Probably not required for anything.
    dv = sys.get_volume() / sys.calc.get_number_of_grid_points().prod()
    integrated_num_electrons = cd.sum() * dv / gridrefinement ** 3
    cd_sum_to_num_electrons = cd / cd.sum() * integrated_num_electrons
    return cd_sum_to_num_electrons


def check_number_of_electrons(sys: Atoms,
                              cd: np.array,
                              gridrefinement: int = 2,
                              grid_dict: dict = None):
    # Get integrated values

    if grid_dict:
        dv = sys.get_volume() / (grid_dict['nx'] * grid_dict['ny'] * grid_dict['nz'])
        integrated_num_electrons = cd.sum() * dv
    else:
        dv = sys.get_volume() / (grid_dict['nx'] * grid_dict['ny'] * grid_dict['nz'])
        integrated_num_electrons = cd.sum() * dv / gridrefinement ** 3
    # print(f"Number of electrons in {sys.get_chemical_formula()} is {np.round(integrated_num_electrons, 2)}")
    return integrated_num_electrons


def chgcar_to_cd(chgcar_file):
    with lz4.frame.open(chgcar_file, mode='rb') as fp:
        filecontent = fp.read()
    tmpfd, tmppath = tempfile.mkstemp(prefix="tmpchgcar")
    with os.fdopen(tmpfd, "wb") as tmpfile:
        tmpfile.write(filecontent)
    vcd = CustomVaspChargeDensity(tmppath)
    os.remove(tmppath)
    charge_density = vcd.chg
    try:
        atoms = vcd.atoms[0]
    except:
        print(f"No atoms in chgcar file {chgcar_file}")
    cd = torch.FloatTensor(charge_density).squeeze(axis=0)
    return cd, atoms

def cd_to_chgcar(atoms: Atoms, cd: np.array, filename: str):
    with open(filename, 'w') as fp:
        pass
    vcd = CustomVaspChargeDensity(filename)
    vcd.chg = [cd]
    vcd.atoms = [atoms]
    vcd.write(filename, format='chgcar')


def get_qm9_density(qm9_path):
    qm9_density, mol = chgcar_to_cd(qm9_path)
    grid_dict = {"nx": qm9_density.shape[0],
                      "ny": qm9_density.shape[1],
                      "nz": qm9_density.shape[2]}
    n_elec_qm9 = valence_electrons(mol.get_chemical_formula())
    return qm9_density, mol, n_elec_qm9, grid_dict


