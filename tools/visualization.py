from ase.calculators.vasp import VaspChargeDensity
from utils.custom_vasp_loader import CustomVaspChargeDensity
from tools.density_conversions import cd_to_chgcar_mol, cd_to_chgcar_mat
import numpy as np
import lz4
import tempfile
import os

def create_chg_delta(pred_dens_file: str,
                     true_dens_file: str,
                     delta_folder: str,
                     name_iter_str: str,
                     type: str = "mol"):
    vcd_pred = CustomVaspChargeDensity(pred_dens_file)
    with lz4.frame.open(true_dens_file, mode='rb') as fp:
        filecontent = fp.read()
    tmpfd, tmppath = tempfile.mkstemp(prefix="tmpchgcar")
    tmpfile = os.fdopen(tmpfd, "wb")
    tmpfile.write(filecontent)
    tmpfile.close()
    vcd_true = CustomVaspChargeDensity(tmppath)
    os.remove(tmppath)

    cd_pred = np.array(vcd_pred.chg, dtype=np.float64).squeeze(axis=0)
    cd_true = np.array(vcd_true.chg, dtype=np.float64).squeeze(axis=0)
    atoms = vcd_true.atoms[0]

    delta = cd_true - cd_pred
    filename = f'{delta_folder}/{name_iter_str}_DELTA.CHGCAR'
    if type == "mol":
        cd_to_chgcar_mol(atoms=atoms, cd=delta, filename=filename)
    else:
        cd_to_chgcar_mat(original_file=true_dens_file, atoms=atoms, cd=delta, filename=filename)
