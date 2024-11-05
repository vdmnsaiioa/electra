from pyscf import gto
import numpy
from pyscf import lib
from pyscf.dft import numint
from pyscf import gto, scf
from pyscf.tools import cubegen
from ase import Atoms
import os
from pyscf.tools.chgcar import density
from tools.density_conversions import chgcar_to_cd, cd_to_chgcar

def ase_atoms_to_pyscf(ase_atoms):
    return [[ase_atoms.get_chemical_symbols()[i], ase_atoms.get_positions()[i]] for i in
            range(len(ase_atoms.get_positions()))]


def get_sad_density_from_ase_atoms(atoms: Atoms,
                                   grid_dict,
                                   guess: str = "minao",
                                   type: str = 'cubegen',
                                   basis: str = 'minao'):
    spin = 0
    charge = 0
    mol = gto.M(
        atom=ase_atoms_to_pyscf(atoms),
        basis=basis,
        spin=spin,
        charge=charge)
    mf = scf.RHF(mol).run()
    if guess == "minao":
        dens_mat = mf.init_guess_by_minao()
    elif guess == "atom":
        dens_mat = mf.init_guess_by_atom()
    elif guess == '1e':
        dens_mat = mf.init_guess_by_1e()
    if type == 'cubegen':
        filename = f"{atoms.get_chemical_formula()}.cube"
        dens = cubegen.density(mol=mol,
                           outfile=filename,
                           dm=dens_mat,
                           nx=grid_dict['nx'],
                           ny=grid_dict['ny'],
                           nz=grid_dict['nz'])
    elif type == 'vasp':
        filename = f"{atoms.get_chemical_formula()}.CHGCAR"
        density(mol, filename, dens_mat, nx=grid_dict['nx'], ny=grid_dict['ny'], nz=grid_dict['nz'])
        dens = chgcar_to_cd(filename)
    os.remove(filename)
    return dens
