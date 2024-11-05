import torch
from ase import Atoms
def grid2pos(mol: Atoms,
             grid_dict):
    """ Convert a density grid to a position array """
    # Get the cell dimensions
    x_size = mol.get_cell()[0][0]
    y_size = mol.get_cell()[1][1]
    z_size = mol.get_cell()[2][2]

    # Get the grid dimensions
    nx = grid_dict["nx"]
    ny = grid_dict["ny"]
    nz = grid_dict["nz"]

    # Create the position array

    # Generate the linspace for each dimension
    x = torch.linspace(0, x_size, nx)
    y = torch.linspace(0, y_size, ny)
    z = torch.linspace(0, z_size, nz)

    # Create a meshgrid
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # Stack the coordinates to form the position grid
    pos_grid = torch.stack((X, Y, Z), dim=-1)

    return pos_grid



