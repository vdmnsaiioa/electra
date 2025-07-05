import torch
from ase import Atoms
def grid2pos(mol: Atoms, grid_dict):
    """Convert a density grid to a position array efficiently."""
    # Extract cell dimensions and grid sizes
    cell = mol.get_cell()
    x_size, y_size, z_size = cell[0, 0], cell[1, 1], cell[2, 2]
    nx, ny, nz = grid_dict["nx"], grid_dict["ny"], grid_dict["nz"]

    # Generate the position grid directly with meshgrid
    x, y, z = (
        torch.linspace(0, x_size, nx, device="cuda" if torch.cuda.is_available() else "cpu"),
        torch.linspace(0, y_size, ny, device="cuda" if torch.cuda.is_available() else "cpu"),
        torch.linspace(0, z_size, nz, device="cuda" if torch.cuda.is_available() else "cpu"),
    )
    pos_grid = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1).to("cpu")

    return pos_grid




