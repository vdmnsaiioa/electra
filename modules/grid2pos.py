import torch
from ase import Atoms
import numpy as np

def grid2pos(mol: Atoms, grid_dict):
    """
    Convert a density grid to a position array efficiently.

    If the cell is orthogonal (i.e., diagonal with no off-diagonal components),
    we can directly linspace along x, y, z. Otherwise, we generate a 3D mesh
    in fractional coordinates and transform to real space using the full cell
    matrix.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract cell dimensions and grid sizes
    cell = mol.get_cell()
    nx, ny, nz = grid_dict["nx"], grid_dict["ny"], grid_dict["nz"]
    cell_t = torch.tensor(cell, dtype=torch.float, device=device)  # shape (3,3)

    # Check if the cell is orthogonal (off-diagonal elements ~ 0)
    tol = 1e-8
    diag = torch.diag(torch.diagonal(cell_t))
    offdiag = (cell_t - diag).abs().max()
    is_ortho = offdiag < tol
    is_ortho = not mol.pbc.all()
    if is_ortho:
        # Orthogonal case
        x_size, y_size, z_size = cell[0, 0], cell[1, 1], cell[2, 2]
        #x = torch.linspace(0, x_size, nx, device=device)
        #y = torch.linspace(0, y_size, ny, device=device)
        #z = torch.linspace(0, z_size, nz, device=device)
        x = (torch.arange(nx, device=device, dtype=torch.float32) / nx) * x_size
        y = (torch.arange(ny, device=device, dtype=torch.float32) / ny) * y_size
        z = (torch.arange(nz, device=device, dtype=torch.float32) / nz) * z_size
        pos_grid = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)

    else:
        # Non-orthogonal case: fractional -> real
        fx = torch.arange(nx, device=device, dtype=torch.float32) / nx
        fy = torch.arange(ny, device=device, dtype=torch.float32) / ny
        fz = torch.arange(nz, device=device, dtype=torch.float32) / nz
        Fx, Fy, Fz = torch.meshgrid(fx, fy, fz, indexing="ij")
        frac_coords = torch.stack([Fx, Fy, Fz], dim=-1)  # shape (nx, ny, nz, 3)
        frac_2d = frac_coords.reshape(-1, 3)  # shape (nx*ny*nz, 3)

        # Real space = fractional * cell
        real_2d = frac_2d @ cell_t  # (nx*ny*nz, 3)
        pos_grid = real_2d.reshape(nx, ny, nz, 3)

    # Return the final grid on CPU for general usage
    return pos_grid.to("cpu")

class LazyMeshGrid():
    def __init__(self, cell, grid_step, origin=None, adjust_grid_step=False):
        self.cell = cell
        if adjust_grid_step:
            n_steps = np.round(self.cell.lengths()/grid_step)
            self.scaled_grid_vectors = [np.arange(n)/n for n in n_steps]
            self.adjusted_grid_step = self.cell.lengths()/n_steps
        else:
            self.scaled_grid_vectors = [np.arange(0, l, grid_step)/l for l in self.cell.lengths()]
        self.shape = np.array([len(g) for g in self.scaled_grid_vectors] + [3])
        if origin is None:
            self.origin = np.zeros(3)
        else:
            self.origin = origin

        self.origin = np.expand_dims(self.origin, 0)

    def __getitem__(self, indices):
        indices = np.array(indices)
        indices_shape = indices.shape
        if not (len(indices_shape) == 2 and indices_shape[0] == 3):
            raise NotImplementedError("Indexing must be a 3xN array-like object")
        gridA = self.scaled_grid_vectors[0][indices[0]]
        gridB = self.scaled_grid_vectors[1][indices[1]]
        gridC = self.scaled_grid_vectors[2][indices[2]]

        grid_pos = np.stack([gridA, gridB, gridC], 1)
        grid_pos = np.dot(grid_pos, self.cell)
        grid_pos += self.origin

        return grid_pos
def grid2pos_scdp(atoms, grid_dict):
    """
    Create real-space grid positions for a periodic cell.

    Args:
        atoms: ASE Atoms object containing cell information.
        grid_dict: Dictionary with grid specifications, must contain:
                   - 'lx', 'ly', 'lz': the number of grid points in each dimension.
                   Optionally, it can contain:
                   - 'origin': an optional origin offset.
                   
    Returns:
        grid_pos: A tensor of shape (lx, ly, lz, 3) with the real-space coordinates.
    """
    lx, ly, lz = grid_dict['nx'], grid_dict['ny'], grid_dict['nz']

    # Obtain the cell from the atoms object and ensure it is a torch tensor.
    # (Assumes atoms.cell is convertible; adjust dtype/device as needed.)
    cell = torch.tensor(atoms.cell, dtype=torch.float32, device=torch.device("cpu"))
    if hasattr(atoms, "positions"):
        # If atoms.positions is available, use its device.
        cell = cell.to(torch.tensor(atoms.positions).device)

    # Create fractional grid coordinates in [0, 1) along each axis.
    frac_x = torch.arange(lx, device=cell.device, dtype=torch.float32) / lx
    frac_y = torch.arange(ly, device=cell.device, dtype=torch.float32) / ly
    frac_z = torch.arange(lz, device=cell.device, dtype=torch.float32) / lz

    # Create a meshgrid of fractional coordinates using indexing "ij"
    grid_frac = torch.meshgrid(frac_x, frac_y, frac_z, indexing="ij")
    grid_frac = torch.stack(grid_frac, dim=3)  # Shape: (lx, ly, lz, 3)

    # Map fractional coordinates into real space by multiplying with the cell matrix.
    # Note: We multiply with cell.T because each fractional vector is a row vector.
    grid_pos = torch.matmul(grid_frac, cell.T)

    # Optionally add an origin offset if provided in grid_dict.
    if "origin" in grid_dict and grid_dict["origin"] is not None:
        origin = torch.tensor(grid_dict["origin"], dtype=grid_pos.dtype, device=grid_pos.device)
        grid_pos = grid_pos + origin

    return grid_pos

