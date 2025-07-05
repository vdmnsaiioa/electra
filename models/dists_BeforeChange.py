import torch
from torch.distributions import Categorical
from models.custom_dists import MixtureSameFamily, MultivariateNormal, MixtureSameFamilyMulti
#from pytorch3d.ops import ball_query

def compile_with_dynamic_shapes(fn):
    if torch.cuda.is_available():
        torch._dynamo.config.dynamic_shapes = True
        return torch.compile(fn)
    else:
        return fn

class gto():
    def __init__(self,
                 pos: torch.tensor,
                 pos_disp: torch.tensor,
                 cov: torch.tensor,
                 scales: torch.tensor,
                 scalar_mults: torch.tensor,
                 cut_distance: int,
                 relu: bool,
                 n_multiples: torch.tensor,
                 device: torch.device,
                 use_links: bool,
                 cell: torch.tensor = None,
                 mask: torch.tensor = None,
                 r_mat: torch.tensor = None,
                 pos_nonrot: torch.tensor = None,
                 pos_disp_nonrot: torch.tensor = None,):

        # Create component distributions (multivariate normal)
        # num_repeats = int(cov.size(0) / pos.size(1))
        self.scalar_mults = scalar_mults
        self.cov = cov
        self.scales = scales
        self.n_multiples = n_multiples
        self.relu = relu
        self.pos = pos
        self.pos_disp = pos_disp
        self.cov3 = None

        n_multiples = n_multiples.to(device)
        pos = pos.to(device)
        pos = pos.reshape(n_multiples.shape[0], -1)
        pos_tens_orig = pos.repeat_interleave(n_multiples, 0).view(-1, 3)
        # pos_tens = pos.repeat(1, int(n_multiples.sum()), 1).view(-1, 3)
        pos_tens_orig = pos_tens_orig.to(device)
        pos_tens = pos_tens_orig + pos_disp

        if r_mat is not None:
            pos_nonrot = pos_nonrot.reshape(1, pos_nonrot.shape[0], pos_nonrot.shape[1])
            pos_nonrot = pos_nonrot.to(device)
            pos_tens_nonrot = pos_nonrot.repeat_interleave(n_multiples, 1).view(-1, 3)
            #pos_tens_nonrot = pos_nonrot.repeat(1, int(n_multiples.sum()), 1).view(-1, 3)
            pos_tens_nonrot = pos_tens_nonrot.to(device)
            pos_tens_nonrot = pos_tens_nonrot + pos_disp_nonrot
            #assert (torch.allclose(pos_tens, pos_tens_nonrot @ r_mat.T, atol=3e-5))

        self.component_distributions = MultivariateNormal(loc=pos_tens,
                                                          covariance_matrix=cov,
                                                          validate_args=False,
                                                          r_mat=r_mat)

        # Create weights for each component
        self.weights = scales.squeeze(-1).squeeze(-1)

        # Create a categorical distribution to choose component distributions
        mixture_distribution = Categorical(self.weights if mask is None else self.weights * mask.squeeze(-1))

        self.dist = MixtureSameFamily(mixture_distribution,
                                    self.component_distributions,
                                scal_mults=scalar_mults,
                                relu=relu,
                                validate_args=False,
                                cell=cell)
        self.use_links = use_links
        self.cut_distance = cut_distance

    def construct_pos_def(self, matrix: torch.tensor):
        """
        Make a non-positive definite matrix positive definite by adding a small epsilon to its diagonal.

        Args:
            matrix (torch.Tensor): Input matrix.

        Returns:
            torch.Tensor: Positive definite matrix.
        """
        eps = 1e-6  # Small epsilon
        matrix += torch.eye(3).unsqueeze(0).expand(matrix.shape) * eps
        return matrix

    def get_dens_at_points(self,
                           points: torch.tensor,
                           links: torch.tensor):
        if not self.use_links:
            links = None
        log_prob = self.dist.log_prob(points, links=links)
        if self.scalar_mults is not None:
            prob_dens = log_prob
        else:
            prob_dens = torch.exp(log_prob)
            #prob_dens = log_prob
        return prob_dens

def get_full_grid_dens(atom_dist: gto,
                       gaus_pos: torch.tensor,
                       n_multiples: torch.tensor,
                       pos_grid: torch.tensor):
    grid_shape = pos_grid.shape[0:3]
    pos_grid = pos_grid.view(-1, 3)
    if atom_dist.use_links:
        links = get_links(pos_grid=pos_grid_r_s,
                                gaus_pos=gaus_pos,
                                cut_distance=cut_distance,
                                  cell=cell.to(cur_dev)if cell is not None else None,
                                )
        links = links.to(cur_dev)
    else:
        links = None
    #idx_tens = torch.cat([torch.tensor(val * [i]) for i, val in enumerate(n_multiples)], dim=0)
    #atom_distances_mask = torch.index_select(split_masks[0], dim=1, index=idx_tens)
    dens = atom_dist.get_dens_at_points(pos_grid, links=links)
    dens = dens.view(grid_shape)
    return dens

#@compile_with_dynamic_shapes
def get_n_points_dens(atom_dist: gto,
                      pos_grid: torch.tensor,
                      gaus_pos: torch.tensor,
                      n_multiples: torch.tensor,
                      n_points: int,
                      sample_all: bool = False,
                      r_mat: torch.tensor = None,
                      grid_nonrot: torch.tensor = None,
                      cell: torch.tensor = None):
    cut_distance = atom_dist.cut_distance
    grid_shape = pos_grid.shape[0:3]
    if torch.cuda.is_available():
        store_device = torch.device("cuda:0")
    else:
        store_device = torch.device("cpu")
    pos_grid = pos_grid.reshape(-1, 3)

    dens = torch.zeros_like(pos_grid[:, 0], dtype=torch.float, device=store_device)
    num_cuda = torch.cuda.device_count()

    if sample_all:
        N = pos_grid.size(0)
        device = store_device
        pos_grid_r_split = torch.split(pos_grid, n_points)
        points_ind = torch.arange(0, pos_grid.size(0), device=store_device)
        # 1) Shuffle them
        perm = torch.randperm(N, device=device)
        # 2) Apply the same permutation to BOTH pos_grid and points_ind
        pos_grid_shuffled = pos_grid[perm]
        points_ind_shuffled = points_ind[perm]

        # 3) Split the shuffled data
        pos_grid_shuffled_split = torch.split(pos_grid_shuffled, n_points)
        points_ind_shuffled_split = torch.split(points_ind_shuffled, n_points)

        dens = torch.zeros(N, device=device)
        points_ind_split = torch.split(points_ind, n_points)
        # Adjust the second column (i.e., the atom indices) based on the desired multiples
        if torch.cuda.is_available():
            for i in range(1, num_cuda):
                locals()[f"atom_dist_{i}"] = create_dist_copy_on_device(atom_dist, torch.device(f'cuda:{i}'))
            for i, (pos_grid_r_s, points_ind_s) in enumerate(zip(pos_grid_shuffled_split, points_ind_shuffled_split)):
                #if i == 0:
                    #dens_split = atom_dist.get_dens_at_points(pos_grid_r_s).to(store_device)
                #else:
                if num_cuda == 1:
                    cuda_num = 0
                    cur_dev = torch.device(f'cuda:{cuda_num}')
                else:
                    cuda_num = 1 + (i % (num_cuda - 1))
                    cur_dev = torch.device(f'cuda:{cuda_num}')
                    pos_grid_r_s = pos_grid_r_s.to(cur_dev)
                    gaus_pos = gaus_pos.to(cur_dev)
                if atom_dist.use_links:
                    links = get_links(pos_grid=pos_grid_r_s,
                                gaus_pos=gaus_pos,
                                cut_distance=cut_distance,
                                  cell=cell.to(cur_dev)if cell is not None else None,
                                )
                    links = links.to(cur_dev)
                else:
                    links = None
                if num_cuda == 1:
                    #idx_tens = torch.cat([torch.tensor(val * [i], device=cur_dev) for i, val in enumerate(n_multiples)], dim=0)
                    #atom_distances_mask = torch.index_select(split_mask, dim=1, index=idx_tens)
                    dens_split = atom_dist.get_dens_at_points(pos_grid_r_s, links=links)
                else:
                    #idx_tens = torch.cat([torch.tensor(val * [i], device=cur_dev) for i, val in enumerate(n_multiples)], dim=0)
                    #atom_distances_mask = torch.index_select(split_mask, dim=1, index=idx_tens)
                    dens_split = locals()[f"atom_dist_{cuda_num}"].get_dens_at_points(pos_grid_r_s, links=links).to(store_device)
                dens[points_ind_s] = dens_split
        else:
            for pos_grid_r_s, points_ind_s in zip(pos_grid_r_split, points_ind_split):
                #idx_tens = torch.cat([torch.tensor(val*[i]) for i, val in enumerate(n_multiples)], dim=0)
                #atom_distances_mask = torch.index_select(split_mask, dim=1, index=idx_tens)
                links = get_links(pos_grid=pos_grid_r_s,
                          gaus_pos=gaus_pos,
                          cut_distance=cut_distance,
                                  cell=cell
                          )
                dens_split = atom_dist.get_dens_at_points(pos_grid_r_s, links=links)
                dens[points_ind_s] = dens_split
    dens = dens.view(grid_shape)
    return dens, points_ind

def create_dist_copy_on_device(dist: gto,
                               new_device: torch.device):
    if dist.cov3 is not None:
        new_dist = gto_multi(pos=dist.pos.to(new_device),
                                            pos_disp=dist.pos_disp.to(new_device),
                                            cov1=dist.cov1.to(new_device),
                                               cov2=dist.cov2.to(new_device),
                                               cov3=dist.cov3.to(new_device),
                                            scales=dist.scales.to(new_device),
                                            n_multiples=dist.n_multiples,
                                            scalar_mults=dist.scalar_mults.to(new_device) if dist.scalar_mults is not None else None,
                                            cut_distance=dist.cut_distance,
                                            use_links=dist.use_links,
                                            relu=dist.relu,
                                            device=new_device)
    else:
        new_dist = gto(pos=dist.pos.to(new_device),
                       pos_disp=dist.pos_disp.to(new_device),
                       cov=dist.cov.to(new_device),
                       scales=dist.scales.to(new_device),
                       n_multiples=dist.n_multiples,
                       scalar_mults=dist.scalar_mults.to(new_device) if dist.scalar_mults is not None else None,
                       cut_distance=dist.cut_distance,
                       use_links=dist.use_links,
                       relu=dist.relu,
                       device=new_device)
    return new_dist

#@compile_with_dynamic_shapes
def get_links_old(pos_grid: torch.tensor,
              gaus_pos: torch.tensor,
              cut_distance: float):
    """
    Get the links between the points and the atoms based on a cutoff distance.

    Args:
        points (torch.Tensor): Points.
        atom_positions (torch.Tensor): Atom positions.
        cut_distance (float): Cutoff distance.

    Returns:
        torch.Tensor: Links.
    """
    gaus_pos = gaus_pos.to(torch.float)
    gaus_distances = torch.cdist(pos_grid, gaus_pos)
    max_distances = gaus_distances.max(dim=-1, keepdim=True).values
    # Logical mask with minimal recomputation
    links = torch.nonzero((gaus_distances >= max_distances) | (gaus_distances < cut_distance))
    return links

@compile_with_dynamic_shapes
def get_links_per_cdist(
        pos_grid: torch.Tensor,
        gaus_pos: torch.Tensor,
        cut_distance: float,
        cell: torch.Tensor,
) -> torch.Tensor:
    """
    Returns a (K, 2) integer tensor of (i, j) pairs, where:
      - i is row index in pos_grid
      - j is row index in gaus_pos
    so that:
      1) distance(pos_grid[i], gaus_pos[j]) < cut_distance
         OR
      2) j is the *farthest* atom for point i.

    Everything is computed in half-precision to speed up cdist.
    """

    device = pos_grid.device

    # 1) Cast inputs to half-precision (float16)
    pos_grid_half = pos_grid.to(dtype=torch.float16)
    gaus_pos_half = gaus_pos.to(dtype=torch.float16)

    # If you want your cutoff also in half precision for the comparison:
    cut_distance_half = torch.tensor(cut_distance, dtype=torch.float16, device=device)

    # 2) Compute pairwise distances (in half).
    #    PyTorch's cdist does a sqrt-based Euclidean distance in half-precision here.
    dist_half = torch.cdist(pos_grid_half, gaus_pos_half)  # shape: [N, M]
    if cell is not None:
        cell_half = cell.to(dtype=torch.float16)
        cell_half = cell_half.to(device)
        dist_half = torch.min(dist_half, torch.cdist(pos_grid_half, gaus_pos_half + cell_half.squeeze(-1)))
        dist_half = torch.min(dist_half, torch.cdist(pos_grid_half, gaus_pos_half - cell_half.squeeze(-1)))

    # 3) Find the farthest atom index for each row i (the max distance along the columns)
    min_vals, min_idx = dist_half.min(dim=1)  # shape [N], [N]

    # 4) Create a boolean mask for "distance < cut_distance"
    mask = dist_half < cut_distance_half  # shape [N, M], dtype=bool

    # 5) Ensure the farthest-atom column for each row i is also included
    #    i.e., for each i, we set mask[i, max_idx[i]] = True
    rows = torch.arange(pos_grid_half.shape[0], device=device)
    mask[rows, min_idx] = True

    # 6) Convert mask into a (K,2) index tensor
    #    mask.nonzero(as_tuple=False) => shape (K,2), with columns [row, col]
    links = mask.nonzero(as_tuple=False)  # each row = [i, j]
    # links is on the same device as 'mask'. It's already a LongTensor in new PyTorch versions.

    return links


def get_links(pos_grid: torch.Tensor,
                        gaus_pos: torch.Tensor,
                        cell: torch.Tensor,
                        cut_distance: float) -> torch.Tensor:
    """
    Computes links (i,j) between grid points and Gaussians using the minimum image convention.

    Args:
        pos_grid (torch.Tensor): Grid points of shape (N, 3).
        gaus_pos (torch.Tensor): Gaussian centers of shape (M, 3).
        cell (torch.Tensor): Cell matrix (3, 3) with lattice vectors as rows.
        cut_distance (float): Cutoff distance.

    Returns:
        torch.Tensor: A (K, 2) integer tensor of indices where the distance (using minimum image)
                      is below the cutoff, or the closest Gaussian if none are within the cutoff.
    """
    device = pos_grid.device
    # 1) Cast inputs to half-precision (float16)
    pos_grid_half = pos_grid.to(dtype=torch.float16)
    gaus_pos_half = gaus_pos.to(dtype=torch.float16)

    # If you want your cutoff also in half precision for the comparison:
    cut_distance_half = torch.tensor(cut_distance, dtype=torch.float16, device=device)
    # Compute pairwise difference: shape (N, M, 3)
    diff = pos_grid_half.unsqueeze(1) - gaus_pos_half.unsqueeze(0)
    #cell_half = cell.to(dtype=torch.float16)
    # Apply minimum image convention
    diff_min = minimum_image_vector(diff, cell)
    # Compute distances
    distances = diff_min.norm(dim=-1)

    # Create a mask for distances below the cutoff.
    mask = distances < cut_distance_half
    # For each grid point, also ensure that at least one Gaussian is chosen:
    # Find the index of the closest Gaussian
    min_vals, min_idx = distances.min(dim=1)
    # Ensure that the closest Gaussian is always included
    rows = torch.arange(pos_grid.shape[0], device=pos_grid.device)
    mask[rows, min_idx] = True
    # Convert the mask to a list of (i, j) index pairs.
    links = mask.nonzero(as_tuple=False)
    return links

def get_mask(pos_grid: torch.tensor,
              gaus_pos: torch.tensor,
              cut_distance: float):
    """
    Get the links between the points and the atoms based on a cutoff distance.

    Args:
        points (torch.Tensor): Points.
        atom_positions (torch.Tensor): Atom positions.
        cut_distance (float): Cutoff distance.

    Returns:
        torch.Tensor: Links.
    """
    ## This part needs to be optimized
    gaus_pos = gaus_pos.to(torch.float)
    gaus_distances = torch.cdist(pos_grid, gaus_pos)

    # Compute max distance per row only once
    max_distances = gaus_distances.max(dim=-1, keepdim=True).values

    # Logical mask with minimal recomputation
    gaus_distances_mask = torch.logical_not(gaus_distances < max_distances) | (gaus_distances < cut_distance)
    return gaus_distances_mask


def get_n_points_dens_single(atom_dist: gto,
                      pos_grid: torch.tensor,
                      gaus_pos: torch.tensor,
                      n_multiples: torch.tensor,
                      n_points: int,
                      sample_all: bool = False,
                      r_mat: torch.tensor = None,
                      grid_nonrot: torch.tensor = None,
                             cell: torch.tensor = None):
    cut_distance = atom_dist.cut_distance
    grid_shape = pos_grid.shape[0:3]
    if torch.cuda.is_available():
        store_device = torch.device("cuda:0")
    else:
        store_device = torch.device("cpu")
    pos_grid = pos_grid.reshape(-1, 3)

    dens = torch.zeros_like(pos_grid[:, 0], dtype=torch.float, device=store_device)

    if sample_all:
        pos_grid_r_split = torch.split(pos_grid, n_points)
        points_ind = torch.arange(0, pos_grid.size(0), device=store_device)
        points_ind_split = torch.split(points_ind, n_points)
        # Adjust the second column (i.e., the atom indices) based on the desired multiples
        for i, (pos_grid_r_s, points_ind_s) in enumerate(zip(pos_grid_r_split, points_ind_split)):
            if atom_dist.use_links:
                links = get_links(pos_grid=pos_grid_r_s,
                            gaus_pos=gaus_pos,
                            cut_distance=cut_distance,
                              cell=cell
                            )
            else:
                links = None
            dens_split = atom_dist.get_dens_at_points(pos_grid_r_s, links=links)
            dens[points_ind_s] = dens_split

    dens = dens.view(grid_shape)
    return dens, points_ind

class gto_multi():
    def __init__(self,
                 pos: torch.tensor,
                 pos_disp: torch.tensor,
                 cov1: torch.tensor,
                 cov2: torch.tensor,
                 cov3: torch.tensor,
                 scales: torch.tensor,
                 scalar_mults: torch.tensor,
                 cut_distance: int,
                 relu: bool,
                 n_multiples: torch.tensor,
                 device: torch.device,
                 use_links: bool,
                 mask: torch.tensor = None,
                 r_mat: torch.tensor = None,
                 pos_nonrot: torch.tensor = None,
                 pos_disp_nonrot: torch.tensor = None):

        # Create component distributions (multivariate normal)
        # num_repeats = int(cov.size(0) / pos.size(1))
        self.scalar_mults = scalar_mults
        self.cov1 = cov1
        self.cov2 = cov2
        self.cov3 = cov3
        self.scales = scales
        self.n_multiples = n_multiples
        self.relu = relu
        self.pos = pos
        self.pos_disp = pos_disp

        n_multiples = n_multiples.to(device)
        pos = pos.to(device)
        pos = pos.reshape(n_multiples.shape[0], -1)
        pos_tens_orig = pos.repeat_interleave(n_multiples, 0).view(-1, 3)
        # pos_tens = pos.repeat(1, int(n_multiples.sum()), 1).view(-1, 3)
        pos_tens_orig = pos_tens_orig.to(device)
        pos_tens = pos_tens_orig + pos_disp

        if r_mat is not None:
            pos_nonrot = pos_nonrot.reshape(1, pos_nonrot.shape[0], pos_nonrot.shape[1])
            pos_nonrot = pos_nonrot.to(device)
            pos_tens_nonrot = pos_nonrot.repeat_interleave(n_multiples, 1).view(-1, 3)
            #pos_tens_nonrot = pos_nonrot.repeat(1, int(n_multiples.sum()), 1).view(-1, 3)
            pos_tens_nonrot = pos_tens_nonrot.to(device)
            pos_tens_nonrot = pos_tens_nonrot + pos_disp_nonrot
            #assert (torch.allclose(pos_tens, pos_tens_nonrot @ r_mat.T, atol=3e-5))

        self.component_distributions1 = MultivariateNormal(loc=pos_tens,
                                                          covariance_matrix=cov1,
                                                          validate_args=False,
                                                          r_mat=r_mat)
        self.component_distributions2 = MultivariateNormal(loc=pos_tens,
                                                            covariance_matrix=cov2,
                                                                validate_args=False,
                                                                r_mat=r_mat)
        self.component_distributions3 = MultivariateNormal(loc=pos_tens,
                                                            covariance_matrix=cov3,
                                                                validate_args=False,
                                                                r_mat=r_mat)

        # Create weights for each component
        self.weights = scales.squeeze(-1).squeeze(-1)

        # Create a categorical distribution to choose component distributions
        mixture_distribution = Categorical(self.weights if mask is None else self.weights * mask.squeeze(-1))

        self.dist = MixtureSameFamilyMulti(mixture_distribution,
                                    self.component_distributions1,
                                    self.component_distributions2,
                                    self.component_distributions3,
                                scal_mults=scalar_mults,
                                relu=relu,
                                validate_args=False)
        self.use_links = use_links
        self.cut_distance = cut_distance

    def construct_pos_def(self, matrix: torch.tensor):
        """
        Make a non-positive definite matrix positive definite by adding a small epsilon to its diagonal.

        Args:
            matrix (torch.Tensor): Input matrix.

        Returns:
            torch.Tensor: Positive definite matrix.
        """
        eps = 1e-6  # Small epsilon
        matrix += torch.eye(3).unsqueeze(0).expand(matrix.shape) * eps
        return matrix

    def get_dens_at_points(self,
                           points: torch.tensor,
                           links: torch.tensor):
        if not self.use_links:
            links = None
        log_prob = self.dist.log_prob(points, links=links)
        if self.scalar_mults is not None:
            prob_dens = log_prob
        else:
            prob_dens = torch.exp(log_prob)
        return prob_dens

def minimum_image_vector(d, cell):
    """
    Compute the minimum image of the difference vector d for a given cell.

    Args:
        d (torch.Tensor): Difference vector of shape (..., 3).
        cell (torch.Tensor): Cell matrix of shape (3, 3) with lattice vectors as rows.

    Returns:
        torch.Tensor: The minimum image difference vector, of shape (..., 3).
    """
    inv_cell = torch.inverse(cell)  # shape (3,3)
    inv_cell = inv_cell.to(dtype=d.dtype)
    cell = cell.to(dtype=d.dtype)
    # Convert to fractional coordinates
    f = torch.matmul(d, inv_cell)
    # Wrap into the interval [-0.5, 0.5]
    f_wrapped = f - torch.round(f)
    # Convert back to Cartesian coordinates
    d_min = torch.matmul(f_wrapped, cell)
    return d_min
