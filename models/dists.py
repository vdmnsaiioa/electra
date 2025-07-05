import torch
from torch.distributions import Categorical
from models.custom_dists import MixtureSameFamily, MultivariateNormal

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
                 relu: bool,
                 n_multiples: torch.tensor,
                 device: torch.device,
                 cell: torch.tensor = None,
                 mask: torch.tensor = None,
                 r_mat: torch.tensor = None,
                 pos_nonrot: torch.tensor = None,
                 pos_disp_nonrot: torch.tensor = None,
                 image_weights: torch.tensor = None):

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
        pos_tens_orig = pos_tens_orig.to(device)
        pos_tens = pos_tens_orig + pos_disp

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
                                cell=cell,
                                image_weights=image_weights)

    def get_dens_at_points(self,
                           points: torch.tensor):
        prob_dens = self.dist.log_prob(points)
        #prob_dens = torch.exp(prob_dens)
        return prob_dens

def get_full_grid_dens(atom_dist: gto,
                       gaus_pos: torch.tensor,
                       n_multiples: torch.tensor,
                       pos_grid: torch.tensor):
    grid_shape = pos_grid.shape[0:3]
    pos_grid = pos_grid.view(-1, 3)
    dens = atom_dist.get_dens_at_points(pos_grid)
    dens = dens.view(grid_shape)
    return dens

def get_n_points_dens(atom_dist: gto,
                      pos_grid: torch.tensor,
                      gaus_pos: torch.tensor,
                      n_multiples: torch.tensor,
                      n_points: int,
                      sample_all: bool = False,
                      r_mat: torch.tensor = None,
                      grid_nonrot: torch.tensor = None,
                      cell: torch.tensor = None):
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
                if num_cuda == 1:
                    cuda_num = 0
                    cur_dev = torch.device(f'cuda:{cuda_num}')
                else:
                    cuda_num = 1 + (i % (num_cuda - 1))
                    cur_dev = torch.device(f'cuda:{cuda_num}')
                    pos_grid_r_s = pos_grid_r_s.to(cur_dev)
                    gaus_pos = gaus_pos.to(cur_dev)
                if num_cuda == 1:
                    dens_split = atom_dist.get_dens_at_points(pos_grid_r_s)
                else:
                    dens_split = locals()[f"atom_dist_{cuda_num}"].get_dens_at_points(pos_grid_r_s).to(store_device)
                dens[points_ind_s] = dens_split
        else:
            for pos_grid_r_s, points_ind_s in zip(pos_grid_r_split, points_ind_split):
                dens_split = atom_dist.get_dens_at_points(pos_grid_r_s)
                dens[points_ind_s] = dens_split
    dens = dens.view(grid_shape)
    return dens, points_ind

def create_dist_copy_on_device(dist: gto,
                               new_device: torch.device):

    new_dist = gto(pos=dist.pos.to(new_device),
                   pos_disp=dist.pos_disp.to(new_device),
                   cov=dist.cov.to(new_device),
                   scales=dist.scales.to(new_device),
                   n_multiples=dist.n_multiples,
                   scalar_mults=dist.scalar_mults.to(new_device) if dist.scalar_mults is not None else None,
                   relu=dist.relu,
                   device=new_device)
    return new_dist

def get_n_points_dens_single(atom_dist: gto,
                      pos_grid: torch.tensor,
                      gaus_pos: torch.tensor,
                      n_multiples: torch.tensor,
                      n_points: int,
                      sample_all: bool = False,
                      r_mat: torch.tensor = None,
                      grid_nonrot: torch.tensor = None,
                             cell: torch.tensor = None):
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
            dens_split = atom_dist.get_dens_at_points(pos_grid_r_s)
            dens[points_ind_s] = dens_split

    dens = dens.view(grid_shape)
    return dens, points_ind
