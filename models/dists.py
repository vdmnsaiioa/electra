import torch
#from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Categorical
#from torch.distributions import MixtureSameFamily
from models.custom_dists import MixtureSameFamily, MultivariateNormal

class gto():
    def __init__(self,
                 pos: torch.tensor,
                 pos_disp: torch.tensor,
                 cov: torch.tensor,
                 kappa: torch.tensor,
                 kappa2: torch.tensor,
                 VMF_vec: torch.tensor,
                 use_vmf: bool,
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
        self.cov = cov
        self.scales = scales
        self.n_multiples = n_multiples
        self.relu = relu
        self.pos = pos
        self.pos_disp = pos_disp
        self.kappa = kappa
        self.kappa2 = kappa2
        self.VMF_vec = VMF_vec
        self.use_vmf = use_vmf

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
                                kappa=kappa,
                                kappa2=kappa2,
                                VMF_vec=VMF_vec+pos_disp,
                                VMF_loc=pos_tens_orig,
                                use_vmf=use_vmf,
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

def get_full_grid_dens(atom_dist: gto,
                       gaus_pos: torch.tensor,
                       n_multiples: torch.tensor,
                       pos_grid: torch.tensor):
    grid_shape = pos_grid.shape[0:3]
    pos_grid = pos_grid.view(-1, 3)
    split_masks = get_masks(pos_grid=pos_grid,
                            gaus_pos=gaus_pos,
                            cut_distance=atom_dist.cut_distance,
                            n_points=pos_grid.size(0),
                            )
    #idx_tens = torch.cat([torch.tensor(val * [i]) for i, val in enumerate(n_multiples)], dim=0)
    #atom_distances_mask = torch.index_select(split_masks[0], dim=1, index=idx_tens)
    links = torch.nonzero(split_masks[0], as_tuple=False)
    dens = atom_dist.get_dens_at_points(pos_grid, links=links)
    dens = dens.view(grid_shape)
    return dens


def get_n_points_dens(atom_dist: gto,
                      pos_grid: torch.tensor,
                      gaus_pos: torch.tensor,
                      n_multiples: torch.tensor,
                      n_points: int,
                      sample_all: bool = False,
                      r_mat: torch.tensor = None,
                      grid_nonrot: torch.tensor = None):
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
        pos_grid_r_split = torch.split(pos_grid, n_points)
        points_ind = torch.arange(0, pos_grid.size(0), device=store_device)
        points_ind_split = torch.split(points_ind, n_points)
        # Adjust the second column (i.e., the atom indices) based on the desired multiples
        if torch.cuda.is_available():
            for i in range(1, num_cuda):
                locals()[f"atom_dist_{i}"] = create_dist_copy_on_device(atom_dist, torch.device(f'cuda:{i}'))
            for i, (pos_grid_r_s, points_ind_s) in enumerate(zip(pos_grid_r_split, points_ind_split)):
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
                split_mask = get_mask(pos_grid=pos_grid_r_s,
                                gaus_pos=gaus_pos,
                                cut_distance=cut_distance,
                                )
                split_mask = split_mask.to(cur_dev)
                if num_cuda == 1:
                    #idx_tens = torch.cat([torch.tensor(val * [i], device=cur_dev) for i, val in enumerate(n_multiples)], dim=0)
                    #atom_distances_mask = torch.index_select(split_mask, dim=1, index=idx_tens)
                    links = torch.nonzero(split_mask, as_tuple=False)
                    dens_split = atom_dist.get_dens_at_points(pos_grid_r_s, links=links)
                else:
                    #idx_tens = torch.cat([torch.tensor(val * [i], device=cur_dev) for i, val in enumerate(n_multiples)], dim=0)
                    #atom_distances_mask = torch.index_select(split_mask, dim=1, index=idx_tens)
                    links = torch.nonzero(split_mask, as_tuple=False)
                    links = links.to(cur_dev)
                    dens_split = locals()[f"atom_dist_{cuda_num}"].get_dens_at_points(pos_grid_r_s, links=links).to(store_device)
                dens[points_ind_s] = dens_split
        else:
            split_masks = get_masks(pos_grid=pos_grid,
                                    gaus_pos=gaus_pos,
                                    cut_distance=cut_distance,
                                    n_points=n_points,
                                    )
            for pos_grid_r_s, points_ind_s, split_mask in zip(pos_grid_r_split, points_ind_split, split_masks):
                #idx_tens = torch.cat([torch.tensor(val*[i]) for i, val in enumerate(n_multiples)], dim=0)
                #atom_distances_mask = torch.index_select(split_mask, dim=1, index=idx_tens)
                links = torch.nonzero(split_mask, as_tuple=False)
                dens_split = atom_dist.get_dens_at_points(pos_grid_r_s, links=links)
                dens[points_ind_s] = dens_split
    else:
        new_sampling = True
        if new_sampling:
            # Assuming pos_grid is a tensor of size (N, 3)
            # pos_grid = torch.tensor([[x, y, z] for x, y, z in ...], device=store_device)
            # Calculate the center (centroid) of the grid
            center = pos_grid.mean(dim=0)

            # Compute Euclidean distances from the center
            distances_from_center = torch.norm(pos_grid - center, dim=1)
            # Calculate Gaussian probabilities based on distance from the center
            # Assume a standard deviation (sigma) for the Gaussian distribution
            sigma = torch.std(distances_from_center)
            probabilities = torch.exp(-0.5 * (distances_from_center / sigma) ** 2)

            # Normalize probabilities
            probabilities /= torch.sum(probabilities)

            # Sample points using the calculated Gaussian probabilities
            distance_sampled_indices = torch.multinomial(probabilities, num_samples=int(0.6*n_points),
                                                         replacement=False)
            fully_random_indices = torch.randint(0, distances_from_center.size(0), (n_points - int(0.4*n_points),),
                                                 device=store_device)

            # Combine and ensure unique indices
            points_ind = torch.unique(torch.cat((distance_sampled_indices, fully_random_indices)))
        else:
            points_ind = torch.randint(0, pos_grid.size(0), (n_points,), device=store_device)
        points_to_sample = pos_grid[points_ind, :]
        if torch.cuda.is_available():
            n_points_per_device = n_points // num_cuda
            split_masks = get_masks(pos_grid=points_to_sample,
                                   gaus_pos=gaus_pos,
                                   cut_distance=cut_distance,
                                   n_points=n_points_per_device,
                                   )
            for i in range(1, num_cuda):
                locals()[f"atom_dist_{i}"] = create_dist_copy_on_device(atom_dist, torch.device(f'cuda:{i}'))
            points_pos_split = torch.split(points_to_sample, n_points_per_device)
            points_ind_split = torch.split(points_ind, n_points_per_device)
            for i, (points_pos, points_slice, split_mask) in enumerate(zip(points_pos_split, points_ind_split, split_masks)):
                if num_cuda == 1:
                    cuda_num = 0
                    cur_dev = torch.device(f'cuda:{cuda_num}')
                else:
                    cuda_num = 1 + (i % (num_cuda - 1))
                    cur_dev = torch.device(f'cuda:{cuda_num}')
                    pos_grid_r_s = points_pos.to(cur_dev)
                split_mask = split_mask.to(cur_dev)
                if num_cuda == 1:
                    #idx_tens = torch.cat([torch.tensor(val * [i], device=cur_dev) for i, val in enumerate(n_multiples)], dim=0)
                    #atom_distances_mask = torch.index_select(split_mask, dim=1, index=idx_tens)
                    links = torch.nonzero(split_mask, as_tuple=False)
                    dens_split = atom_dist.get_dens_at_points(pos_grid_r_s, links=links)
                else:
                    #idx_tens = torch.cat([torch.tensor(val * [i], device=cur_dev) for i, val in enumerate(n_multiples)], dim=0)
                    #atom_distances_mask = torch.index_select(split_mask, dim=1, index=idx_tens)
                    links = torch.nonzero(split_mask, as_tuple=False)
                    links = links.to(cur_dev)
                    dens_split = locals()[f"atom_dist_{cuda_num}"].get_dens_at_points(pos_grid_r_s, links=links).to(
                        store_device)
                dens[points_slice] = dens_split
        else:
            split_mask = get_masks(pos_grid=points_to_sample,
                      gaus_pos=gaus_pos,
                      cut_distance=cut_distance,
                      n_points=points_to_sample.shape[0],
                      )[0]
            #idx_tens = torch.cat([torch.tensor(val * [i]) for i, val in enumerate(n_multiples)], dim=0)
            #atom_distances_mask = torch.index_select(split_mask, dim=1, index=idx_tens)
            links = torch.nonzero(split_mask, as_tuple=False)
            dens_points = atom_dist.get_dens_at_points(points_to_sample, links=links)
            dens[points_ind] = dens_points
    dens = dens.view(grid_shape)
    return dens, points_ind

def create_dist_copy_on_device(dist: gto,
                               new_device: torch.device):
    new_dist = gto(pos=dist.pos.to(new_device),
                                        pos_disp=dist.pos_disp.to(new_device),
                                        cov=dist.cov.to(new_device),
                                        kappa=dist.kappa.to(new_device),
                                        kappa2=dist.kappa2.to(new_device),
                                        VMF_vec=dist.VMF_vec.to(new_device),
                                        use_vmf=dist.use_vmf,
                                        scales=dist.scales.to(new_device),
                                        n_multiples=dist.n_multiples,
                                        scalar_mults=dist.scalar_mults.to(new_device),
                                        cut_distance=dist.cut_distance,
                                        use_links=dist.use_links,
                                        relu=dist.relu,
                                        device=new_device)
    return new_dist

def get_masks(pos_grid: torch.tensor,
              gaus_pos: torch.tensor,
              n_points: int,
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
    gaus_distances = torch.norm(pos_grid.unsqueeze(1) - gaus_pos.unsqueeze(0), dim=2)
    gaus_distances_mask = torch.logical_not(gaus_distances < gaus_distances.max(dim=-1).values.unsqueeze(-1)) + (gaus_distances < cut_distance)
    gaus_distances_mask_split = torch.split(gaus_distances_mask, n_points)
    return gaus_distances_mask_split

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
    gaus_distances = torch.norm(pos_grid.unsqueeze(1) - gaus_pos.unsqueeze(0), dim=2)
    gaus_distances_mask = torch.logical_not(gaus_distances < gaus_distances.max(dim=-1).values.unsqueeze(-1)) + (gaus_distances < cut_distance)
    return gaus_distances_mask
