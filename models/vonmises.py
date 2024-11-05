import torch
from torch.distributions import Distribution
from torch.distributions.utils import lazy_property
from torch.distributions import constraints
import math


class VonMisesFisher(Distribution):
    arg_constraints = {'mu': constraints.real, 'kappa': constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self, mu, kappa, kappa2, loc, validate_args=None):
        # Ensure the mean direction (mu) is a unit vector
        denom = mu.norm(dim=-1, keepdim=True)
        denom = torch.where(denom == 0, torch.tensor(1.0, device=denom.device), denom)
        mu = mu / denom
        self.loc = loc.float()
        self.mu = mu
        self.kappa = kappa2
        self.dim = mu.size(-1)

        # Compute normalization constant C_d(kappa)
        self.log_c = self._compute_log_normalizer(self.dim, kappa2)

        super(VonMisesFisher, self).__init__(batch_shape=self.mu.shape[:-1], validate_args=validate_args)

    @staticmethod
    def _compute_log_normalizer(dim, kappa):
        """Compute the log of the normalization constant C_d(kappa)."""
        if dim == 3:
            # Special case for 3D vMF
            return (kappa - math.log(2 * math.pi) - torch.log(1 - torch.exp(-2 * kappa)))
        else:
            # For higher dimensions, approximate using large-kappa approximation
            return ((dim / 2 - 1) * torch.log(kappa) - (dim / 2) * math.log(2 * math.pi) -
                    torch.lgamma(torch.tensor(dim / 2.0, dtype=torch.float)))

    def rsample(self, sample_shape=torch.Size()):
        """Generate samples from the vMF distribution."""
        shape = self._extended_shape(sample_shape)

        # Sample the magnitude on the unit sphere
        w = self._sample_w(shape, self.kappa, self.dim)

        # Generate a point uniformly on the sphere
        v = torch.randn(*shape, self.dim - 1, dtype=self.mu.dtype, device=self.mu.device)
        v = v / v.norm(dim=-1, keepdim=True)

        # Embed the sample into the full space
        z = torch.cat([w.unsqueeze(-1), (1 - w ** 2).sqrt() * v], dim=-1)

        # Rotate the sample to the mean direction mu
        return self._householder_transform(z, self.mu)

    def _sample_w(self, shape, kappa, dim):
        """Sample w ~ p(w) using rejection sampling."""
        b = dim / (2 * kappa + (4 * kappa ** 2 + dim ** 2) ** 0.5)
        x = (1 - b) / (1 + b)
        c = kappa * x + dim - 1

        w = torch.zeros(shape, dtype=self.mu.dtype, device=self.mu.device)
        for _ in range(100):  # Up to 100 tries to sample
            u = torch.rand(shape, dtype=self.mu.dtype, device=self.mu.device)
            z = (1 - b) / (1 + b) + (2 * b) * u
            w_prop = (1 - z) / (1 + z)
            u_prime = torch.rand(shape, dtype=self.mu.dtype, device=self.mu.device)
            if u_prime < c * (1 - w_prop ** 2).sqrt() * torch.exp(kappa * w_prop):
                w = w_prop
                break
        return w

    def _householder_transform(self, z, mu):
        """Apply a Householder transformation to rotate z onto the direction mu."""
        v = torch.tensor([1.0, 0.0, 0.0], device=mu.device) - mu
        v = v / v.norm(dim=-1, keepdim=True)
        return z - 2 * torch.einsum('...i,...j->...ij', v, v) @ z.unsqueeze(-1).squeeze(-1)

    def log_prob(self, value, links):
        """Compute the log probability of a given sample."""
        if links is None:
            diff = value - self.mu
            diff = diff / diff.norm(dim=-1, keepdim=True)
            log_prob = self.kappa * (self.mu * diff).sum(dim=-1) + self.log_c
        else:
            sel_values = self.mu[links[:, 1]]
            diff = value.squeeze(1)[links[:, 0]] - self.loc[links[:, 1]]
            log_prob = self.kappa[links[:, 1]] * torch.sum(sel_values * diff, dim=-1, keepdim=True) + self.log_c[links[:, 1]]
            log_prob = log_prob.squeeze(1)
        return log_prob



    @lazy_property
    def mean(self):
        """Return the mean direction of the distribution."""
        return self.mu

    @lazy_property
    def variance(self):
        """Return the variance of the distribution."""
        return (1 / self.kappa) * torch.eye(self.dim, device=self.mu.device)


