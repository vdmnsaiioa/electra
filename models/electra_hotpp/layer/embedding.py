import torch
from torch import nn
from typing import List, Optional, Dict
from .base import CutoffLayer, EmbeddingLayer
from ..utils import find_distances, _scatter_add
from ase.data import atomic_numbers, chemical_symbols
from tools.atom_tools import valence_electrons
from ase import Atom

__all__ = ["AtomicOneHot",
           "AtomicNumber",
           "AtomicEmbedding",
           "AufbauEmbedding",
           "BehlerG1",
           ]


class AtomicOneHot(EmbeddingLayer):
    def __init__(self, 
                 atomic_number : List[int], 
                 trainable     : bool=False
                 ) -> None:
        super().__init__()
        max_atomic_number = max(atomic_number)
        n_atomic_number = len(atomic_number)
        weights = torch.zeros(max_atomic_number + 1, n_atomic_number)
        for idx, z in enumerate(atomic_number):
            weights[z, idx] = 1.
        self.z_weights = nn.Embedding(max_atomic_number + 1, n_atomic_number)
        self.z_weights.weight.data = weights
        if not trainable:
            self.z_weights.weight.requires_grad = False
        self.n_channel = n_atomic_number

    def forward(self, 
                batch_data : Dict[str, torch.Tensor],
                ) -> torch.Tensor:
        return self.z_weights(batch_data['atomic_number'])


class AtomicNumber(EmbeddingLayer):
    def __init__(self, 
                 atomic_number : List[int], 
                 trainable     : bool=False
                 ) -> None:
        super().__init__()
        max_atomic_number = max(atomic_number)
        weights = torch.arange(max_atomic_number + 1)[:, None].float()
        self.z_weights = nn.Embedding(max_atomic_number + 1, 1)
        self.z_weights.weight.data = weights
        if not trainable:
            self.z_weights.weight.requires_grad = False
        self.n_channel = 1

    def forward(self, 
                batch_data : Dict[str, torch.Tensor],
                ) -> torch.Tensor:
        return self.z_weights(batch_data['atomic_number'])


class AtomicEmbedding(EmbeddingLayer):
    def __init__(self, 
                 atomic_number : List[int], 
                 n_channel     : int,
                 ) -> None:
        super().__init__()
        max_atomic_number = int(max(atomic_number))
        self.z_weights = nn.Embedding(max_atomic_number + 1, n_channel)
        self.n_channel = n_channel

    def forward(self, 
                batch_data : Dict[str, torch.Tensor],
                ) -> torch.Tensor:
        return self.z_weights(batch_data['atomic_number'])

class AufbauEmbedding(EmbeddingLayer):
    def __init__(self,
                 atomic_number : List[int],
                 n_channel     : int,
                 device: torch.device,
                 ) -> None:
        super().__init__()
        max_atomic_number = int(max(atomic_number))
        #self.z_weights = nn.Embedding(max_atomic_number + 1, n_channel)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_channel = n_channel
        self.embed_nn = nn.Sequential(
            nn.Linear(25, self.n_channel),
            nn.SiLU(),
            nn.Linear(self.n_channel, self.n_channel),
        )
        self.embed_nn = self.embed_nn.to(self.device)
        self.shells = {'1s': 2,
                  '2s': 2, '2p': 6,
                  '3s': 2, '3p': 6,
                  '4s': 2, '3d': 10,
                  }
        self.shells_orb = {'1st': 2,
                      '2nd': 8,
                      '3rd': 18,
                      '4th': 32,
                      }
        self.max_filling = torch.tensor([2, 2, 6, 2, 6, 2, 10], device=self.device)
        self.max_filling_orb = torch.tensor([2, 8, 18, 32], device=self.device)

    def forward(self,
                batch_data : Dict[str, torch.Tensor]) -> torch.Tensor:
        atomic_nums = batch_data['atomic_number']
        full_tensor = torch.zeros(len(atomic_nums), 25, device=self.device)
        for n, atom_no in enumerate(atomic_nums):
            atomic_symbol = chemical_symbols[atom_no]

            filling = torch.zeros(len(self.shells), dtype=int, device=self.device)
            filling_orb = torch.zeros(len(self.shells_orb), dtype=int, device=self.device)
            remaining_electrons = atom_no.clone()
            for i, shell in enumerate(self.shells):
                capacity = self.shells[shell]
                if remaining_electrons >= capacity:
                    filling[i] = capacity
                    remaining_electrons -= capacity
                else:
                    filling[i] = remaining_electrons
                    break
            remaining_electrons = atom_no.clone()
            for i, shell in enumerate(self.shells_orb):
                capacity = self.shells_orb[shell]
                if remaining_electrons >= capacity:
                    filling_orb[i] = capacity
                    remaining_electrons -= capacity
                else:
                    filling_orb[i] = remaining_electrons
                    break
            if atomic_symbol in ['Cu', 'Cr']:
                filling[6] += 1
                filling[5] -= 1
            filling = filling
            filling_orb = filling_orb
            fill_fraction = filling / self.max_filling
            fill_fraction_orb = filling_orb / self.max_filling_orb
            free_electrons = (1 - fill_fraction) * self.max_filling * (filling != 0)
            free_electrons_orb = (1 - fill_fraction_orb) * self.max_filling_orb * (filling_orb != 0)
            elec_structure = torch.cat(
                [filling, free_electrons, filling_orb, free_electrons_orb])
            n_prot = atom_no.clone()
            n_neut = int(torch.floor(Atom(chemical_symbols[atom_no]).mass - n_prot))
            val_elec = valence_electrons(atomic_symbol)

            core_tensor = torch.tensor([n_prot, n_neut, val_elec],
                                       device=self.device,
                                       dtype=torch.float)
            elec_tensor = torch.tensor(elec_structure,
                                       device=self.device,
                                       dtype=torch.float)
            at_tensor = torch.cat([core_tensor, elec_tensor])

            full_tensor[n] = at_tensor
        full_tensor = full_tensor.to(self.device)
        auf_atom_tensor = self.embed_nn(full_tensor)

        return auf_atom_tensor

    def reset_parameters(self):
        for layer in self.embed_nn:
            classname = layer.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.orthogonal_(layer.weight)

class BehlerG1(EmbeddingLayer):
    """
    wACSF—Weighted atom-centered symmetry functions as descriptors in machine learning potentials
    J. Chem. Phys. 148, 241709 (2018)
    https://doi.org/10.1063/1.5019667
    """
    def __init__(self, 
                 n_radial      : int, 
                 cut_fn        : CutoffLayer, 
                 etas          : Optional[List[float]]=None, 
                 rss           : Optional[List[float]]=None,
                 trainable     : bool=False,
                 ) -> None:
        super().__init__()
        self.cut_fn = cut_fn
        if rss is None or etas is None:
            cutoff = cut_fn.cutoff.numpy()
            rss = torch.linspace(0.3, cutoff - 0.3, n_radial)
            etas = 0.5 * torch.ones_like(rss) / (rss[1] - rss[0]) ** 2
        assert (len(etas) == n_radial) and (len(rss) == n_radial), "Lengths of 'etas' or 'rss' error"

        if trainable:
            self.etas = nn.Parameter(etas)
            self.rss = nn.Parameter(rss)
            # self.etas = PositiveParameter(etas)
        else:
            self.register_buffer("etas", etas)
            self.register_buffer("rss", rss)
        self.n_channel = n_radial

    def forward(self,
                batch_data  : Dict[str, torch.Tensor],
                ) -> torch.Tensor:
        n_atoms = batch_data['atomic_number'].shape[0]
        idx_i = batch_data['idx_i']
        idx_j = batch_data['idx_j']
        _, dij, _ = find_distances(batch_data)
        zij = batch_data['atomic_number'][idx_j].unsqueeze(-1)                      # [n_edge, 1]
        dij = dij.unsqueeze(-1)
        f = torch.exp(-self.etas * (dij - self.rss) ** 2) * self.cut_fn(dij) * zij  # [n_edge, n_channel]
        f = _scatter_add(f, idx_i, dim_size=n_atoms)
        # f = segment_coo(f, idx_i, dim_size=n_atoms, reduce="sum").view(n_atoms, -1)
        return f
