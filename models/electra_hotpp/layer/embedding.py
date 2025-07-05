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
        #max_atomic_number = int(max(atomic_number))
        max_atomic_number = 119
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

        # If you'd like to use an embedding for atomic_number directly, you could:
        # self.z_weights = nn.Embedding(max_atomic_number + 1, n_channel)
        # But here we proceed with the custom orbital-based embedding:

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_channel = n_channel

        # ----------------------------------------------------------
        # 1) Extended shells up to 7p (covering elements up to Z=118).
        #    For Cm (Z=96), you only actually fill up to 5f^7 7s^2.
        #    This ordering follows the standard Madelung (n+ℓ) rule:
        #    1s → 2s → 2p → 3s → 3p → 4s → 3d → 4p → 5s → 4d → 5p → 6s
        #         → 4f → 5d → 6p → 7s → 5f → 6d → 7p
        # You may stop after 5f if you only care about up to Z=96, but
        # let's include 6d and 7p for completeness up to "Og" (118).
        self.shells = [
            ('1s', 2),
            ('2s', 2), ('2p', 6),
            ('3s', 2), ('3p', 6),
            ('4s', 2), ('3d', 10), ('4p', 6),
            ('5s', 2), ('4d', 10), ('5p', 6),
            ('6s', 2), ('4f', 14), ('5d', 10), ('6p', 6),
            ('7s', 2), ('5f', 14), ('6d', 10), ('7p', 6),
        ]

        # We only need the capacity:
        self.max_filling = torch.tensor([orb[1] for orb in self.shells],
                                        device=self.device)

        # 2) Extended principal shells up to n=7, each can hold 2n^2 electrons:
        #    n=1 → 2, n=2 → 8, n=3 → 18, n=4 → 32, n=5 → 50, n=6 → 72, n=7 → 98
        self.shells_orb = [
            ('1st', 2),
            ('2nd', 8),
            ('3rd', 18),
            ('4th', 32),
            ('5th', 50),
            ('6th', 72),
            ('7th', 98),
        ]
        self.max_filling_orb = torch.tensor([orb[1] for orb in self.shells_orb],
                                            device=self.device)

        # A small MLP to turn the 25+ (now bigger) input features into n_channel
        # We'll have more than 25 features now because shells are more than 7 entries.
        # Let's compute how many features exactly:
        # filling in each of self.shells -> len(self.shells)
        # free_e in each of self.shells -> len(self.shells)
        # filling in each of self.shells_orb -> len(self.shells_orb)
        # free_e in each of self.shells_orb -> len(self.shells_orb)
        # plus 3 (protons, neutrons, valence)
        # So total_input_dim = 2*len(self.shells) + 2*len(self.shells_orb) + 3
        total_input_dim = 2*len(self.shells) + 2*len(self.shells_orb) + 3

        self.embed_nn = nn.Sequential(
            nn.Linear(total_input_dim, self.n_channel),
            nn.SiLU(),
            nn.Linear(self.n_channel, self.n_channel),
        ).to(self.device)

    def forward(self,
                batch_data : Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch_data['atomic_number'] is a tensor of shape (B,) containing Z for each atom.
        We'll construct a per-atom feature vector capturing orbital fillings.
        """
        atomic_nums = batch_data['atomic_number']  # shape: (B,)
        B = len(atomic_nums)

        # Prepare a big tensor to store all features. Each row -> one atom.
        total_input_dim = (2*len(self.shells) + 2*len(self.shells_orb) + 3)
        full_tensor = torch.zeros((B, total_input_dim), device=self.device)

        for n, Z in enumerate(atomic_nums):
            atom_no = int(Z.item())  # Convert from 1-dim tensor to Python int
            atomic_symbol = chemical_symbols[atom_no]

            # ~~~~~~~~~~~~~~
            # 1) Shell-by-shell filling
            # ~~~~~~~~~~~~~~
            filling = torch.zeros(len(self.shells), dtype=int, device=self.device)
            remaining = atom_no
            for i, (shell_label, capacity) in enumerate(self.shells):
                if remaining >= capacity:
                    filling[i] = capacity
                    remaining -= capacity
                else:
                    filling[i] = remaining
                    remaining = 0
                    break

            # ~~~~~~~~~~~~~~
            # 2) Principal shells (1st, 2nd, 3rd, etc.)
            # ~~~~~~~~~~~~~~
            filling_orb = torch.zeros(len(self.shells_orb), dtype=int, device=self.device)
            remaining2 = atom_no
            for i, (shell_label, capacity) in enumerate(self.shells_orb):
                if remaining2 >= capacity:
                    filling_orb[i] = capacity
                    remaining2 -= capacity
                else:
                    filling_orb[i] = remaining2
                    remaining2 = 0
                    break

            # ~~~~~~~~~~~~~~
            # 3) Special-case electron rearrangements
            #    e.g., Cu, Cr in the standard approach:
            #    (Move 1 electron from s to d)
            #    You can add more special cases for Mo, Ag, etc.
            # ~~~~~~~~~~~~~~
            if atomic_symbol in ['Cu', 'Cr']:
                # The last two shells we defined in self.shells up to 3d...
                # Typically 4s is index 5, 3d is index 6 in an old short list,
                # but here it's different. Let's find them dynamically:
                # Find the index of 4s and 3d in self.shells
                # (If they exist in the final fill)
                shell_names = [s[0] for s in self.shells]
                idx_4s = shell_names.index('4s') if '4s' in shell_names else None
                idx_3d = shell_names.index('3d') if '3d' in shell_names else None

                if idx_4s is not None and idx_3d is not None:
                    if filling[idx_4s] > 0:
                        filling[idx_4s] -= 1
                        filling[idx_3d] += 1

            # ~~~~~~~~~~~~~~
            # 4) Compute fill fractions & free electrons
            # ~~~~~~~~~~~~~~
            fill_fraction = filling / self.max_filling
            free_electrons = (1 - fill_fraction) * self.max_filling * (filling != 0)

            fill_fraction_orb = filling_orb / self.max_filling_orb
            free_electrons_orb = (1 - fill_fraction_orb) * self.max_filling_orb * (filling_orb != 0)

            # ~~~~~~~~~~~~~~
            # 5) Combine orbital data into a single feature vector
            # ~~~~~~~~~~~~~~
            elec_structure = torch.cat([
                fill_fraction.float(),
                free_electrons.float(),
                fill_fraction_orb.float(),
                free_electrons_orb.float()
            ], dim=0)

            # ~~~~~~~~~~~~~~
            # 6) Basic nuclear & valence info
            # ~~~~~~~~~~~~~~
            n_prot = float(atom_no) / 119
            # Approx. neutrons = floor(mass_number - protons)
            mass_approx = Atom(atomic_symbol).mass  # e.g. from ASE, average isotopic mass
            n_neut = int(torch.floor(torch.tensor(mass_approx - n_prot))) / 119
            val_elec = valence_electrons(atomic_symbol)/ 18

            core_tensor = torch.tensor([n_prot, n_neut, val_elec],
                                       device=self.device,
                                       dtype=torch.float)

            # ~~~~~~~~~~~~~~
            # 7) Final per-atom input vector
            # ~~~~~~~~~~~~~~
            at_tensor = torch.cat([core_tensor, elec_structure], dim=0)
            full_tensor[n] = at_tensor

        # Pass the aggregated per-atom features through the MLP:
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

