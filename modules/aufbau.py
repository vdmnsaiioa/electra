from ase.data import atomic_numbers, chemical_symbols
import numpy as np
from torch import nn
import torch
from ase import Atom
from tools.atom_tools import valence_electrons


class aufbau_atom:
    def __init__(self, atom_no: int):
        shells = {'1s': 2,
                  '2s': 2, '2p': 6,
                  '3s': 2, '3p': 6,
                  '4s': 2, '3d': 10,
                  }
        shells_orb = {'1st': 2,
                      '2nd': 8,
                      '3rd': 18,
                      '4th': 32,
                      }
        self.max_filling = np.array([2, 2, 6, 2, 6, 2, 10])
        self.max_filling_orb = np.array([2, 8, 18, 32])
        # self.max_filling = np.array([2, 2, 6])
        # self.max_filling_orb = np.array([2, 8])
        atomic_symbol = chemical_symbols[atom_no]

        filling = np.zeros(len(shells), dtype=int)
        filling_orb = np.zeros(len(shells_orb), dtype=int)
        remaining_electrons = atom_no
        for i, shell in enumerate(shells):
            capacity = shells[shell]
            if remaining_electrons >= capacity:
                filling[i] = capacity
                remaining_electrons -= capacity
            else:
                filling[i] = remaining_electrons
                break
        remaining_electrons = atom_no
        for i, shell in enumerate(shells_orb):
            capacity = shells_orb[shell]
            if remaining_electrons >= capacity:
                filling_orb[i] = capacity
                remaining_electrons -= capacity
            else:
                filling_orb[i] = remaining_electrons
                break
        if atomic_symbol in ['Cu', 'Cr']:
            filling[6] += 1
            filling[5] -= 1

        self.filling = filling
        self.filling_orb = filling_orb
        self.fill_fraction = filling / self.max_filling
        self.fill_fraction_orb = filling_orb / self.max_filling_orb
        self.free_electrons = (1 - self.fill_fraction) * self.max_filling * (self.filling != 0)
        self.free_electrons_orb = (1 - self.fill_fraction_orb) * self.max_filling_orb * (self.filling_orb != 0)
        self.elec_structure = np.concatenate(
            [self.filling, self.free_electrons, self.filling_orb, self.free_electrons_orb])
        self.n_prot = atom_no
        self.n_neut = int(np.floor(Atom(chemical_symbols[atom_no]).mass - self.n_prot))
        self.valence_electrons = valence_electrons(atomic_symbol)

class AufEmbedding(nn.Module):
    def __init__(self,
                 embed_dim: int):
        super(AufEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.embed_nn = nn.Sequential(
            nn.Linear(25, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.combined_layer = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(self, n_prot, n_neut, val_elec, elec_structure, device: torch.device):
        core_tensor = torch.tensor([n_prot, n_neut, val_elec],
                                   device=device,
                                   dtype=torch.float)
        elec_tensor = torch.tensor(elec_structure,
                                   device=device,
                                   dtype=torch.float)
        full_tensor = torch.cat([core_tensor, elec_tensor])

        auf_atom_tensor = self.embed_nn(full_tensor)
        return auf_atom_tensor
    def embed_bonds(self, bond_tensor: torch.tensor):
        return self.combined_layer(bond_tensor)

    def reset_parameters(self):
        for layer in self.embed_nn:
            classname = layer.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.orthogonal_(layer.weight)


def embed_atoms(atom_numbers: torch.tensor,
                bond_between_tensor: torch.tensor,
                auf_embedding: AufEmbedding,
                device: torch.device,
                embedding_dim=10,
                ):
    atom_numbers = atom_numbers.cpu().numpy()
    embedded_atoms = torch.zeros(len(atom_numbers), embedding_dim, device=device)
    for i, atom_number in enumerate(atom_numbers):
        if atom_number != (-1):
            auf_atom = aufbau_atom(atom_number)
            embedded = auf_embedding(auf_atom.n_prot,
                                     auf_atom.n_neut,
                                     auf_atom.valence_electrons,
                                     auf_atom.elec_structure,
                                     device)
            embedded_atoms[i] = embedded
        else:
            b_atoms = atom_numbers[bond_between_tensor[i].cpu().numpy()]
            auf_atom_1 = aufbau_atom(b_atoms[0])
            embedded_1 = auf_embedding(auf_atom_1.n_prot,
                                     auf_atom_1.n_neut,
                                     auf_atom_1.valence_electrons,
                                     auf_atom_1.elec_structure,
                                     device)
            auf_atom_2 = aufbau_atom(b_atoms[1])
            embedded_2 = auf_embedding(auf_atom_2.n_prot,
                                       auf_atom_2.n_neut,
                                       auf_atom_2.valence_electrons,
                                       auf_atom_2.elec_structure,
                                       device)
            embedded_atoms[i] = auf_embedding.embed_bonds(torch.cat([embedded_1, embedded_2]))

    return embedded_atoms


if __name__ == '__main__':
    atom = Atom('Cu')
    auf_atom = aufbau_atom(atom)
    layer_dim = 5
    auf_embedding = AufEmbedding(layer_dim)
    embedded = auf_embedding(auf_atom.n_prot, auf_atom.n_neut, auf_atom.elec_structure)
