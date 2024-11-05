from typing import List
import ase
import torch
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

class CollateFuncAtoms:
    def __init__(self, cutoff, pin_memory=False, set_pbc_to=None):
        self.cutoff = cutoff
        if torch.cuda.is_available():
            self.pin_memory = True
        else:
            self.pin_memory = False
        self.set_pbc = set_pbc_to

    def __call__(self, atom_list: List):
        graphs = []
        for atoms in atom_list:
            if self.set_pbc is not None:
                atoms.set_pbc(self.set_pbc)

            graphs.append(atoms_to_graph_dict(
                atoms,
                self.cutoff,
            ))

        return collate_list_of_dicts(graphs, pin_memory=self.pin_memory)

def collate_list_of_dicts(list_of_dicts, pin_memory=False):
    # Convert from "list of dicts" to "dict of lists"
    dict_of_lists = {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}

    # Convert each list of tensors to single tensor with pad and stack
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x

    collated = {k: pin(pad_and_stack(dict_of_lists[k])) for k in dict_of_lists}
    return collated

def atoms_to_graph_dict(atoms, cutoff):
    atom_edges, atom_edges_displacement, _, _ = atoms_to_graph(atoms, cutoff)

    default_type = torch.get_default_dtype()

    # pylint: disable=E1102
    res = {
        "nodes": torch.tensor(atoms.get_atomic_numbers()),
        "atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0)),
        "atom_edges_displacement": torch.tensor(
            np.concatenate(atom_edges_displacement, axis=0), dtype=default_type
        ),
    }
    res["num_nodes"] = torch.tensor(res["nodes"].shape[0])
    res["num_atom_edges"] = torch.tensor(res["atom_edges"].shape[0])
    res["atom_xyz"] = torch.tensor(atoms.get_positions(), dtype=default_type)
    res["cell"] = torch.tensor(np.array(atoms.get_cell()), dtype=default_type)

    return res

def atoms_to_graph(atoms, cutoff):
    atom_edges = []
    atom_edges_displacement = []

    inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

    neighborlist = AseNeigborListWrapper(cutoff, atoms)

    atom_positions = atoms.get_positions()

    for i in range(len(atoms)):
        neigh_idx, neigh_vec, _ = neighborlist.get_neighbors(i, cutoff)

        self_index = np.ones_like(neigh_idx) * i
        edges = np.stack((neigh_idx, self_index), axis=1)

        neigh_pos = atom_positions[neigh_idx]
        this_pos = atom_positions[i]
        neigh_origin = neigh_vec + this_pos - neigh_pos
        neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

        atom_edges.append(edges)
        atom_edges_displacement.append(neigh_origin_scaled)

    return atom_edges, atom_edges_displacement, neighborlist, inv_cell_T

def pad_and_stack(tensors: List[torch.Tensor]):
    """Pad list of tensors if tensors are arrays and stack if they are scalars"""
    if tensors[0].shape:
        return torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=0
        )
    return torch.stack(tensors)

class AseNeigborListWrapper:
    """
    Wrapper around ASE neighborlist to have the same interface as asap3 neighborlist

    """

    def __init__(self, cutoff, atoms):
        self.neighborlist = ase.neighborlist.NewPrimitiveNeighborList(
            cutoff, skin=0.0, self_interaction=False, bothways=True
        )
        self.neighborlist.build(
            atoms.get_pbc(), atoms.get_cell(), atoms.get_positions()
        )
        self.cutoff = cutoff
        self.atoms_positions = atoms.get_positions()
        self.atoms_cell = atoms.get_cell()

    def get_neighbors(self, i, cutoff):
        assert (
                cutoff == self.cutoff
        ), "Cutoff must be the same as used to initialise the neighborlist"

        indices, offsets = self.neighborlist.get_neighbors(i)

        rel_positions = (
                self.atoms_positions[indices]
                + offsets @ self.atoms_cell
                - self.atoms_positions[i][None]
        )

        dist2 = np.sum(np.square(rel_positions), axis=1)

        return indices, rel_positions, dist2


import numpy as np
import plotly.graph_objs as go
import plotly.io as pio


def plot_gaussian_arrows(atom_positions, gaus_positions, filename):
    # Convert tensors to numpy if needed
    if isinstance(atom_positions, torch.Tensor):
        atom_positions = atom_positions.detach().cpu().numpy()
    if isinstance(gaus_positions, torch.Tensor):
        gaus_positions = gaus_positions.detach().cpu().numpy()

    # Create scatter plot for atom positions
    trace_atoms = go.Scatter3d(
        x=atom_positions[:, 0],
        y=atom_positions[:, 1],
        z=atom_positions[:, 2],
        mode='markers',
        marker=dict(size=5, color='black'),
        name="Atoms"
    )

    # Create arrows from atom positions to Gaussian positions
    arrows = []
    for i, atom_pos in enumerate(atom_positions):
        for gaus_pos in gaus_positions[i]:
            arrows.append(
                go.Scatter3d(
                    x=[atom_pos[0], atom_pos[0]+gaus_pos[0]],
                    y=[atom_pos[1], atom_pos[1]+gaus_pos[1]],
                    z=[atom_pos[2], atom_pos[2]+gaus_pos[2]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    showlegend=False
                )
            )

    # Define layout
    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    # Create figure and save to file
    fig = go.Figure(data=[trace_atoms] + arrows, layout=layout)
    pio.write_html(fig, file=filename, auto_open=False)

