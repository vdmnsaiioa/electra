from typing import Callable, List, Dict, Optional, Literal, Tuple
import torch
from torch import nn
from torch.cuda import device

from .base import AtomicModule
from ..layer import EmbeddingLayer, RadialLayer, ReadoutLayer
from ..layer.equivalent import NonLinearLayer, GraphConvLayer, SelfInteractionLayer
from ..utils import find_distances, _scatter_add, res_add, TensorAggregateOP
import copy
from ase.data import chemical_symbols
from tools.atom_tools import valence_electrons
from ase import Atoms
from ase.neighborlist import neighbor_list
from tools.graph_tools import plot_gaussian_arrows

class UpdateNodeBlock(nn.Module):
    def  __init__(self,
                 radial_fn      : RadialLayer,
                 max_r_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 norm_factor    : float=1.0,
                 activate_fn    : str='silu',
                 conv_mode      : Literal['node_j', 'node_edge']='node_j',
                 ) -> None:
        super().__init__()
        self.graph_conv = GraphConvLayer(radial_fn=radial_fn,
                                         input_dim=input_dim,
                                         output_dim=output_dim,
                                         max_in_way=max_in_way,
                                         max_out_way=max_out_way,
                                         max_r_way=max_r_way,
                                         conv_mode=conv_mode,
                                         )
        self.self_interact = SelfInteractionLayer(input_dim=input_dim,
                                                  max_way=max_out_way,
                                                  output_dim=output_dim)
        self.non_linear = NonLinearLayer(activate_fn=activate_fn,
                                         max_way=max_out_way,
                                         input_dim=output_dim)
        self.SymmetryBreak_Linear = nn.Linear(output_dim, output_dim)
        self.register_buffer("norm_factor", torch.tensor(norm_factor))

    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor],
                sym_break_set: Dict=None,
                ) -> Dict[int, torch.Tensor]:
        #mol_id = ''.join(map(str, batch_data['atomic_number'].tolist()))
        message = self.graph_conv(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        res_info = torch.jit.annotate(Dict[int, torch.Tensor], {})
        idx_i = batch_data["idx_i"]
        n_atoms = batch_data['atomic_number'].shape[0]
        for way in message.keys():
            res_info[way] = _scatter_add(message[way], idx_i, dim_size=n_atoms) / self.norm_factor
         #put in the symmetry breaking object here if it's a head block
        for sym_break_object in sym_break_set:
            res_info[1] += sym_break_object
            res_info = self.non_linear(self.self_interact(res_info))
            final_res_info += res_info
        #plot_gaussian_arrows(batch_data['coordinate'], result[1], f"AA_temp_pos_plots/res_add_{mol_id}.html")
        return res_add(node_info, res_info)


class UpdateEdgeBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_r_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 activate_fn    : str='silu',
                 conv_mode      : Literal['node_j', 'node_edge']='node_j',
                 ) -> None:
        super().__init__()
        self.graph_conv = GraphConvLayer(radial_fn=radial_fn,
                                         input_dim=input_dim,
                                         output_dim=output_dim,
                                         max_in_way=max_in_way,
                                         max_out_way=max_out_way,
                                         max_r_way=max_r_way,
                                         conv_mode=conv_mode)
        self.self_interact = SelfInteractionLayer(input_dim=input_dim,
                                                  max_way=max_out_way,
                                                  output_dim=output_dim)
        self.non_linear = NonLinearLayer(activate_fn=activate_fn,
                                         max_way=max_out_way,
                                         input_dim=output_dim)

    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        message = self.graph_conv(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        res_info = self.non_linear(self.self_interact(message))
        return res_add(edge_info, res_info)

class MiaoBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_r_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 norm_factor    : float=1.0,
                 activate_fn    : str='silu',
                 conv_mode      : Literal['node_j', 'node_edge']='node_j',
                 update_edge    : bool=False,
                 ) -> None:
        super().__init__()
        self.node_block = UpdateNodeBlock(radial_fn=radial_fn,
                                          max_r_way=max_r_way,
                                          max_in_way=max_in_way,
                                          max_out_way=max_out_way,
                                          input_dim=input_dim,
                                          output_dim=output_dim,
                                          norm_factor=norm_factor,
                                          activate_fn=activate_fn,
                                          conv_mode=conv_mode,
                                          )
        if update_edge:
            self.edge_block = UpdateEdgeBlock(radial_fn=radial_fn,
                                              max_r_way=max_r_way,
                                              max_in_way=max_in_way,
                                              max_out_way=max_out_way,
                                              input_dim=input_dim,
                                              output_dim=output_dim,
                                              activate_fn=activate_fn,
                                              conv_mode=conv_mode,
                                              )
        else:
            self.edge_block = None

    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor],
                sym_break_set: Dict=None,
                ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        node_info = self.node_block(node_info=node_info, edge_info=edge_info, batch_data=batch_data, sym_break_set=sym_break_set)
        if self.edge_block is not None:
            edge_info = self.edge_block(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        return node_info, edge_info

class MiaoNet(AtomicModule):
    """
    Miao nei ga
    duo xi da miao nei
    """
    def __init__(self,
                 embedding_layer : EmbeddingLayer,
                 radial_fn       : RadialLayer,
                 n_layers        : int,
                 max_r_way       : List[int],
                 max_out_way     : List[int],
                 max_out_heads   : List[int],
                 output_dim      : List[int],
                 activate_fn     : str="silu",
                 head_activate_list   : List[str]=['silu', 'jilu', 'tanh'],
                 mean            : float=0.,
                 std             : float=1.,
                 norm_factor     : float=1.,
                 bilinear        : bool=False,
                 conv_mode       : Literal['node_j', 'node_edge']='node_j',
                 update_edge     : bool=False,
                 ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).float())
        self.register_buffer("std", torch.tensor(std).float())
        self.embedding_layer = embedding_layer
        self.radial_fn = radial_fn
        self.max_out_heads = max_out_heads

        max_in_way = [0] + max_out_way[:-1]
        hidden_nodes = [embedding_layer.n_channel] + output_dim
        self.en_equivalent_blocks = self.get_eq_blocks(activate_fn, max_r_way, max_in_way, max_out_way,
            hidden_nodes, norm_factor, conv_mode, update_edge, n_layers)
        self.block_heads = self.get_block_heads(head_activate_list, max_r_way, max_in_way[1:], max_out_way, max_out_heads,
            hidden_nodes, norm_factor, conv_mode, update_edge, n_layers)
        self.mean_vec_mlps_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(4*hidden_nodes[0], 4*hidden_nodes[0]),
            nn.ReLU(),
            nn.Linear(4*hidden_nodes[0], hidden_nodes[0]),
            nn.Sigmoid(),
        ) for _ in range(3)])
        self.l0_nets_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_nodes[0], hidden_nodes[0]),
            nn.ReLU(),
            nn.Linear(hidden_nodes[0], hidden_nodes[0]),
        ) for _ in range(3)])
        self.mean_vec_mlps_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(4 * hidden_nodes[0], 4 * hidden_nodes[0]),
            nn.ReLU(),
            nn.Linear(4 * hidden_nodes[0], hidden_nodes[0]),
            nn.Sigmoid()
        ) for _ in range(3)])
        self.l0_nets_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_nodes[0], hidden_nodes[0]),
            nn.ReLU(),
            nn.Linear(hidden_nodes[0], hidden_nodes[0]),
        ) for _ in range(3)])

    def calculate(self,
                  batch_data : Dict[str, torch.Tensor],
                  symmetry_dict: Dict[str, torch.Tensor],
                  ) -> Dict[str, torch.Tensor]:
        node_info, edge_info, init_embed = self.get_init_info(batch_data)
        #plot_gaussian_arrows(batch_data['coordinate'], node_info[1],f"AA_temp_pos_plots/initial_gaussian_arrows_{mol_ID}.html")
        for i, en_equivalent in enumerate(self.en_equivalent_blocks):
            node_info, edge_info = en_equivalent(node_info, edge_info, batch_data)
        #plot_gaussian_arrows(batch_data['coordinate'], node_info[1],f"AA_temp_pos_plots/gaus_ar_block_{i}_{mol_ID}.html")
        ni_1, ei_1 = ({i: node_info[i] for i in node_info.keys()},
                        {i: edge_info[i] for i in edge_info.keys()})

        ni_2, ei_2 = ({i: node_info[i] for i in node_info.keys()},
                        {i: edge_info[i] for i in edge_info.keys()})

        ni_3, ei_3 = ({i: node_info[i] for i in node_info.keys()},
                        {i: edge_info[i] for i in edge_info.keys()})

        ## 1ST HEAD
        node_info_1, edge_info_1 = self.block_heads[0](ni_1, ei_1, batch_data, symmetry_dict)
        ## 2ND HEAD
        node_info_2, edge_info_2 = self.block_heads[1](ni_2, ei_2, batch_data, symmetry_dict)
        # 3RD HEAD
        node_info_3, edge_info_3 = self.block_heads[2](ni_3, ei_3, batch_data, symmetry_dict)
        return node_info_1, node_info_2, node_info_3, init_embed

    def get_init_info(self,
                      batch_data : Dict[str, torch.Tensor],
                      )->Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        emb = self.embedding_layer(batch_data=batch_data)

        sb_axes, l2_tensor = self.get_axes_of_inertia(batch_data)
        n_ax_oi = sb_axes.shape[0]

        n_atoms, emb_dim = emb.shape

        #n_tensors = l2_tensor.shape[0]

        emb_l1 = torch.zeros(n_atoms, emb_dim, 3, device=emb.device)
        #emb_l2 = torch.zeros(n_atoms, emb_dim, 3, 3, device=emb.device)
        sb_axes_extended = sb_axes.unsqueeze(0).expand(n_atoms, -1, -1)
        #l2_tensor = l2_tensor.unsqueeze(0).expand(n_atoms, -1, -1, -1)
        emb_l1[:, :n_ax_oi, :] = sb_axes_extended
        #emb_l2[:, :n_tensors, :, :] = l2_tensor

        node_info = {0: emb}
        #node_info = {0: emb}
        _, dij, _ = find_distances(batch_data)
        rbf = self.radial_fn(dij)
        edge_info = {0: rbf}

        return node_info, edge_info, emb

    def extract_parallel_components(self, vectors, ev_num):
        N = vectors.shape[0]

        # Step 1: Compute the covariance matrix for each set of 120 vectors
        # Outer product for each vector in each set, summed and normalized by 120
        cov_matrix = torch.mean(vectors.unsqueeze(-1) * vectors.unsqueeze(-2),dim=1).detach()  # Resulting shape: (N, 3, 3)

        # Step 2: Eigen-decomposition of the covariance matrix to find the principal axis for each set
        # Compute eigenvalues and eigenvectors for each 3x3 covariance matrix
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)  # eigenvalues, eigenvectors shapes: (N, 3), (N, 3, 3)
        eigenvectors = eigenvectors.detach()

        # Select the eigenvector with the largest eigenvalue (principal axis) for each set
        principal_axis = eigenvectors[:, :, ev_num]  # shape: (N, 3), last column has the largest eigenvalue

        # Step 3: Project vectors onto the principal axis and perpendicular components for each set
        # Parallel component (projection along the principal axis)
        parallel_components = torch.einsum('nij,nj->ni', vectors, principal_axis).unsqueeze(-1) * principal_axis.unsqueeze(1)
        return parallel_components

    def convert_vec_to_skew_mat(self, vectors):
        skew_mat = torch.zeros(vectors.shape[0], vectors.shape[1], 3, 3, device=vectors.device)
        skew_mat[:, :, 0, 1] = -vectors[:, :,  2]
        skew_mat[:, :, 0, 2] = vectors[:, :, 1]
        skew_mat[:, :, 1, 0] = vectors[:, :, 2]
        skew_mat[:, :, 1, 2] = -vectors[:, :, 0]
        skew_mat[:, :, 2, 0] = -vectors[:, :, 1]
        skew_mat[:, :, 2, 1] = vectors[:, :, 0]
        return skew_mat

    def get_eq_blocks(self, activate_fn, max_r_way, max_in_way, max_out_way,
            hidden_nodes, norm_factor, conv_mode, update_edge, n_layers):
        return nn.ModuleList([
            MiaoBlock(activate_fn=activate_fn,
                      radial_fn=self.radial_fn.replicate(),
                      # Use factory method, so the radial_fn in each layer are different
                      max_r_way=max_r_way[i],
                      max_in_way=max_in_way[i],
                      max_out_way=max_out_way[i],
                      input_dim=hidden_nodes[i],
                      output_dim=hidden_nodes[i + 1],
                      norm_factor=norm_factor,
                      conv_mode=conv_mode,
                      update_edge=update_edge,
                      ) for i in range(n_layers-3)])
    def get_block_heads(self, head_activate_list, max_r_way, max_in_way, max_out_way, max_out_heads,
            hidden_nodes, norm_factor, conv_mode, update_edge, n_layers):
        return nn.ModuleList([
            MiaoBlock(activate_fn=head_activate_list[i],
                      radial_fn=self.radial_fn.replicate(),
                      # Use factory method, so the radial_fn in each layer are different
                      max_r_way=max_r_way[i],
                      max_in_way=max_in_way[i],
                      max_out_way=max_out_heads[i],
                      input_dim=hidden_nodes[i],
                      output_dim=hidden_nodes[i + 1],
                      norm_factor=norm_factor,
                      conv_mode=conv_mode,
                      update_edge=update_edge,
                      symbreak=True
                      ) for i in range(3)])
    def get_axes_of_inertia(self, batch_data):
        positions = batch_data['coordinate']
        masses = batch_data['atomic_number']
        #masses = torch.abs(self.mass_embed(masses).squeeze(-1))
        COM = torch.sum(positions * masses[:, None], axis=0) / torch.sum(masses)
        centroid = torch.mean(positions, axis=0)

        # Center the positions
        centered_positions = positions - COM

        # Calculate the moment of inertia tensor
        I = torch.zeros((3, 3), device=positions.device)
        for i in range(len(masses)):
            I += (torch.eye(3, device=positions.device) * torch.linalg.norm(centered_positions[i]) ** 2 - torch.outer(
                centered_positions[i], centered_positions[i])) * masses[i]
        vector_to_dot = COM

        eigenvalues, axes_of_inertia = torch.linalg.eigh(I)
        ax_1 = axes_of_inertia[:, 0]  # First principal axis
        ax_2 = axes_of_inertia[:, 1]  # Second principal axis
        ax_3 = axes_of_inertia[:, 2]  # Third principal axis

        dot_products = torch.tensor([
            torch.dot(ax_1, vector_to_dot),
            torch.dot(ax_2, vector_to_dot),
            torch.dot(ax_3, vector_to_dot),
            torch.dot(-ax_3, vector_to_dot),
            torch.dot(-ax_2, vector_to_dot),
            torch.dot(-ax_1, vector_to_dot),
        ])

        sorted_indices = torch.argsort(dot_products, descending=True)
        #sorted_axes_aoi = torch.stack([ax_1, ax_2, ax_3]) * (torch.randint(0, 2, (3,), device=positions.device).float() * 2 - 1)
        sorted_axes_aoi = torch.stack([ax_1, ax_2, ax_3, -ax_3, -ax_2, -ax_1])
        sorted_axes_aoi = sorted_axes_aoi[sorted_indices]
        n_vecs = 3
        I = I / torch.linalg.norm(I)
        decomposed_tensor = self.split_tensor(I, sorted_axes_aoi[0], sorted_axes_aoi[1], sorted_axes_aoi[2])
        return sorted_axes_aoi[:n_vecs], decomposed_tensor

    def get_gravitational_tensor(self, batch_data):
        positions = batch_data['coordinate']
        val_elec = torch.tensor([valence_electrons(chemical_symbols[int(i)]) for i in batch_data['atomic_number']], device=positions.device)
        val_elec = torch.abs(self.grav_embed(val_elec).squeeze(-1))
        val_elec = torch.clamp(val_elec, min=1, max=50)
        n_atoms = positions.shape[0]
        gradients = torch.zeros_like(positions, device=positions.device)
        centered_positions = positions - torch.mean(positions, axis=0)
        for i in range(n_atoms):
            diff = centered_positions - centered_positions[i]
            distance = torch.norm(diff, dim=1)
            distance = torch.where(distance == 0, torch.tensor(1e-10, device=centered_positions.device), distance)
            gradient = torch.sum(val_elec[:, None] * (diff / distance[:, None] ** 3), dim=0)
            gradients[i] = gradient
        I = torch.zeros((3, 3), device=positions.device)
        for i in range(n_atoms):
            I += (torch.eye(3, device=positions.device) * torch.dot(gradients[i], gradients[i])  - torch.outer(gradients[i], gradients[i])) * val_elec[i]
        I = I / torch.linalg.norm(I)
        return I

    def split_tensor(self, tensor, ax_1, ax_2, ax_3):
        split_tensor = torch.zeros((5, 3, 3), device=tensor.device)
        scalar_part = (torch.trace(tensor) / 3) * torch.eye(3, device=tensor.device)
        deviatoric_tensor = tensor - scalar_part * torch.eye(3, device=tensor.device)
        # Use cross product to keep the matrix reflection invariant
        c12 = torch.cross(ax_1, ax_2)
        c23 = torch.cross(ax_2, ax_3)
        c31 = torch.cross(ax_3, ax_1)
        ax1_tens = torch.tensor([[0, c12[2], -c12[1]], [-c12[2], 0, c12[0]], [c12[1], -c12[0], 0]])
        ax2_tens = torch.tensor([[0, c23[2], -c23[1]], [-c23[2], 0, c23[0]], [c23[1], -c23[0], 0]])
        ax3_tens = torch.tensor([[0, c31[2], -c31[1]], [-c31[2], 0, c31[0]], [c31[1], -c31[0], 0]])
        split_tensor[0] = scalar_part
        split_tensor[1] = deviatoric_tensor
        split_tensor[2] = ax1_tens
        split_tensor[3] = ax2_tens
        split_tensor[4] = ax3_tens
        return split_tensor

