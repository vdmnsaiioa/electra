from typing import Callable, List, Dict, Optional, Literal, Tuple
import torch
from torch import nn
from torch.cuda import device

from .base import AtomicModule
from ..layer import EmbeddingLayer, RadialLayer, ReadoutLayer
from ..layer.equivalent import NonLinearLayer, GraphConvLayer, SelfInteractionLayer, FixedLinearTransformVector, FixedLinearTransformTensor
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
        self.register_buffer("norm_factor", torch.tensor(norm_factor))

    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor],
                sym_break_set: Dict,
                ) -> Dict[int, torch.Tensor]:
        #mol_id = ''.join(map(str, batch_data['atomic_number'].tolist()))
        message = self.graph_conv(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        res_info = torch.jit.annotate(Dict[int, torch.Tensor], {})
        idx_i = batch_data["idx_i"]
        n_atoms = batch_data['atomic_number'].shape[0]
        for way in message.keys():
            res_info[way] = _scatter_add(message[way], idx_i, dim_size=n_atoms) / self.norm_factor
        res_info = self.non_linear(self.self_interact(res_info))
        if self.norm is not None:
            res_info = self.norm(res_info, batch_data['batch'], batch_data['n_atoms'])
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
                ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        node_info = self.node_block(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
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
                 prune         : bool=False,
                 ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).float())
        self.register_buffer("std", torch.tensor(std).float())
        self.embedding_layer = embedding_layer
        self.radial_fn = radial_fn
        self.max_out_heads = max_out_heads
        self.prune = prune

        max_in_way = [1] + max_out_way[:-1]
        hidden_nodes = [embedding_layer.n_channel] + output_dim
        self.en_equivalent_blocks = self.get_eq_blocks(activate_fn, max_r_way, max_in_way, max_out_way,
            hidden_nodes, norm_factor, conv_mode, update_edge, n_layers)
        self.block_heads = self.get_block_heads(head_activate_list, max_r_way, max_in_way[1:], max_out_way, max_out_heads,
            hidden_nodes, norm_factor, conv_mode, update_edge, n_layers)
        self.mean_vec_mlps_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(7 * hidden_nodes[0], 4 * hidden_nodes[0]),
            nn.Mish(),
            nn.Linear(4 * hidden_nodes[0], hidden_nodes[0]),
            nn.Sigmoid()
        ) for _ in range(n_layers - 3)])
        self.mean_vec_mlps_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(7 * hidden_nodes[0], 4 * hidden_nodes[0]),
            nn.Mish(),
            nn.Linear(4 * hidden_nodes[0], hidden_nodes[0]),
            nn.Sigmoid()
        ) for _ in range(3)])
        self.split_tensor_weights_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(7 * hidden_nodes[0], 4 * hidden_nodes[0]),
            nn.Mish(),
            nn.Linear(4 * hidden_nodes[0], 3 * hidden_nodes[0]),
            nn.Sigmoid()
        ) for _ in range(n_layers - 3)])
        self.split_tensor_weights_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(7 * hidden_nodes[0], 4 * hidden_nodes[0]),
            nn.Mish(),
            nn.Linear(4 * hidden_nodes[0], 3 * hidden_nodes[0]),
            nn.Sigmoid()
        ) for _ in range(3)])
        self.l0_nets_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(7 * hidden_nodes[0], 7*hidden_nodes[0]),
            nn.Mish(),
            nn.Linear(7*hidden_nodes[0], hidden_nodes[0]),
        ) for _ in range(n_layers - 3)])
        self.l0_nets_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(7 * hidden_nodes[0], 7*hidden_nodes[0]),
            nn.Mish(),
            nn.Linear(7*hidden_nodes[0], hidden_nodes[0]),
        ) for _ in range(3)])
        self.init_aoi_linear = FixedLinearTransformVector(output_dim=hidden_nodes[0], freeze=True)
        self.init_tensors_linear = FixedLinearTransformTensor(input_dim=5, output_dim=hidden_nodes[0], freeze=True)

    def calculate(self,
                  batch_data : Dict[str, torch.Tensor]
                  ) -> Dict[str, torch.Tensor]:
        node_info, edge_info, init_embed, aoi, symmetries = self.get_init_info(batch_data)
        mol_ID = ''.join(map(str, batch_data['atomic_number'].tolist()))
        #plot_gaussian_arrows(batch_data['coordinate'], node_info[1],f"AA_temp_pos_plots/initial_gaussian_arrows_{mol_ID}.html")
        n_units = node_info[0].shape[1]

        for i, en_equivalent in enumerate(self.en_equivalent_blocks):
            node_info, edge_info = en_equivalent(node_info, edge_info, batch_data)
            if self.prune:
                node_info = self.prune_nodes(node_info, i, n_units, batch_data, block=True)
        #plot_gaussian_arrows(batch_data['coordinate'], node_info[1],f"AA_temp_pos_plots/gaus_ar_block_{i}_{mol_ID}.html")
        ni_1, ei_1 = ({i: node_info[i] for i in node_info.keys()},
                        {i: edge_info[i] for i in edge_info.keys()})

        ni_2, ei_2 = ({i: node_info[i] for i in node_info.keys()},
                        {i: edge_info[i] for i in edge_info.keys()})

        ni_3, ei_3 = ({i: node_info[i] for i in node_info.keys()},
                        {i: edge_info[i] for i in edge_info.keys()})

        ## 1ST HEAD
        node_info_1, edge_info_1 = self.block_heads[0](ni_1, ei_1, batch_data)
        if self.prune:
            node_info_1 = self.prune_nodes(node_info_1, 0, n_units, batch_data, block=False)

        ## 2ND HEAD
        node_info_2, edge_info_2 = self.block_heads[1](ni_2, ei_2, batch_data)
        if self.prune:
            node_info_2 = self.prune_nodes(node_info_2, 1, n_units, batch_data, block=False)

        # 3RD HEAD
        node_info_3, edge_info_3 = self.block_heads[2](ni_3, ei_3, batch_data)
        if self.prune:
            node_info_3 = self.prune_nodes(node_info_3, 2, n_units, batch_data, block=False)
        #plot_gaussian_arrows(batch_data['coordinate'], node_info_3[1], f"AA_temp_pos_plots/tensor_head_{mol_ID}.html")
        return node_info_1, node_info_2, node_info_3, init_embed

    def prune_nodes(self, ni_dict, i, n_units, batch_data, block=True):
        n_atoms = batch_data['atomic_number'].shape[0]
        COM = torch.sum(batch_data['coordinate'] * batch_data['atomic_number'][:, None], axis=0) / torch.sum(
            batch_data['atomic_number'])

        rel_pos = batch_data['coordinate'] - COM
        rel_pos = rel_pos.unsqueeze(1).repeat(1, n_units, 1)
        norms_rel_pos = torch.linalg.norm(rel_pos, dim=-1)
        norm_rel_pos = self.normalize_vector(rel_pos)
        norms_nv = torch.linalg.norm(ni_dict[1], dim=-1)
        norm_node_vectors = self.normalize_vector(ni_dict[1])
        parallel_components = self.extract_parallel_components(ni_dict[1], ev_num=-1)
        norms_pc = torch.linalg.norm(parallel_components, dim=-1)
        norm_parallel_components = self.extract_parallel_components(norm_node_vectors, ev_num=-1)
        dp_pc_nnv = torch.sum(norm_parallel_components * norm_node_vectors, dim=-1)
        dp_pc_rp = torch.sum(norm_parallel_components * norm_rel_pos, dim=-1)
        dp_rp_nv = torch.sum(norm_rel_pos * norm_node_vectors, dim=-1)

        input_vec = torch.cat([ni_dict[0], dp_pc_nnv, dp_pc_rp, dp_rp_nv, norms_nv, norms_pc, norms_rel_pos], dim=-1)
        if block:
            mean_vec_weights = self.mean_vec_mlps_blocks[i](input_vec)
            split_tensor_weights = self.split_tensor_weights_blocks[i](input_vec)
        else:
            mean_vec_weights = self.mean_vec_mlps_heads[i](input_vec)
            split_tensor_weights = self.split_tensor_weights_heads[i](input_vec)
        vectors_to_subtract_l1 = norm_parallel_components * mean_vec_weights[:, 0:n_units].unsqueeze(-1)
        pruned_vectors = norm_node_vectors - vectors_to_subtract_l1
        pruned_vectors = self.normalize_vector(pruned_vectors)
        ni_dict[1] = pruned_vectors
        if block:
            ni_dict[0] = self.l0_nets_blocks[i](input_vec)
        else:
            ni_dict[0] = self.l0_nets_heads[i](input_vec)
        ni_dict[2] = self.normalize_matrix(self.split_batch_tensor(self.normalize_matrix(ni_dict[2]), split_tensor_weights))
        return ni_dict

    def get_init_info(self,
                      batch_data : Dict[str, torch.Tensor],
                      )->Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        emb = self.embedding_layer(batch_data=batch_data)

        sb_axes, l2_tensor, symmetries = self.get_individual_axes_of_inertia(batch_data, emb)
        #sb_axes, l2_tensor = self.get_individual_axes_of_inertia_alt(batch_data, emb)
        n_ax_oi = sb_axes.shape[1]

        n_atoms, emb_dim = emb.shape

        n_tensors = l2_tensor.shape[1]

        emb_l1 = torch.zeros(n_atoms, emb_dim, 3, device=emb.device, dtype=emb.dtype)
        emb_l2 = torch.zeros(n_atoms, emb_dim, 3, 3, device=emb.device, dtype=emb.dtype)
        #sb_axes_extended = sb_axes.unsqueeze(0).expand(n_atoms, -1, -1)
        #l2_tensor = l2_tensor.unsqueeze(0).expand(n_atoms, -1, -1, -1)
        emb_l1[:, :n_ax_oi, :] = sb_axes
        emb_l2[:, :n_tensors, :, :] = l2_tensor

        node_info = {0: emb, 1: emb_l1, 2: emb_l2}
        #node_info = {0: emb}
        _, dij, _ = find_distances(batch_data)
        rbf = self.radial_fn(dij)
        edge_info = {0: rbf}
        return node_info, edge_info, emb, emb_l1, symmetries

    def normalize_vector(self, vector):
        denom = torch.norm(vector, dim=-1, keepdim=True)
        denom = torch.where(denom == 0, torch.tensor(1.0, device=denom.device), denom)
        return vector / denom
    def extract_parallel_components(self, vectors, ev_num):
        N = vectors.shape[0]

        # Step 1: Compute the covariance matrix for each set of 120 vectors
        # Outer product for each vector in each set, summed and normalized by 120
        cov_matrix = torch.mean(vectors.unsqueeze(-1) * vectors.unsqueeze(-2),dim=1).detach()  # Resulting shape: (N, 3, 3)

        # Step 2: Eigen-decomposition of the covariance matrix to find the principal axis for each set
        # Compute eigenvalues and eigenvectors for each 3x3 covariance matrix
        cov_matrix = (cov_matrix + cov_matrix.transpose(-1, -2)) / 2
        eps = 1e-6
        cov_matrix += torch.eye(3, device=cov_matrix.device) * eps
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)  # eigenvalues, eigenvectors shapes: (N, 3), (N, 3, 3)
        eigenvectors = eigenvectors.detach()

        # Select the eigenvector with the largest eigenvalue (principal axis) for each set
        principal_axis = eigenvectors[:, :, ev_num]  # shape: (N, 3), last column has the largest eigenvalue

        # Step 3: Project vectors onto the principal axis and perpendicular components for each set
        # Parallel component (projection along the principal axis)
        parallel_components = torch.einsum('nij,nj->ni', vectors, principal_axis).unsqueeze(-1) * principal_axis.unsqueeze(1)
        return parallel_components

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
                      ) for i in range(3)])
    def get_axes_of_inertia(self, batch_data, centroid):
        positions = batch_data['coordinate']
        masses = batch_data['atomic_number']
        COM = torch.sum(positions * masses[:, None], axis=0) / torch.sum(masses)

        # Center the positions
        centered_positions = positions - COM

        # Calculate the moment of inertia tensor
        I = torch.zeros((3, 3), device=positions.device)
        for i in range(len(masses)):
            I += (torch.eye(3, device=positions.device) * torch.linalg.norm(centered_positions[i]) ** 2 - torch.outer(
                centered_positions[i], centered_positions[i])) * masses[i]

        eigenvalues, axes_of_inertia = torch.linalg.eigh(I)
        ax_1 = axes_of_inertia[:, 0]  # First principal axis
        ax_2 = axes_of_inertia[:, 1]  # Second principal axis
        ax_3 = axes_of_inertia[:, 2]  # Third principal axis
        if centroid is not None:
            vector_to_dot = COM - centroid
            dot_products_1 = torch.tensor([
                torch.dot(ax_1, vector_to_dot),
                torch.dot(-ax_1, vector_to_dot),
            ])
            dot_products_2 = torch.tensor([
                torch.dot(ax_2, vector_to_dot),
                torch.dot(-ax_2, vector_to_dot),
            ])
            dot_products_3 = torch.tensor([
                torch.dot(ax_3, vector_to_dot),
                torch.dot(-ax_3, vector_to_dot),
            ])
            sorted_indices_1 = torch.argsort(dot_products_1, descending=True)
            sorted_indices_2 = torch.argsort(dot_products_2, descending=True)
            sorted_indices_3 = torch.argsort(dot_products_3, descending=True)

            sa_aoi_1 = torch.stack([ax_1, -ax_1])[sorted_indices_1]
            sa_aoi_2 = torch.stack([ax_2, -ax_2])[sorted_indices_2]
            sa_aoi_3 = torch.stack([ax_3, -ax_3])[sorted_indices_3]

            sorted_axes_aoi = torch.stack([sa_aoi_1[0], sa_aoi_2[0], sa_aoi_3[0], sa_aoi_3[1], sa_aoi_2[1], sa_aoi_1[1]])
        else:
            sorted_axes_aoi = torch.stack([ax_1, ax_2, ax_3, -ax_3, -ax_2, -ax_1])
        n_vecs = 3
        I = I / torch.linalg.norm(I)
        decomposed_tensor = self.split_tensor(I,  sorted_axes_aoi[0], sorted_axes_aoi[1], sorted_axes_aoi[2])
        return sorted_axes_aoi[:n_vecs] , decomposed_tensor

    def get_individual_axes_of_inertia(self, batch_data, emb):
        positions = batch_data['coordinate']
        masses = batch_data['atomic_number']
        n_atoms = positions.shape[0]
        # Calculate the moment of inertia tensor
        I = torch.zeros((n_atoms, 3, 3), device=positions.device, dtype=positions.dtype)
        dec_tensors = torch.zeros((n_atoms, 5, 3, 3), device=positions.device, dtype=positions.dtype)
        all_axes = torch.zeros((n_atoms, 6, 3), device=positions.device, dtype=positions.dtype)
        all_align_tensors = torch.zeros((n_atoms, 3), device=positions.device, dtype=positions.dtype)
        COM_axes_default, _ = self.get_axes_of_inertia(batch_data, None)
        COM = torch.sum(positions * masses[:, None], axis=0) / torch.sum(masses)
        for i in range(n_atoms):
            centroid = positions[i]
            centered_positions = positions - centroid
            COM_axes, COM_dec_tensor = self.get_axes_of_inertia(batch_data, centroid)

            for j in range(n_atoms):
                I[i] += (torch.eye(3, device=positions.device, dtype=positions.dtype) * torch.linalg.norm(centered_positions[j]) ** 2 - torch.outer(centered_positions[j], centered_positions[j])) * masses[j]
            eigenvalues, axes_of_inertia = torch.linalg.eigh(I[i])
            ax_1 = axes_of_inertia[:, 0]  # First principal axis
            ax_2 = axes_of_inertia[:, 1]  # Second principal axis
            ax_3 = axes_of_inertia[:, 2]  # Third principal axis
            #sorted_axes_aoi = self.canonicalize_aoi_simple(centroid, ax_1, ax_2, ax_3)
            sorted_axes_aoi, aligned_tensor = self.canonicalize_aoi_COM_aoi(COM_axes, ax_1, ax_2, ax_3, centroid, COM)
            decomposed_tensor = self.split_tensor(self.normalize_matrix(I[i]), sorted_axes_aoi[0], sorted_axes_aoi[1], sorted_axes_aoi[2])
            all_axes[i] = sorted_axes_aoi
            dec_tensors[i] = decomposed_tensor
            all_align_tensors[i] = aligned_tensor
        n_vecs = 3
        all_axes_full = self.init_aoi_linear(all_axes[:, :n_vecs, :])
        dec_tensors_full = self.init_tensors_linear(dec_tensors)
        symmetry_dict = {"axes_symmetries": all_align_tensors, "COM_axes": COM_axes_default}
        return all_axes_full, dec_tensors_full, symmetry_dict
        #return all_axes[:, :n_vecs, :], dec_tensors

    def canonicalize_aoi_simple(self, vector_to_dot, ax_1, ax_2, ax_3):
        dot_products_1 = torch.tensor([
            torch.dot(ax_1, vector_to_dot),
            torch.dot(-ax_1, vector_to_dot),
        ])
        dot_products_2 = torch.tensor([
            torch.dot(ax_2, vector_to_dot),
            torch.dot(-ax_2, vector_to_dot),
        ])
        dot_products_3 = torch.tensor([
            torch.dot(ax_3, vector_to_dot),
            torch.dot(-ax_3, vector_to_dot),
        ])

        sorted_indices_1 = torch.argsort(dot_products_1, descending=True)
        sorted_indices_2 = torch.argsort(dot_products_2, descending=True)
        sorted_indices_3 = torch.argsort(dot_products_3, descending=True)

        sa_aoi_1 = torch.stack([ax_1, -ax_1])[sorted_indices_1]
        sa_aoi_2 = torch.stack([ax_2, -ax_2])[sorted_indices_2]
        sa_aoi_3 = torch.stack([ax_3, -ax_3])[sorted_indices_3]
        # sorted_axes_aoi = torch.stack([ax_1, ax_2, ax_3]) * (torch.randint(0, 2, (3,), device=COM.device).float() * 2 - 1)
        # sorted_axes_aoi = torch.cat([sorted_axes_aoi, -sorted_axes_aoi])
        sorted_axes_aoi = torch.stack([sa_aoi_1[0], sa_aoi_2[0], sa_aoi_3[0], sa_aoi_3[1], sa_aoi_2[1], sa_aoi_1[1]])
        return sorted_axes_aoi

    def canonicalize_aoi_COM_aoi(self, COM_axes, ax_1, ax_2, ax_3, centroid, COM):
        COM_a1 = COM_axes[0]
        COM_a2 = COM_axes[1]
        COM_a3 = COM_axes[2]
        aligned_tensor = torch.tensor([0, 0, 0], device=COM.device, dtype=torch.bool)
        dot_vec = COM-centroid
        dots_with_rel_vec = torch.tensor([
            torch.dot(ax_1, dot_vec),
            torch.dot(ax_2, dot_vec),
            torch.dot(ax_3, dot_vec),
        ])
        if torch.abs(dots_with_rel_vec[0]) > 0.01:
            vector_to_use_1 = dot_vec
            aligned_tensor[0] = True
        else:
            find_axis_dot_products_1 = torch.abs(torch.tensor([
                torch.dot(ax_1, COM_a1),
                torch.dot(ax_1, COM_a2),
                torch.dot(ax_1, COM_a3),
            ]))
            vector_to_use_1 = COM_axes[torch.argmax(find_axis_dot_products_1)]

        if torch.abs(dots_with_rel_vec[1]) > 0.01:
            vector_to_use_2 = dot_vec
            aligned_tensor[1] = True
        else:
            find_axis_dot_products_2 = torch.abs(torch.tensor([
                torch.dot(ax_2, COM_a1),
                torch.dot(ax_2, COM_a2),
                torch.dot(ax_2, COM_a3),
            ]))
            vector_to_use_2 = COM_axes[torch.argmax(find_axis_dot_products_2)]

        if torch.abs(dots_with_rel_vec[2]) > 0.01:
            vector_to_use_3 = dot_vec
            aligned_tensor[2] = True
        else:
            find_axis_dot_products_3 = torch.abs(torch.tensor([
                torch.dot(ax_3, COM_a1),
                torch.dot(ax_3, COM_a2),
                torch.dot(ax_3, COM_a3),
            ]))
            vector_to_use_3 = COM_axes[torch.argmax(find_axis_dot_products_3)]

        dot_products_1 = torch.tensor([
            torch.dot(ax_1, vector_to_use_1),
            torch.dot(-ax_1, vector_to_use_1),
        ])
        dot_products_2 = torch.tensor([
            torch.dot(ax_2, vector_to_use_2),
            torch.dot(-ax_2, vector_to_use_2),
        ])
        dot_products_3 = torch.tensor([
            torch.dot(ax_3, vector_to_use_3),
            torch.dot(-ax_3, vector_to_use_3),
        ])

        sorted_indices_1 = torch.argsort(dot_products_1, descending=True)
        sorted_indices_2 = torch.argsort(dot_products_2, descending=True)
        sorted_indices_3 = torch.argsort(dot_products_3, descending=True)

        sa_aoi_1 = torch.stack([ax_1, -ax_1])[sorted_indices_1]
        sa_aoi_2 = torch.stack([ax_2, -ax_2])[sorted_indices_2]
        sa_aoi_3 = torch.stack([ax_3, -ax_3])[sorted_indices_3]
        # sorted_axes_aoi = torch.stack([ax_1, ax_2, ax_3]) * (torch.randint(0, 2, (3,), device=COM.device).float() * 2 - 1)
        # sorted_axes_aoi = torch.cat([sorted_axes_aoi, -sorted_axes_aoi])
        sorted_axes_aoi = torch.stack([sa_aoi_1[0], # if aligned_tensor[0] else COM_a1,
                                       sa_aoi_2[0], # if aligned_tensor[1] else COM_a2,
                                       sa_aoi_3[0], # if aligned_tensor[2] else COM_a3,
                                        sa_aoi_3[1], # if aligned_tensor[2] else COM_a3,
                                        sa_aoi_2[1], # if aligned_tensor[1] else COM_a2,
                                        sa_aoi_1[1], # if aligned_tensor[0] else COM_a1
                                      ]
                                         )
        return sorted_axes_aoi, aligned_tensor

    def canonicalize_aoi(self, centroid, positions, masses, ax_1, ax_2, ax_3, i):
        alignment_vec = torch.tensor([0, 0, 0], device=positions.device, dtype=torch.bool)
        centered_positions = positions - centroid
        n_atoms = positions.shape[0]
        indices = torch.tensor([j for j in range(n_atoms) if j != i], device=positions.device)
        local_COM = torch.sum(positions[indices] * masses[indices][:, None], axis=0) / torch.sum(masses[indices])
        COM = torch.sum(positions * masses[:, None], axis=0) / torch.sum(masses)
        dists = torch.linalg.norm(centered_positions[indices], dim=-1)
        nearest_neighbors = positions[indices][dists == dists.min()]
        if len(nearest_neighbors) > 1:
            nn_vec = torch.mean(nearest_neighbors, axis=0)
        else:
            nn_vec = nearest_neighbors[0]
        nn_vec_rel_pos = nn_vec - centroid
        nn_vec_rel_to_COM = nn_vec - COM
        nn_vec_rel_to_local_COM = nn_vec - local_COM
        COM_rel_pos = COM - centroid
        local_COM_rel_pos = local_COM - centroid
        vectors_to_dot = [COM_rel_pos, local_COM_rel_pos]

        deltas = []
        dp_vecs = []
        for vector_to_dot in vectors_to_dot:
            dot_products_1 = torch.tensor([
                torch.dot(ax_1, vector_to_dot),
                torch.dot(-ax_1, vector_to_dot),
            ])
            deltas.append(torch.abs(dot_products_1[0] - dot_products_1[1]))
            dp_vecs.append(dot_products_1)
        max_d_ind = torch.argmax(torch.tensor(deltas))
        dot_products_1 = dp_vecs[max_d_ind]
        sorted_indices_1 = torch.argsort(dot_products_1, descending=True)
        if torch.tensor(deltas)[max_d_ind] > 0.3:
            sa_aoi_1 = torch.stack([ax_1, -ax_1])[sorted_indices_1]
            alignment_vec[0] = True
        else:
            #rand_ind = torch.randperm(2)
            sa_aoi_1 = torch.stack([ax_1, -ax_1])

        deltas = []
        dp_vecs = []
        for vector_to_dot in vectors_to_dot:
            dot_products_2 = torch.tensor([
                torch.dot(ax_2, vector_to_dot),
                torch.dot(-ax_2, vector_to_dot),
            ])
            deltas.append(torch.abs(dot_products_2[0] - dot_products_2[1]))
            dp_vecs.append(dot_products_2)
        max_d_ind = torch.argmax(torch.tensor(deltas))
        dot_products_2 = dp_vecs[max_d_ind]
        sorted_indices_2 = torch.argsort(dot_products_2, descending=True)
        if torch.tensor(deltas)[max_d_ind] > 0.1:
            sa_aoi_2 = torch.stack([ax_2, -ax_2])[sorted_indices_2]
            alignment_vec[1] = True
        else:
            #rand_ind = torch.randperm(2)
            sa_aoi_2 = torch.stack([ax_2, -ax_2])

        deltas = []
        dp_vecs = []
        for vector_to_dot in vectors_to_dot:
            dot_products_3 = torch.tensor([
                torch.dot(ax_3, vector_to_dot),
                torch.dot(-ax_3, vector_to_dot),
            ])
            deltas.append(torch.abs(dot_products_3[0] - dot_products_3[1]))
            dp_vecs.append(dot_products_3)
        max_d_ind = torch.argmax(torch.tensor(deltas))
        dot_products_3 = dp_vecs[max_d_ind]
        sorted_indices_3 = torch.argsort(dot_products_3, descending=True)
        if torch.tensor(deltas)[max_d_ind] > 0.1:
            sa_aoi_3 = torch.stack([ax_3, -ax_3])[sorted_indices_3]
            alignment_vec[2] = True
        else:
            #rand_ind = torch.randperm(2)
            sa_aoi_3 = torch.stack([ax_3, -ax_3])
        sorted_axes_aoi = torch.stack([sa_aoi_1[0], sa_aoi_2[0], sa_aoi_3[0]])
        return sorted_axes_aoi, alignment_vec

    def normalize_matrix(self, matrix):
        orig_shape = matrix.shape
        matrix = matrix.view(-1, 3, 3)
        denom = torch.norm(matrix, dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
        denom = torch.where(denom == 0, torch.tensor(1.0, device=denom.device), denom)
        # denom = denom.mean()
        matrix = matrix / denom
        matrix = matrix.view(orig_shape)
        return matrix

    def split_tensor(self, tensor, ax_1, ax_2, ax_3):
        split_tensor = torch.zeros((5, 3, 3), device=tensor.device, dtype=tensor.dtype)
        scalar_part = (torch.trace(tensor) / 3) * torch.eye(3, device=tensor.device, dtype=tensor.dtype)

        deviatoric_tensor = tensor - scalar_part * torch.eye(3, device=tensor.device, dtype=tensor.dtype)
        # Use cross product to keep the matrix reflection invariant
        c12 = torch.cross(ax_1, ax_2)
        c23 = torch.cross(ax_2, ax_3)
        c31 = torch.cross(ax_3, ax_1)
        ax1_tens = torch.tensor([[0, c12[2], -c12[1]], [-c12[2], 0, c12[0]], [c12[1], -c12[0], 0]])
        ax2_tens = torch.tensor([[0, c23[2], -c23[1]], [-c23[2], 0, c23[0]], [c23[1], -c23[0], 0]])
        ax3_tens = torch.tensor([[0, c31[2], -c31[1]], [-c31[2], 0, c31[0]], [c31[1], -c31[0], 0]])
        split_tensor[0] = scalar_part
        split_tensor[1] = deviatoric_tensor
        split_tensor[2] = ax1_tens * (1/3)
        split_tensor[3] = ax2_tens * (1/3)
        split_tensor[4] = ax3_tens * (1/3)
        return split_tensor

    def split_batch_tensor(self, tensor, weights):
        n_atoms, n_units = tensor.shape[0], tensor.shape[1]
        scalar_part = (torch.vmap(torch.trace)(tensor.reshape(-1, 3, 3)) / 3).reshape(n_atoms, n_units, 1, 1) * torch.eye(3, device=tensor.device).unsqueeze(0).unsqueeze(0).expand(n_atoms, n_units, -1, -1) *  weights[:, 0:n_units][:, :, None, None]
        skew_part = (1/2) * (tensor - torch.transpose(tensor, -1, -2)) * weights[:, 2*n_units:3*n_units][:, :, None, None]
        sym_part = (1/2) * (tensor + torch.transpose(tensor, -1, -2)) * weights[:, n_units:2*n_units][:, :, None, None]
        output_tensors = scalar_part + sym_part +  skew_part
        return output_tensors

    def get_individual_axes_of_inertia_alt(self, batch_data, emb):
        positions = batch_data['coordinate']
        masses = batch_data['atomic_number']
        n_atoms = positions.shape[0]
        n_units = emb.shape[1]
        # Calculate the moment of inertia tensor
        I = torch.zeros((n_atoms, 3, 3), device=positions.device, dtype=positions.dtype)
        dec_tensors = torch.zeros((n_atoms, 5, 3, 3), device=positions.device, dtype=positions.dtype)
        prelim_all_axes = torch.zeros((n_atoms, 3, 3), device=positions.device, dtype=positions.dtype)
        all_axes = torch.zeros((n_atoms, 3, 3), device=positions.device, dtype=positions.dtype)
        alignment_bool = torch.zeros((n_atoms, 3), device=positions.device, dtype=torch.bool)
        for i in range(n_atoms):
            centroid = positions[i]
            centered_positions = positions - centroid
            COM = torch.sum(positions * masses[:, None], axis=0) / torch.sum(masses)

            for j in range(len(masses)):
                I[i] += (torch.eye(3, device=positions.device, dtype=positions.dtype) * torch.linalg.norm(centered_positions[j]) ** 2 - torch.outer(centered_positions[j], centered_positions[j])) * masses[j]
            eigenvalues, axes_of_inertia = torch.linalg.eigh(I[i])
            ax_1 = axes_of_inertia[:, 0]  # First principal axis
            ax_2 = axes_of_inertia[:, 1]  # Second principal axis
            ax_3 = axes_of_inertia[:, 2]  # Third principal axis
            sorted_axes_aoi, alignment_vec = self.canonicalize_aoi(centroid, positions, masses, ax_1, ax_2, ax_3, i)
            # sorted_axes_aoi = torch.stack([ax_1, ax_2, ax_3, -ax_3, -ax_2, -ax_1])
            alignment_bool[i] = alignment_vec
            prelim_all_axes[i] = sorted_axes_aoi
        found_breaker = False
        for i in range(n_atoms):
            if torch.all(alignment_bool[i]):
                all_axes[i] = prelim_all_axes[i]
                if not found_breaker:
                    found_breaker = True
                    breaker = prelim_all_axes[i][:3].sum(dim=0)
        breakers = {'ax_1': None, 'ax_2': None, 'ax_3': None}
        if not found_breaker:
            for i in range(n_atoms):
                for j in range(3):
                    if alignment_bool[i][j]:
                        if breakers[f"ax_{j+1}"] is None:
                            breakers[f"ax_{j+1}"] = prelim_all_axes[i][j].view(1, 3)
                        else:
                            breakers[f"ax_{j + 1}"] = torch.cat([breakers[f"ax_{j+1}"], prelim_all_axes[i][j].view(1, 3)], dim=0)
        for key in breakers.keys():
            if breakers[key] is not None:
                if breakers[key].shape[0] > 1:
                    rand_breaker = torch.randint(0, breakers[key].shape[0], (1,)).item()
                    breakers[key] = breakers[key][rand_breaker]
                else:
                    breakers[key] = breakers[key][0]
        nonbroken_dict = {'ax_1': None, 'ax_2': None, 'ax_3': None}
        for i, b in enumerate(alignment_bool.t()):
            if ~torch.any(b):
                rand_atom_breaker = torch.randint(0, n_atoms, (1,)).item()
                nonbroken_dict[f"ax_{i+1}"] = prelim_all_axes[rand_atom_breaker][i]

        for i in range(n_atoms):
            if not torch.all(alignment_bool[i]):
                for j in range(3):
                    if not alignment_bool[i][j]:
                        ax = prelim_all_axes[i][j]
                        to_align = torch.stack([ax, -ax])
                        if found_breaker:
                            dot_with_breaker = torch.tensor([torch.dot(ax, breaker), torch.dot(-ax, breaker)])
                            sorted_indices = torch.argsort(dot_with_breaker, descending=True)
                            all_axes[i][j] = to_align[sorted_indices][0]
                        elif breakers[f"ax_{j+1}"] is not None:
                            breaker_to_use = breakers[f"ax_{j + 1}"]
                            dot_with_breaker = torch.tensor([torch.dot(ax, breaker_to_use), torch.dot(-ax, breaker_to_use)])
                            sorted_indices = torch.argsort(dot_with_breaker, descending=True)
                            all_axes[i][j] = to_align[sorted_indices][0]
                        else:
                            all_axes[i][j] = nonbroken_dict[f"ax_{j+1}"]
                    else:
                        all_axes[i][j] = prelim_all_axes[i][j]

        all_axes_full = self.init_aoi_linear(all_axes)
        for i in range(n_atoms):
            sorted_axes_aoi = all_axes[i]
            decomposed_tensor = self.split_tensor(I[i], sorted_axes_aoi[0], sorted_axes_aoi[1], sorted_axes_aoi[2])
            dec_tensors[i] = decomposed_tensor
        dec_tensors_full = self.init_tensors_linear(dec_tensors)
        return all_axes_full, dec_tensors_full
