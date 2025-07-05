from typing import Callable, List, Dict, Optional, Literal, Tuple
import torch
from torch import nn
from torch.cuda import device
from .base import AtomicModule
from ..layer import EmbeddingLayer, RadialLayer, ReadoutLayer
from ..layer.equivalent import NonLinearLayer, GraphConvLayer, SelfInteractionLayer, FixedLinearTransformVector, FixedLinearTransformTensor, GraphNorm
from ..utils import find_distances, _scatter_add, res_add, TensorAggregateOP
import copy
from ase.data import chemical_symbols
from tools.atom_tools import valence_electrons
from ase import Atoms
from ase.neighborlist import neighbor_list
from tools.graph_tools import plot_gaussian_arrows

def compile_with_dynamic_shapes(fn):
    torch._dynamo.config.dynamic_shapes = True
    return torch.compile(fn)
    #return fn

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
        self.norm = GraphNorm(max_way=max_out_way, n_channel=output_dim)

    #@compile_with_dynamic_shapes
    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor],
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

    #@compile_with_dynamic_shapes
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
            nn.Linear(7 * hidden_nodes[0], 5 * hidden_nodes[0]),
            nn.Mish(),
            nn.Linear(5 * hidden_nodes[0], 3 * hidden_nodes[0]),
            nn.Sigmoid()
        ) for _ in range(n_layers - 3)])
        self.split_tensor_weights_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(7 * hidden_nodes[0], 5 * hidden_nodes[0]),
            nn.Mish(),
            nn.Linear(5 * hidden_nodes[0], 3 * hidden_nodes[0]),
            nn.Sigmoid()
        ) for _ in range(3)])
        self.l0_nets_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(7 * hidden_nodes[0], 3*hidden_nodes[0]),
            nn.Mish(),
            nn.Linear(3*hidden_nodes[0], hidden_nodes[0]),
        ) for _ in range(n_layers - 3)])
        self.l0_nets_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(7 * hidden_nodes[0], 3*hidden_nodes[0]),
            nn.Mish(),
            nn.Linear(3*hidden_nodes[0], hidden_nodes[0]),
        ) for _ in range(3)])
        self.units = hidden_nodes[0]
        self.init_aoi_linear = FixedLinearTransformVector(output_dim=self.units, freeze=True)
        self.init_tensors_linear = FixedLinearTransformTensor(input_dim=5, output_dim=self.units, freeze=True)

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
        #plot_gaussian_arrows(batch_data['coordinate'], node_info_3[1], f"/home/energy/jels/gaussian_arrows_trained/tensor_head_{mol_ID}.html")
        node_info_1[1] = node_info_1[1]
        node_info_2[1] = node_info_2[1]
        node_info_3[1] = node_info_3[1]
        return node_info_1, node_info_2, node_info_3, init_embed

    #@compile_with_dynamic_shapes
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
        # Makes sense to normalize here to ensure a bit more smoothness (these can be small sometimes and we are mostly interested in the direction)
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
        ni_dict[2] = self.split_batch_tensor(ni_dict[2], split_tensor_weights)
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
        #cell_vectors = batch_data['cell'].view(1, 3, 3)
        #cell_vectors = cell_vectors.repeat(n_atoms, 1, 1)
        #emb_l1[:, n_ax_oi:3+n_ax_oi, :] = cell_vectors / torch.norm(cell_vectors, dim=-1, keepdim=True)

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

    #@compile_with_dynamic_shapes
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

    def get_individual_axes_of_inertia_nonopt(self, batch_data, emb):
        positions = batch_data['coordinate']
        masses = batch_data['atomic_number'].clone().detach()
        n_atoms = positions.shape[0]
        # Calculate the moment of inertia tensor
        I = torch.zeros((n_atoms, 3, 3), device=positions.device, dtype=positions.dtype)
        dec_tensors = torch.zeros((n_atoms, 5, 3, 3), device=positions.device, dtype=positions.dtype)
        all_axes = torch.zeros((n_atoms, 6, 3), device=positions.device, dtype=positions.dtype)
        for i in range(n_atoms):
            centroid = positions[i]
            centered_positions = positions - centroid
            COM = torch.sum(positions * masses[:, None], axis=0) / torch.sum(masses)
            if (batch_data["offset"] != 0).sum() != 0:
                # Assume cell is a (3,3) matrix from batch_data["cell"]
                cell = batch_data["cell"].view(3, 3)
                cell_inv = torch.inverse(cell)

                # centered_positions is already defined as positions - centroid (shape: [n_atoms, 3])
                # Convert to fractional coordinates
                fractional = torch.matmul(centered_positions, cell_inv.T)

                # Wrap the fractional coordinates into the [-0.5, 0.5] interval
                fractional_wrapped = fractional - torch.round(fractional)

                # Convert back to Cartesian coordinates
                centered_positions = torch.matmul(fractional_wrapped, cell.T)

            for j in range(n_atoms):
                I[i] += (torch.eye(3, device=positions.device, dtype=positions.dtype) * torch.linalg.norm(centered_positions[j]) ** 2 - torch.outer(centered_positions[j], centered_positions[j])) * masses[j]
            eigenvalues, axes_of_inertia = torch.linalg.eigh(I[i])
            ax_1 = axes_of_inertia[:, 0]  # First principal axis
            ax_2 = axes_of_inertia[:, 1]  # Second principal axis
            ax_3 = axes_of_inertia[:, 2]  # Third principal axis
            sorted_axes_aoi = self.canonicalize_aoi_simple(centroid, ax_1, ax_2, ax_3)
            # sorted_axes_aoi = torch.stack([ax_1, ax_2, ax_3, -ax_3, -ax_2, -ax_1])
            decomposed_tensor = self.split_tensor(I[i], sorted_axes_aoi[0], sorted_axes_aoi[1], sorted_axes_aoi[2])
            all_axes[i] = sorted_axes_aoi
            dec_tensors[i] = decomposed_tensor
        n_vecs = 3
        #all_axes_full = self.init_aoi_linear(all_axes[:, :n_vecs, :])
        #dec_tensors_full = self.init_tensors_linear(dec_tensors)
        all_axes_full = all_axes
        dec_tensors_full = dec_tensors
        return all_axes_full, dec_tensors_full, None

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

    def normalize_matrix(self, matrix):
        orig_shape = matrix.shape
        matrix = matrix.view(-1, 3, 3)
        denom = torch.norm(matrix, dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
        denom = torch.where(denom == 0, torch.tensor(1.0, device=denom.device), denom)
        # denom = denom.mean()
        matrix = matrix / denom
        matrix = matrix.view(orig_shape)
        return matrix
    #@compile_with_dynamic_shapes
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

    #@compile_with_dynamic_shapes
    def split_batch_tensor(self, tensor, weights):
        n_atoms, n_units = tensor.shape[0], tensor.shape[1]

        scalar_part = (torch.vmap(torch.trace)(tensor.reshape(-1, 3, 3)) / 3).reshape(n_atoms, n_units, 1, 1) * torch.eye(3, device=tensor.device).unsqueeze(0).unsqueeze(0).expand(n_atoms, n_units, -1, -1)
        scalar_part = self.normalize_matrix(scalar_part) *  weights[:, 0:n_units][:, :, None, None]

        skew_part = (1 / 2) * (tensor - torch.transpose(tensor, -1, -2))
        skew_part = self.normalize_matrix(skew_part) * weights[:, 2*n_units:3*n_units][:, :, None, None]

        sym_part = (1/2) * (tensor + torch.transpose(tensor, -1, -2))
        sym_part = self.normalize_matrix(sym_part) * weights[:, n_units:2*n_units][:, :, None, None]

        output_tensors = scalar_part + sym_part + skew_part
        return output_tensors

    def batch_cross_matrix(self, v):
        x = v[:, 0]
        y = v[:, 1]
        z = v[:, 2]
        zero = torch.zeros_like(x)

        return torch.stack([
            torch.stack([ zero,   z,   -y  ], dim=-1),
            torch.stack([-z,      zero,  x ], dim=-1),
            torch.stack([ y,     -x,    zero], dim=-1)
        ], dim=1)  # shape: (n,3,3)

    def split_tensor_batched(self, tensors, axes):
        device = tensors.device
        dtype  = tensors.dtype
        n      = tensors.shape[0]

        trace_vals = torch.vmap(torch.trace)(tensors) #torch.trace(tensors, dim1=-2, dim2=-1)  # (n,)
        scalar_vals = trace_vals / 3.0
        eye3 = torch.eye(3, device=device, dtype=dtype).expand(n, -1, -1) # (n,3,3)
        scalar_part = scalar_vals.view(n,1,1) * eye3                       # (n,3,3)

        deviatoric_tensor = tensors - scalar_part

        ax_1, ax_2, ax_3 = axes[:,0,:], axes[:,1,:], axes[:,2,:]
        c12 = torch.cross(ax_1, ax_2, dim=-1)  # (n,3)
        c23 = torch.cross(ax_2, ax_3, dim=-1)  # (n,3)
        c31 = torch.cross(ax_3, ax_1, dim=-1)  # (n,3)

        ax1_tens = self.batch_cross_matrix(c12)/3
        ax2_tens = self.batch_cross_matrix(c23)/3
        ax3_tens = self.batch_cross_matrix(c31)/3

        return torch.stack([
            scalar_part,
            deviatoric_tensor,
            ax1_tens,
            ax2_tens,
            ax3_tens
        ], dim=1)  # => (n,5,3,3)


    def canonicalize_aoi_simple_batched(self, vectors_to_dot, axes):
        dot_products = (axes * vectors_to_dot.unsqueeze(1)).sum(dim=-1)  # (n,3)
        sign = torch.where(dot_products >= 0, 1.0, -1.0)  # (n,3)
        best_sign_axes  = sign.unsqueeze(-1) * axes       # (n,3,3)
        worst_sign_axes = -best_sign_axes                 # (n,3,3)

        sorted_axes_aoi = torch.stack([
            best_sign_axes[:,0],
            best_sign_axes[:,1],
            best_sign_axes[:,2],
            worst_sign_axes[:,2],
            worst_sign_axes[:,1],
            worst_sign_axes[:,0],
        ], dim=1)
        return sorted_axes_aoi


    def get_individual_axes_of_inertia(self, batch_data, emb):

        positions = batch_data['coordinate']      # (n,3)
        masses    = batch_data['atomic_number']   # (n,)
        n_atoms   = positions.shape[0]
        r = positions.unsqueeze(1) - positions.unsqueeze(0)  # (n,n,3)

        if (batch_data["offset"] != 0).sum() != 0:
            cell     = batch_data["cell"].view(3, 3)
            cell_inv = torch.inverse(cell)

            r_flat = r.reshape(-1, 3)  # (n*n,3)
            frac   = torch.matmul(r_flat, cell_inv.T)
            frac_wrapped = frac - torch.round(frac)
            r_flat_wrapped = torch.matmul(frac_wrapped, cell.T)
            r = r_flat_wrapped.reshape(n_atoms, n_atoms, 3)

        r_norm2 = (r**2).sum(dim=-1, keepdim=True)   # (n,n,1)
        eye3    = torch.eye(3, dtype=r.dtype, device=r.device).reshape(1,1,3,3)
        outer_r = r.unsqueeze(-1) * r.unsqueeze(-2)  # (n,n,3,3)

        masses_reshaped = masses.reshape(1,-1,1,1)   # (1,n,1,1)
        inertia_terms   = (r_norm2.unsqueeze(-1)*eye3 - outer_r) * masses_reshaped
        I = inertia_terms.sum(dim=1)  # sum over j => (n,3,3)

        eigenvals, eigenvecs = torch.linalg.eigh(I)

        axes = eigenvecs.transpose(-2, -1)  # now axes[i,j,:] is the j-th axis

        vectors_to_dot  = positions  # or your preferred reference
        sorted_axes_aoi = self.canonicalize_aoi_simple_batched(vectors_to_dot, axes)

        all_axes = sorted_axes_aoi[:, :3, :]

        dec_tensors = self.split_tensor_batched(I, all_axes)

        return all_axes, dec_tensors, None
