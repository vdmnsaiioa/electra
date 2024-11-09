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
        self.register_buffer("norm_factor", torch.tensor(norm_factor))

    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor]
                ) -> Dict[int, torch.Tensor]:
        mol_id = ''.join(map(str, batch_data['atomic_number'].tolist()))
        message = self.graph_conv(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        res_info = torch.jit.annotate(Dict[int, torch.Tensor], {})
        idx_i = batch_data["idx_i"]
        n_atoms = batch_data['atomic_number'].shape[0]
        for way in message.keys():
            res_info[way] = _scatter_add(message[way], idx_i, dim_size=n_atoms) / self.norm_factor
        res_info = self.non_linear(self.self_interact(res_info))
        result = res_add(node_info, res_info)
        #plot_gaussian_arrows(batch_data['coordinate'], result[1], f"AA_temp_pos_plots/res_add_{mol_id}.html")
        return result


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
                 ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).float())
        self.register_buffer("std", torch.tensor(std).float())
        self.embedding_layer = embedding_layer
        self.radial_fn = radial_fn
        self.max_out_heads = max_out_heads

        max_in_way = [1] + max_out_way[:-1]
        hidden_nodes = [embedding_layer.n_channel] + output_dim
        self.en_equivalent_blocks = self.get_eq_blocks(activate_fn, max_r_way, max_in_way, max_out_way,
            hidden_nodes, norm_factor, conv_mode, update_edge, n_layers)
        self.block_heads = self.get_block_heads(head_activate_list, max_r_way, max_in_way[1:], max_out_way, max_out_heads,
            hidden_nodes, norm_factor, conv_mode, update_edge, n_layers)
        self.mean_vec_mlps_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(3*hidden_nodes[0], 2*hidden_nodes[0]),
            nn.ReLU(),
            nn.Linear(2*hidden_nodes[0], hidden_nodes[0]),
            nn.Tanh(),
        ) for _ in range(3)])
        self.mean_vec_mlps_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(5 * hidden_nodes[0], 4 * hidden_nodes[0]),
            nn.ReLU(),
            nn.Linear(4 * hidden_nodes[0], 2 * hidden_nodes[0]),
            nn.Tanh(),
        ) for _ in range(3)])

    def calculate(self,
                  batch_data : Dict[str, torch.Tensor],
                  ) -> Dict[str, torch.Tensor]:
        node_info, edge_info, init_embed, aoi = self.get_init_info(batch_data)
        mol_ID = ''.join(map(str, batch_data['atomic_number'].tolist()))
        #plot_gaussian_arrows(batch_data['coordinate'], node_info[1],f"AA_temp_pos_plots/initial_gaussian_arrows_{mol_ID}.html")
        n_atoms = batch_data['atomic_number'].shape[0]
        n_units = init_embed.shape[1]
        COM = torch.sum(batch_data['coordinate'] * batch_data['atomic_number'][:, None], axis=0) / torch.sum(batch_data['atomic_number'])
        rel_pos = batch_data['coordinate'] - COM
        rel_pos = rel_pos.unsqueeze(1).repeat(1, n_units, 1)
        for i, en_equivalent in enumerate(self.en_equivalent_blocks):
            node_info, edge_info = en_equivalent(node_info, edge_info, batch_data)
            #mean_vec = node_info[1].mean(dim=0)
            #mean_vec_exp = mean_vec.expand(n_atoms, -1, -1)
            mean_vec = node_info[1].mean(dim=1)
            mean_vec_exp = mean_vec.unsqueeze(1).repeat(1, n_units, 1)
            norm_mean = mean_vec_exp/torch.linalg.norm(mean_vec_exp, dim=-1, keepdim=True)
            norm_node_vectors = node_info[1]/torch.linalg.norm(node_info[1], dim=-1, keepdim=True)
            dot_products = torch.sum(norm_mean * norm_node_vectors, dim=-1)
            mean_vec_weights = self.mean_vec_mlps_blocks[i](torch.cat([init_embed, node_info[0], dot_products], dim=-1))
            node_info[1] = norm_node_vectors * mean_vec_weights.unsqueeze(-1)
            #plot_gaussian_arrows(batch_data['coordinate'], node_info[1],f"AA_temp_pos_plots/gaus_ar_block_{i}_{mol_ID}.html")
        ni_1, ei_1 = ({i: node_info[i] for i in node_info.keys()},
                        {i: edge_info[i] for i in edge_info.keys()})

        ni_2, ei_2 = ({i: node_info[i] for i in node_info.keys()},
                        {i: edge_info[i] for i in edge_info.keys()})

        ni_3, ei_3 = ({i: node_info[i] for i in node_info.keys()},
                        {i: edge_info[i] for i in edge_info.keys()})

        ## 1ST HEAD
        node_info_1, edge_info_1 = self.block_heads[0](ni_1, ei_1, batch_data)
        mean_vec_atom_1 = node_info_1[1].mean(dim=1).unsqueeze(1).repeat(1, n_units, 1)
        mean_vec_atom_1 = mean_vec_atom_1 / torch.linalg.norm(mean_vec_atom_1, dim=-1, keepdim=True)
        #mean_vec_system_1 =  node_info_1[1].mean(dim=0).repeat(n_atoms, 1, 1)
        #mean_vec_system_1 = mean_vec_system_1 / torch.linalg.norm(mean_vec_system_1, dim=-1, keepdim=True)
        norm_node_vec_1 = node_info_1[1]/ torch.linalg.norm(node_info_1[1], dim=-1, keepdim=True)
        dot_products_atom_1 = torch.sum(mean_vec_atom_1 * norm_node_vec_1, dim=-1)
        #dot_products_system_1 = torch.sum(mean_vec_system_1 * norm_node_vec_1, dim=-1)
        dot_products_rel_pos_atom_1 = torch.sum(rel_pos * mean_vec_atom_1, dim=-1)
        #dot_products_rel_pos_system_1 = torch.sum(rel_pos * mean_vec_system_1, dim=-1)
        dot_products_node_vecs_1 = torch.sum(rel_pos * norm_node_vec_1, dim=-1)
        mean_weights_1 = self.mean_vec_mlps_heads[0](torch.cat([init_embed, node_info_1[0], dot_products_node_vecs_1, dot_products_atom_1,  dot_products_rel_pos_atom_1], dim=-1))
        node_info_1[1] = mean_weights_1[:, 0:n_units].unsqueeze(-1) * norm_node_vec_1 + mean_vec_atom_1 * mean_weights_1[:, n_units:n_units*2].unsqueeze(-1)
        #node_info_1[1] = norm_node_vec_1 - mean_vec_1 * mean_weights_1.unsqueeze(-1) * torch.sign(torch.sum(rel_pos*mean_vec_1, dim=-1, keepdim=True)) * mean_vec_weights_1.unsqueeze(-1)

        ## 2ND HEAD
        node_info_2, edge_info_2 = self.block_heads[1](ni_2, ei_2, batch_data)
        mean_vec_atom_2 = node_info_2[1].mean(dim=1).unsqueeze(1).repeat(1, init_embed.shape[1], 1)
        mean_vec_atom_2 = mean_vec_atom_2 / torch.linalg.norm(mean_vec_atom_2, dim=-1, keepdim=True)
        #mean_vec_system_2 = node_info_2[1].mean(dim=0).repeat(n_atoms, 1, 1)
        #mean_vec_system_2 = mean_vec_system_2 / torch.linalg.norm(mean_vec_system_2, dim=-1, keepdim=True)
        norm_node_vec_2 = node_info_2[1] / torch.linalg.norm(node_info_2[1], dim=-1, keepdim=True)
        dot_products_atom_2 = torch.sum(mean_vec_atom_2 * norm_node_vec_2, dim=-1)
        #dot_products_system_2 = torch.sum(mean_vec_system_2 * norm_node_vec_2, dim=-1)
        dot_products_rel_pos_atom_2 = torch.sum(rel_pos * mean_vec_atom_2, dim=-1)
        #dot_products_rel_pos_system_2 = torch.sum(rel_pos * mean_vec_system_2, dim=-1)
        dot_products_node_vecs_2 = torch.sum(rel_pos * norm_node_vec_2, dim=-1)
        mean_weights_2 = self.mean_vec_mlps_heads[1](torch.cat([init_embed, node_info_2[0], dot_products_node_vecs_2, dot_products_atom_2, dot_products_rel_pos_atom_2], dim=-1))
        node_info_2[1] = mean_weights_2[:, 0:n_units].unsqueeze(-1) * norm_node_vec_2 + mean_vec_atom_2 * mean_weights_2[:, n_units:n_units*2].unsqueeze(-1)
        # node_info_2[1] = norm_node_vec_2 - mean_vec_2 * mean_weights_2.unsqueeze(-1) * torch.sign(torch.sum(rel_pos*mean_vec_2, dim=-1, keepdim=True)) * mean_vec_weights_2.unsqueeze(-1)

        # 3RD HEAD
        node_info_3, edge_info_3 = self.block_heads[2](ni_3, ei_3, batch_data)
        mean_vec_atom_3 = node_info_3[1].mean(dim=1).unsqueeze(1).repeat(1, init_embed.shape[1], 1)
        mean_vec_atom_3 = mean_vec_atom_3 / torch.linalg.norm(mean_vec_atom_3, dim=-1, keepdim=True)
        #mean_vec_system_3 = node_info_3[1].mean(dim=0).repeat(n_atoms, 1, 1)
        #mean_vec_system_3 = mean_vec_system_3 / torch.linalg.norm(mean_vec_system_3, dim=-1, keepdim=True)
        norm_node_vec_3 = node_info_3[1] / torch.linalg.norm(node_info_3[1], dim=-1, keepdim=True)
        dot_products_atom_3 = torch.sum(mean_vec_atom_3 * norm_node_vec_3, dim=-1)
        #dot_products_system_3 = torch.sum(mean_vec_system_3 * norm_node_vec_3, dim=-1)
        dot_products_rel_pos_atom_3 = torch.sum(rel_pos * mean_vec_atom_3, dim=-1)
        #dot_products_rel_pos_system_3 = torch.sum(rel_pos * mean_vec_system_3, dim=-1)
        dot_products_node_vecs_3 = torch.sum(rel_pos * norm_node_vec_3, dim=-1)
        mean_weights_3 = self.mean_vec_mlps_heads[2](torch.cat([init_embed, node_info_3[0], dot_products_node_vecs_3, dot_products_atom_3, dot_products_rel_pos_atom_3], dim=-1))
        node_info_3[1] = mean_weights_3[:, 0:n_units].unsqueeze(-1) * norm_node_vec_3 + mean_vec_atom_3 * mean_weights_3[:, n_units:n_units*2].unsqueeze(-1)
        #plot_gaussian_arrows(batch_data['coordinate'], node_info_tensor[1], f"AA_temp_pos_plots/tensor_head_{mol_ID}.html")

        return node_info_1, node_info_2, node_info_3, init_embed

    def get_init_info(self,
                      batch_data : Dict[str, torch.Tensor],
                      )->Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        emb = self.embedding_layer(batch_data=batch_data)

        sb_axes = self.get_axes_of_inertia(batch_data)
        n_ax_oi = sb_axes.shape[1]

        n_atoms, emb_dim = emb.shape

        emb_l1 = torch.zeros(n_atoms, emb_dim, 3, device=emb.device)
        #sb_axes_extended = sb_axes.unsqueeze(0).expand(n_atoms, -1, -1)
        emb_l1[:, :n_ax_oi, :] = sb_axes

        node_info = {0: emb, 1: emb_l1}
        #node_info = {0: emb}
        _, dij, _ = find_distances(batch_data)
        rbf = self.radial_fn(dij)
        edge_info = {0: rbf}

        return node_info, edge_info, emb, emb_l1

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
    def get_axes_of_inertia(self, batch_data):
        positions = batch_data['coordinate']
        masses = batch_data['atomic_number']
        COM = torch.sum(positions * masses[:, None], axis=0) / torch.sum(masses)
        centroid = torch.mean(positions, axis=0)
        mass_disp_vector = COM - centroid
        n_atoms = positions.shape[0]

        # Center the positions
        centered_positions = positions - COM

        # Calculate the moment of inertia tensor
        I = torch.zeros((3, 3), device=positions.device)
        for i in range(len(masses)):
            I += (torch.eye(3, device=positions.device) * torch.linalg.norm(centered_positions[i]) ** 2 - torch.outer(centered_positions[i], centered_positions[i])) * masses[i]
        vector_to_dot = COM

        eigenvalues, axes_of_inertia = torch.linalg.eigh(I)
        ax_1 = axes_of_inertia[:, 0]  # First principal axis
        ax_2 = axes_of_inertia[:, 1] # Second principal axis
        ax_3 = axes_of_inertia[:, 2] # Third principal axis
        aoi_all_atoms = torch.zeros((n_atoms, 6, 3), device=positions.device)
        for i in range(n_atoms):
            #vector_to_dot = centered_positions[i] # Just hash this to still use COM
            dot_products = torch.tensor([
                torch.dot(ax_1, vector_to_dot),
                torch.dot(-ax_1, vector_to_dot),
                torch.dot(ax_2, vector_to_dot),
                torch.dot(-ax_2, vector_to_dot),
                torch.dot(ax_3, vector_to_dot),
                torch.dot(-ax_3, vector_to_dot),
            ])
            sorted_indices = torch.argsort(dot_products, descending=True)
            sorted_axes_aoi = torch.stack([ax_1, -ax_1, ax_2, -ax_2, ax_3, -ax_3])[sorted_indices]
            aoi_all_atoms[i] = sorted_axes_aoi
        return aoi_all_atoms



