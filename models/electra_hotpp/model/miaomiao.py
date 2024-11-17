from typing import Callable, List, Dict, Optional, Literal
import torch
from torch import nn
from .base import AtomicModule
from ..layer import EmbeddingLayer, RadialLayer, ReadoutLayer, SelfInteractionLayer
from ..layer.equivalent import MultiBodyLayer, GraphConvLayer, NonLinearLayer, GraphNorm, TensorProductLayer
from ..utils import find_distances, _scatter_add, res_add
from .miao import MiaoNet
from tools.graph_tools import plot_gaussian_arrows

#TODO graph norm
class UpdateNodeBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_n_body     : int,
                 max_r_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 norm_factor    : float=1.0,
                 norm           : str='none',
                 activate_fn    : str='silu',
                 conv_mode      : Literal['node_j', 'node_edge']='node_j',
                 symbreak       : bool=False,
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
        self.self_interact = MultiBodyLayer(max_n_body=max_n_body,
                                            input_dim=input_dim, 
                                            output_dim=output_dim,
                                            max_way=max_out_way)
        self.non_linear = NonLinearLayer(activate_fn=activate_fn,
                                         max_way=max_out_way,
                                         input_dim=output_dim)
        if symbreak:
            self.symbreak_linear = nn.Linear(output_dim+1, output_dim, bias=False)
            self.symbreak_linear_final = nn.Linear(output_dim, output_dim, bias=False)
            self.symbreak_mlp_weights = nn.Sequential(
                nn.Linear(2*output_dim, 2*output_dim),
                nn.ReLU(),
                nn.Linear(2*output_dim, 2*output_dim),
                nn.Sigmoid()
            )
            self.symbreak_mlp_final = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, 4*output_dim),
                nn.Sigmoid()
            )
            self.self_interact_sb = SelfInteractionLayer(input_dim=input_dim,
                                                         max_way=max_out_way,
                                                         output_dim=output_dim)
            self.non_linear_sb = NonLinearLayer(activate_fn=activate_fn,
                                             max_way=max_out_way,
                                             input_dim=output_dim)
        self.register_buffer("norm_factor", torch.tensor(norm_factor))

        self.norm = None
        if norm == 'graph':
            self.norm = GraphNorm(max_way=max_out_way, n_channel=output_dim)
            

    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor],
                sym_break_set: Optional[Dict]=None,
                ) -> Dict[int, torch.Tensor]:
        message = self.graph_conv(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        res_info = {}
        idx_i = batch_data["idx_i"]
        n_atoms = batch_data['atomic_number'].shape[0]
        n_units = node_info[0].shape[1]
        for way in message.keys():
            res_info[way] = _scatter_add(message[way], idx_i, dim_size=n_atoms) / self.norm_factor
        #res_info = self.non_linear(self.self_interact(res_info))
        if sym_break_set is not None:
            break_symmetry = True
            if break_symmetry:
                added = res_add(node_info, res_info)
                new_dict = {key: val for key, val in added.items()}
                new_dict[1] = torch.zeros_like(new_dict[1], device=new_dict[1].device)
                for i, key in enumerate(sym_break_set.keys()):
                    for vector in sym_break_set[key]:
                        symbreak_input = torch.cat([added[1], vector[None, None, :].expand(n_atoms, 1, -1)], dim=1)
                        sb_lin = self.symbreak_linear(torch.transpose(symbreak_input, -1, -2))
                        sb_lin = torch.transpose(sb_lin, -1, -2)
                        symbreak_weights = self.symbreak_mlp_weights(torch.cat([added[0], added[0][i][None, :].expand(n_atoms, -1)], dim= -1))
                        new_dict[1] += sb_lin * symbreak_weights[:, 0:n_units][:, :, None] - added[1] * symbreak_weights[:, n_units:2*n_units][:, :, None]
                symbreak_weights_final = self.symbreak_mlp_final(new_dict[0])
                new_dict[1] = new_dict[1] * symbreak_weights_final[:, 0:n_units][:, :, None]
                new_dict[1] = torch.transpose(self.symbreak_linear_final(torch.transpose(new_dict[1], 1, 2)), 1, 2)
                new_dict = self.non_linear_sb(self.self_interact_sb(new_dict))
                #plot_gaussian_arrows(batch_data['coordinate'], new_dict[1], f"AA_temp_pos_plots/new_dict.html")
                new_dict[1] = self.normalize_vector(new_dict[1])
                #plot_gaussian_arrows(batch_data['coordinate'], new_dict[1], f"AA_temp_pos_plots/new_dict_2.html")
                added[0] = new_dict[0] * symbreak_weights_final[:, 2*n_units: 3*n_units]
                added[1] = new_dict[1] * symbreak_weights_final[:, 2*n_units: 3*n_units][:, :, None]
                added[2] = new_dict[2] * symbreak_weights_final[:, 2*n_units: 3*n_units][:, :, None, None]
                #plot_gaussian_arrows(batch_data['coordinate'], res_info[1], f"AA_temp_pos_plots/res_info.html")
                added = self.non_linear(self.self_interact(added))
                added[1] = added[1] * symbreak_weights_final[:, 3*n_units:4*n_units][:, :, None]
                added[1] = self.normalize_vector(added[1])
                return added
                #plot_gaussian_arrows(batch_data['coordinate'], res_info[1], f"AA_temp_pos_plots/res_info_final.html")
                #print("Done")
            else:
                res_info = self.non_linear(self.self_interact(res_info))
                if self.norm is not None:
                    res_info = self.norm(res_info, batch_data['batch'], batch_data['n_atoms'])
        return res_add(node_info, res_info)

    def normalize_vector(self, vector):
        denom = torch.norm(vector, dim=-1, keepdim=True)
        denom = torch.where(denom == 0, torch.tensor(1.0, device=denom.device), denom)
        return vector / denom


class UpdateEdgeBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_n_body     : int,
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
        self.self_interact = MultiBodyLayer(max_n_body=max_n_body,
                                            input_dim=input_dim,
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


class MiaoMiaoBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_n_body     : int,
                 max_r_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 norm_factor    : float=1.0,
                 activate_fn    : str='silu',
                 norm           : str='none',
                 conv_mode      : Literal['node_j', 'node_edge']='node_j',
                 update_edge    : bool=False,
                 symbreak       : bool=False,
                 ) -> None:
        super().__init__()
        self.node_block = UpdateNodeBlock(radial_fn=radial_fn, 
                                          max_n_body=max_n_body,
                                          max_r_way=max_r_way, 
                                          max_in_way=max_in_way,
                                          max_out_way=max_out_way,
                                          input_dim=input_dim, 
                                          output_dim=output_dim, 
                                          norm_factor=norm_factor,
                                          norm=norm,
                                          activate_fn=activate_fn,
                                          conv_mode=conv_mode,
                                          symbreak=symbreak)
        if update_edge:
            self.edge_block = UpdateEdgeBlock(radial_fn=radial_fn, 
                                            max_n_body=max_n_body,
                                            max_r_way=max_r_way, 
                                            max_in_way=max_in_way,
                                            max_out_way=max_out_way,
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            activate_fn=activate_fn,
                                            conv_mode=conv_mode)
        self.update_edge = update_edge


    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor],
                sym_break_set: Optional[Dict]=None,
                ) -> Dict[int, torch.Tensor]:
        node_info = self.node_block(node_info=node_info, edge_info=edge_info, batch_data=batch_data, sym_break_set=sym_break_set)
        if self.update_edge:
            edge_info = self.edge_block(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        return node_info, edge_info

class MiaoMiaoNet(MiaoNet):
    def __init__(self,
                 embedding_layer : EmbeddingLayer,
                 radial_fn       : RadialLayer,
                 n_layers        : int,
                 max_n_body      : List[int],
                 max_r_way       : List[int],
                 max_out_way     : List[int],
                 max_out_heads   : List[int],
                 output_dim      : List[int],
                 activate_fn     : str="silu",
                 head_activate_list: List[str]=['silu', 'jilu', 'tanh'],
                 mean            : float=0.,
                 std             : float=1.,
                 norm_factor     : float=1.,
                 norm_heads      : List[str] = ['none', 'none', 'none'],
                 norm_blocks     : str='none',
                 bilinear        : bool=False,
                 conv_mode       : Literal['node_j', 'node_edge']='node_j',
                 update_edge     : bool=False,
                 ):
        self.max_n_body = max_n_body
        self.norm_heads = norm_heads
        self.norm_blocks = norm_blocks
        self.max_out_heads = max_out_heads
        super().__init__(embedding_layer, radial_fn, n_layers, max_r_way, max_out_way, max_out_heads, output_dim, activate_fn, head_activate_list, mean, std, norm_factor, bilinear, conv_mode, update_edge)

    def get_eq_blocks(self, activate_fn, max_r_way, max_in_way, max_out_way,
            hidden_nodes, norm_factor, conv_mode, update_edge, n_layers):
        return nn.ModuleList([
            MiaoMiaoBlock(activate_fn=activate_fn,
                          radial_fn=self.radial_fn.replicate(),
                          # Use factory method, so the radial_fn in each layer are different
                          max_n_body=self.max_n_body[i],
                          max_r_way=max_r_way[i],
                          max_in_way=max_in_way[i],
                          max_out_way=max_out_way[i],
                          input_dim=hidden_nodes[i],
                          output_dim=hidden_nodes[i + 1],
                          norm_factor=norm_factor,
                          conv_mode=conv_mode,
                          norm=self.norm_blocks,
                          update_edge=update_edge,
                          ) for i in range(n_layers-3)])
            # Subtract 3 because we use 3 block heads too and need the lists to be at least

    def get_block_heads(self, head_activate_list, max_r_way, max_in_way, max_out_way, max_out_heads,
                      hidden_nodes, norm_factor, conv_mode, update_edge, n_layers):
        return nn.ModuleList([
            MiaoMiaoBlock(activate_fn=head_activate_list[i],
                          radial_fn=self.radial_fn.replicate(),
                          # Use factory method, so the radial_fn in each layer are different
                          max_n_body=self.max_n_body[i],
                          max_r_way=max_r_way[i],
                          max_in_way=max_in_way[i],
                          max_out_way=self.max_out_heads[i],
                          input_dim=hidden_nodes[i],
                          output_dim=hidden_nodes[i + 1],
                          norm_factor=norm_factor,
                          conv_mode=conv_mode,
                          update_edge=update_edge,
                          norm=self.norm_heads[i],
                          symbreak=True,
                          ) for i in range(3)])
