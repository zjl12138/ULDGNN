from torch.nn import Linear
from typing import List
import torch.nn as nn
import importlib
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch
import torch_geometric.nn as pygnn
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor

pygnn.conv.GINEConv
def make_fully_connected_layer(in_dim, out_dim, act_fn='', norm_type=''):    
    fc_layer = nn.Sequential()
    fc_layer.add_module('fc', nn.Linear(in_dim, out_dim))
    if norm_type != '':
        norm_module = importlib.import_module("torch.nn")
        fc_layer.add_module(norm_type, getattr(norm_module,norm_type)(out_dim))
    if act_fn == 'LeakyReLU':
        #act_module = importlib.import_module("torch.nn")
        fc_layer.add_module(act_fn,torch.nn.LeakyReLU())
    elif act_fn=='ReLU':
       fc_layer.add_module(act_fn, torch.nn.ReLU()) 
    return fc_layer

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads,  batch_norm=True, residual=False, activation=F.elu, concat=True):
        super(GATLayer,self).__init__()
        self.residual = residual
        self.activation = activation
        self.batch_norm = batch_norm
        self.linear = nn.Linear(in_dim,out_dim*num_heads)
        self.gatconv = GATConv(in_dim, out_dim, num_heads, concat)
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim * num_heads)

    def forward(self, x, edge_index):
        h_in =self.linear(x) # for residual connection
        h = self.gatconv(x, edge_index).flatten(1) 
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = self.activation(h)
        if self.residual:
            h = h_in + h # residual connection
        return h 

class GPSLayer(nn.Module):
    def __init__(self, dim_h, 
                    local_gnn_type, 
                    global_model_type, 
                    num_heads, 
                    act =' relu',
                    dropout = 0.0, 
                    attn_dropout = 0.0, 
                    layer_norm = False, 
                    batch_norm = True,
                    ):
        super(GPSLayer, self).__init__()
        self.dim_h =  dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.activation = torch.nn.ReLU() if act=='relu' else torch.nn.LeakyReLU()
        self.attn_weights = 0
        self.local_gnn_type = local_gnn_type

        if local_gnn_type == 'None':
            self.local_model = None
        
        elif local_gnn_type == 'GINConv':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation,
                                   Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINConv(gin_nn)
        
        elif local_gnn_type == 'GINEConv':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation,
                                   Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINEConv(gin_nn)

        elif local_gnn_type == 'GATConv':
            self.local_model = pygnn.GATConv(in_channels = dim_h,
                                             out_channels = dim_h // num_heads,
                                             heads = num_heads)
        
        if global_model_type =='None':
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first = True)

        self.global_model_type = global_model_type

        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
           
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
        
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def process_attn_weight(self, attn_weights, mask):
        '''
        attn_weights: 
        '''
        tensor_list = []
        #print(attn_weights.shape, attn_weights)
        for i in range(mask.shape[0]):
            rows = attn_weights[i, mask[i]]
            cols = rows[:, mask[i]]
            #mask_ = torch.ones((cols.shape[0],cols.shape[0]), device = cols.get_device()) - torch.eye(cols.shape[0],cols.shape[0], device = cols.get_device())
            tensor_list.append(cols)
        attn_weights = torch.block_diag(*tensor_list)       
        #print(attn_weights)
        return attn_weights

    def forward(self, batch, need_attn_weight = False, edge_attr = None):      
        x, edge_index, node_indices = batch     #node_indices is used to denote the idx of the graph that each node belongs to        
        h = x
        h_in1 = h
        h_out_list = []
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing
            if self.local_gnn_type == 'GINEConv':
                assert(edge_attr is not None)
                h_local = self.local_model(x, edge_index, edge_attr)
            else:
                h_local = self.local_model(x, edge_index)
            h_local = self.dropout_local(h_local)
            h_local = h_in1 + h_local
        assert(not (self.layer_norm and self.batch_norm) )
        if self.layer_norm:
            h_local = self.norm1_local(h_local, node_indices)
        if self.batch_norm:
            h_local = self.batch_norm(h_local)
        h_out_list.append(h_local)

        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, node_indices)
            if self.global_model_type == 'Transformer':
                h_attn, attn_weight = self._sa_block(h_dense, None, ~mask, need_attn_weight)
                h_attn = h_attn[mask]
                if attn_weight is not None:
                    self.attn_weights = self.process_attn_weight(attn_weight, mask)
                #print("h_attn: ", h_attn.shape)
                #print("attn_weights: ", self.attn_weights.shape)
            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, node_indices)
            if self.batch_norm:
                h_attn =  self.norm1_attn(h_attn)
            h_out_list.append(h_attn)
        h = sum(h_out_list)
        h = h + self._ff_block(h)
        h = self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, node_indices)
        if self.batch_norm:
            h = self.norm2(h)
        return h

    def _ff_block(self, x):
        """
        Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))
    
    def _sa_block(self, x, attn_mask, key_padding_mask, need_attn_weight = False):
        """Self-attention block.
        """
        if need_attn_weight:
            x, A = self.self_attn(x, x, x,
                               attn_mask = attn_mask,
                               key_padding_mask = key_padding_mask,
                               need_weights = True)
            return x, A.detach()
       
        x = self.self_attn(x, x, x,
                               attn_mask = attn_mask,
                               key_padding_mask = key_padding_mask,
                               need_weights = False)[0]
        return x, None
        #self.attn_weights = self.process_attn_weight(attn_weights, mask)
                
        