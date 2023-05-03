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
from torch.nn.functional import normalize

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
import math

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim = None, 
                       vdim = None, dropout = 0.0, bias = True,
                       self_attention = True, q_noise = 0.0,
                       qn_block_size = 8, batch_first = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else self.embed_dim
        self.vdim = vdim if vdim is not None else self.embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.batch_first = batch_first
        
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert( self.head_dim * num_heads == self.embed_dim )
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention

        assert( self.self_attention )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias = bias), q_noise, qn_block_size
        )

        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias = bias), q_noise, qn_block_size
        )

        self.q_proj = quant_noise(
            nn.Linear(self.embed_dim, embed_dim, bias = bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(self.embed_dim, embed_dim, bias = bias), q_noise, qn_block_size
        )

        self.reset_parameters()
        self.onnx_trace = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def forward(self, query, key, value, attn_bias, key_padding_mask, need_weights, 
                      attn_mask, before_softmax, need_head_weights):
        
        if need_head_weights:
            need_weights = True

        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]
        
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if k is not None:
            k = k.reshape(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if v is not None:
            v = v.reshape(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        #print(q.shape, k.shape, v.shape, attn_weights.shape, attn_bias.shape)
        if attn_bias is not None:
            attn_weights += attn_bias.reshape(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim = -1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        if self.batch_first:
            return attn.transpose(1, 0), attn_weights
        return attn, attn_weights


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
                    edge_dim  = 0
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
        print("local gnn_type: ",local_gnn_type)
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
            #assert(edge_dim is not None)
            if edge_dim != 0:
                print("Encoding edge_attr!")
            else:
                print("Not encoding edge_attr!")
            self.local_model = pygnn.GINEConv(gin_nn, edge_dim = edge_dim if edge_dim else None)

        elif local_gnn_type == 'GATConv':
            self.local_model = pygnn.GATConv(in_channels = dim_h,
                                             out_channels = dim_h // num_heads,
                                             heads = num_heads, edge_dim = edge_dim if edge_dim else None)
         
        if global_model_type =='None':
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout = self.attn_dropout, batch_first = True)
        elif global_model_type == 'MultheadAttention_with_bias':
            self.self_attn = MultiheadAttention(dim_h, num_heads, dropout = self.attn_dropout, batch_first = True)

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

    '''def get_attn_weight_bias(self, pos_enc: torch.Tensor):
    
        batchsize, seq_len, dim = pos_enc.size()
        pos_enc = pos_enc.reshape(batchsize, self.num_heads, seq_len, dim // self.num_heads)
        pos_enc = normalize(pos_enc, p = 2, dim = -1)
        return torch.einsum('bijl, bilk -> bijk', pos_enc, pos_enc.transpose(3, 2))
    '''
    def get_attn_weight_bias(self, pos_enc: torch.Tensor):
        batchsize, seq_len, dim = pos_enc.size()
        #print(pos_enc[0])
        '''a_2 = torch.sum(torch.square(pos_enc), dim = 2, keepdim = True)
        b_2 = torch.sum(torch.square(pos_enc), dim = 2).unsqueeze(1)
        ab = torch.einsum('ijl, ilk -> ijk', pos_enc, pos_enc.transpose(2, 1).contiguous())
        dists = a_2 + b_2 - 2 * ab
        attn_bias = torch.sqrt(dists + 1e-24)  # [bsize, seq_len, seq_len]
        attn_bias = 2 + 1e-12 - attn_bias[:, None, ...].repeat(1, self.num_heads, 1, 1)
        assert(torch.sum(attn_bias < 0).item() < 1e-12)'''
       
        tmp_list = []
        for i in range(seq_len):
            tmp_list.append(torch.sum(torch.square(pos_enc[:, i:i+1, :] - pos_enc[:, :, :]), dim = 2, keepdim = True))
        attn_bias = 2 + 1e-24 - torch.sqrt(torch.cat(tmp_list, dim = 2))
        assert(torch.sum(attn_bias < 0).item() < 1e-12)
        return attn_bias[:, None, ...].repeat(1, self.num_heads, 1, 1)    

    def forward(self, batch, need_attn_weight = False, edge_attr = None, pos_enc = None):      
        x, edge_index, node_indices = batch     #node_indices is used to denote the idx of the graph that each node belongs to        
        h = x
        h_in1 = h
        h_out_list = []
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing
            if self.local_gnn_type == 'GINEConv' or self.local_gnn_type == 'GATConv0':
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
            h_dense, mask = to_dense_batch(h, node_indices) # h_dense: [batch_size, seq_len, dim]
            if self.global_model_type == 'Transformer':
                h_attn, attn_weight = self._sa_block(h_dense, None, ~mask, need_attn_weight)
                h_attn = h_attn[mask]
                if attn_weight is not None:
                    self.attn_weights = self.process_attn_weight(attn_weight, mask)
                #print("h_attn: ", h_attn.shape)
                #print("attn_weights: ", self.attn_weights.shape)
            elif self.global_model_type == 'MultheadAttention_with_bias':
                assert(pos_enc is not None)
                pos_enc, _ = to_dense_batch(pos_enc, node_indices)
                attn_weight_bias = self.get_attn_weight_bias(pos_enc)
                h_attn, attn_weight = self._sa_block(h_dense, None, ~mask, need_attn_weight, attn_weight_bias)
                h_attn = h_attn[mask]
                
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
    
    def _sa_block(self, x, attn_mask, key_padding_mask, need_attn_weight = False, attn_weight_bias = None):
        """Self-attention block.
        """
        if attn_weight_bias is None:
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
        else:
            x, A = self.self_attn(x, x, x, attn_weight_bias, key_padding_mask, need_weights = need_attn_weight, 
                      attn_mask = attn_mask, before_softmax = False, need_head_weights = True)
            return x, A.detach()
        #self.attn_weights = self.process_attn_weight(attn_weights, mask)
                
        