from torch.nn import Linear
from typing import List
import torch.nn as nn
import importlib
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch

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
       fc_layer.add_module(act_fn, torch.nn.ReLUa()) 
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
        