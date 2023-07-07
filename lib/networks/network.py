from threading import local
from turtle import pos
from ssl import cert_time_to_seconds
import torch
import torch.nn as nn
import torch_geometric
from lib.utils.nms import contains_how_much
from lib.utils import contains
from lib.networks.layers import make_fully_connected_layer, GATLayer, GPSLayer
from lib.networks.layers import GetPosEmbedder, PosEmbedder, ImageEmbedder, TypeEmbedder
from lib.networks.loss import make_classifier_loss, make_regression_loss
from lib.config import cfg as CFG
import importlib
import torch.nn.functional as F
from lib.utils import vote_clustering, vote_clustering_each_layer, IoU
from torchvision.ops import roi_align
import os

cfg = CFG.network

def make_gnn(gnn_type):
    norm_module = importlib.import_module("lib.networks.network")
    return getattr(norm_module, gnn_type)

class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.in_dim = cfg.in_dim
        self.latent_dims = cfg.latent_dims
        self.make(cfg)
        
    def make(self, cfg):
        layer_list = []
        in_dim = self.in_dim
        
        for idx, latent_dim in enumerate(self.latent_dims):
            self.add_module(
                            f'fc_{idx}',make_fully_connected_layer(in_dim,
                                                             latent_dim, 
                                                             cfg.act_fn, 
                                                             cfg.norm_type)
                           )
            in_dim = latent_dim

        self.add_module(f'fc_{len(self.latent_dims)+1}',make_fully_connected_layer(self.latent_dims[-1],
                                                          cfg.classes,
                                                          '', '')
                              )
        
    def forward(self, x, clip_val=False):
        for _, fc_layer in self.named_children():
            x = fc_layer(x)
        if clip_val:
            x = torch.nn.Tanh()(x)
        return x

class Classifier_two_branches(nn.Module):
    def __init__(self, cfg):
        super(Classifier_two_branches, self).__init__()
        self.in_dim = cfg.in_dim
        self.latent_dims = cfg.latent_dims
        self.make(cfg)
        
    def make(self, cfg):
        layer_list = []
        in_dim = self.in_dim
        
        for idx, latent_dim in enumerate(self.latent_dims):
            self.add_module(
                            f'loc_{idx}',make_fully_connected_layer(in_dim,
                                                             latent_dim, 
                                                             cfg.act_fn, 
                                                             cfg.norm_type)
                           )
            in_dim = latent_dim

        self.add_module(f'loc_{len(self.latent_dims)+1}',make_fully_connected_layer(self.latent_dims[-1],
                                                          cfg.classes,
                                                          '', '')) # cfg.classes = 36
        self.add_module(f'confid_{0}', make_fully_connected_layer(self.in_dim + cfg.classes, cfg.classes // 4,
                                                          '', ''))
        
    def forward(self, x, clip_val = False):
        x_old = x
        for layer_name, fc_layer in self.named_children():
            if 'loc' in layer_name:
                x = fc_layer(x)
                loca_params = x
            elif 'confid' in layer_name:
                confid_scores = fc_layer(torch.cat([x_old, x], dim = 1))
        x = torch.cat([loca_params, confid_scores], dim = 1)
        if clip_val:
            x = torch.nn.Tanh()(x)
        return x

class Classifier_with_gnn(nn.Module):
    def __init__(self, cfg):
        super(Classifier_with_gnn, self).__init__()
        self.in_dim = cfg.in_dim
        self.latent_dims = cfg.latent_dims
        self.gnn_latent_dims = cfg.gnn_latent_dims
        self.gnn_num_heads = cfg.num_heads
        self.make(cfg)
        

    def make(self, cfg):
        layer_list = []
        in_dim = self.in_dim
        
        for idx, (latent_dim, num_head) in enumerate(zip(self.gnn_latent_dims, self.gnn_num_heads)):
            self.add_module(
                f'GPSLayer{idx}', GPSLayer(
                    in_dim, 
                    cfg.local_gnn_type,
                    cfg.global_model_type,
                    num_head,
                    cfg.act_fn,
                    cfg.dropout,
                    cfg.attn_dropout,
                    cfg.layer_norm,
                    cfg.batch_norm
                )
            )
            in_dim = latent_dim

        for idx, latent_dim in enumerate(self.latent_dims):
            self.add_module(
                            f'fc_{idx}',make_fully_connected_layer(in_dim,
                                                             latent_dim, 
                                                             cfg.act_fn, 
                                                             cfg.norm_type)
                           )
            in_dim = latent_dim

        self.add_module(f'fc_{len(self.latent_dims)+1}',make_fully_connected_layer(self.latent_dims[-1],
                                                          cfg.classes,
                                                          '', '')
                              )
        
    def forward(self, x, edges, node_indices, clip_val=False):
        for name, fc_layer in self.named_children():
            if 'GPSLayer' in name:
                x = fc_layer((x, edges, node_indices))
            else:
                x = fc_layer(x)
        if clip_val:
            x = torch.nn.Tanh()(x)
        return x

class LayerGNN(nn.Module):
    def __init__(self, cfg):
        super(LayerGNN, self).__init__()
        self.latent_dims = cfg.latent_dims
        self.num_heads = cfg.num_heads
        self.make(cfg)
        
    def make(self, cfg):
        gnn_layer_list = []
        in_dim = cfg.in_dim
        L = len(self.latent_dims) + 1
        assert(len(self.num_heads) == len(self.latent_dims)+1)
        for idx, (latent_dim, num_head) in enumerate(zip(self.latent_dims, self.num_heads[:-1])):
            self.add_module(f'gatlayer_{idx}', GATLayer(in_dim, latent_dim,
                                                 num_head, cfg.batch_norm, 
                                                 cfg.residual, concat = cfg.concat))
            in_dim = latent_dim * num_head if cfg.concat else latent_dim
        self.add_module(f'gatlayer_{L+1}',GATLayer(in_dim, cfg.out_dim, self.num_heads[-1], False, concat = False))
        
    def forward(self, batch):
        x, edges, _ = batch
        #print(torch.max(edges))
        #print(x.shape)
        for _, gnn_layer in self.named_children():
            x = gnn_layer(x, edges)
        return x

'''
GPSLayer
def __init__(self, dim_h, 
                    local_gnn_type, 
                    global_model_type, 
                    num_heads, 
                    act='relu',
                    dropout=0.0, 
                    attn_dropout=0.0, 
                    layer_norm=False, 
                    batch_norm=True,
                    ):
'''

class GPSModel(nn.Module):
    def __init__(self, cfg):
        super(GPSModel, self).__init__()
        self.latent_dims = cfg.latent_dims
        self.num_heads = cfg.num_heads
        self.make(cfg)

    def make(self, cfg):
        in_dim = cfg.in_dim
        L = len(self.latent_dims) + 1
        for idx, (latent_dim, num_head) in enumerate(zip(self.latent_dims, self.num_heads)):
            self.add_module(
                f'GPSLayer{idx}', GPSLayer(
                    in_dim, 
                    cfg.local_gnn_type,
                    cfg.global_model_type,
                    num_head,
                    cfg.act_fn,
                    cfg.dropout,
                    cfg.attn_dropout,
                    cfg.layer_norm,
                    cfg.batch_norm,
                    cfg.edge_dim
                )
            )
            in_dim = latent_dim
    
    def set_edge_embedder(self, edge_embed):
        self.edge_embedder = edge_embed

    def forward(self, batch, edge_attr = None, pos_enc = None): # 
        x, edge_index, node_indices = batch
        #print(torch.max(edges))
        #print(x.shape)
        for idx, (name, gnn_layer) in enumerate(self.named_children()):
            if 'GPS' in name:
                x = gnn_layer((x, edge_index, node_indices), edge_attr = edge_attr, pos_enc = pos_enc)
                #if idx >= 5:
        return x 

class GPSModel_anchor_voting(nn.Module):
    def __init__(self, cfg):
        super(GPSModel_anchor_voting, self).__init__()
        self.latent_dims = cfg.latent_dims
        self.num_heads = cfg.num_heads
        self.offset_head = nn.Linear(self.latent_dims[-1], 4 * 9)
        self.make(cfg)

    def make(self, cfg):
        in_dim = cfg.in_dim
        L = len(self.latent_dims) + 1
        for idx, (latent_dim, num_head) in enumerate(zip(self.latent_dims, self.num_heads)):
            self.add_module(
                f'GPSLayer{idx}', GPSLayer(
                    in_dim, 
                    cfg.local_gnn_type,
                    cfg.global_model_type,
                    num_head,
                    cfg.act_fn,
                    cfg.dropout,
                    cfg.attn_dropout,
                    cfg.layer_norm,
                    cfg.batch_norm,
                    cfg.edge_dim
                )
            )
            in_dim = latent_dim
    
    def set_edge_embedder(self, edge_embed):
        self.edge_embedder = edge_embed

    def forward(self, batch, edge_attr = None, pos_enc = None): # 
        x, edge_index, node_indices = batch
        #print(torch.max(edges))
        #print(x.shape)
        padding_zeros = torch.zeros((x.shape[0], 4 * 9), device = x.get_device())
        #x = torch.cat((padding_zeros, x), dim = 1)
        for idx, (name, gnn_layer) in enumerate(self.named_children()):
            if 'GPS' in name:
                x = gnn_layer((x, edge_index, node_indices), edge_attr = edge_attr, pos_enc = pos_enc)
                #if idx >= 5:
                padding_zeros = padding_zeros + self.offset_head(x)
        #x = torch.cat((padding_zeros, x), dim = 1)
        #del padding_zeros
        return x, padding_zeros

class GPSModel_with_voting(nn.Module):
    def __init__(self, cfg):
        super(GPSModel_with_voting, self).__init__()
        self.latent_dims = cfg.latent_dims
        self.num_heads = cfg.num_heads
        self.offset_head = nn.Linear(self.latent_dims[-1], 4)
        self.make(cfg)

    def make(self, cfg):
        in_dim = cfg.in_dim
        L = len(self.latent_dims) + 1
        for idx, (latent_dim, num_head) in enumerate(zip(self.latent_dims, self.num_heads)):
            self.add_module(
                f'GPSLayer{idx}', GPSLayer(
                    in_dim, 
                    cfg.local_gnn_type,
                    cfg.global_model_type,
                    num_head,
                    cfg.act_fn,
                    cfg.dropout,
                    cfg.attn_dropout,
                    cfg.layer_norm,
                    cfg.batch_norm,
                    cfg.edge_dim
                )
            )
            in_dim = latent_dim
    
    def set_edge_embedder(self, edge_embed):
        self.edge_embedder = edge_embed

    def forward(self, batch, edge_attr = None, pos_enc = None): # 
        x, edge_index, node_indices = batch
        #print(torch.max(edges))
        #print(x.shape)
        padding_zeros = torch.zeros((x.shape[0], 4), device = x.get_device())
        #x = torch.cat((padding_zeros, x), dim = 1)
        for idx, (name, gnn_layer) in enumerate(self.named_children()):
            if 'GPS' in name:
                x = gnn_layer((x, edge_index, node_indices), edge_attr = edge_attr, pos_enc = pos_enc)
                #if idx >= 5:
                padding_zeros = padding_zeros + self.offset_head(x)
        x = torch.cat((padding_zeros, x), dim = 1)
        del padding_zeros
        return x 

class GPSModel_voting_update_edge_attr(nn.Module):
    def __init__(self, cfg):
        super(GPSModel_voting_update_edge_attr, self).__init__()
        self.latent_dims = cfg.latent_dims
        self.num_heads = cfg.num_heads
        self.offset_head = nn.Linear(self.latent_dims[-1], 4)
        self.make(cfg)

    def set_edge_embedder(self, edge_embed):
        self.edge_embedder = edge_embed

    def make(self, cfg):
        in_dim = cfg.in_dim
        L = len(self.latent_dims) + 1
        for idx, (latent_dim, num_head) in enumerate(zip(self.latent_dims, self.num_heads)):
            self.add_module(
                f'GPSLayer{idx}', GPSLayer(
                    in_dim, 
                    cfg.local_gnn_type,
                    cfg.global_model_type,
                    num_head,
                    cfg.act_fn,
                    cfg.dropout,
                    cfg.attn_dropout,
                    cfg.layer_norm,
                    cfg.batch_norm,
                    cfg.edge_dim
                )
            )
            in_dim = latent_dim
        
    def forward(self, batch, edge_attr = None, pos_enc = None): # pos_enc [xywh]
        x, edge_index, node_indices = batch
        #print(torch.max(edges))
        #print(x.shape)
        padding_zeros = torch.zeros((x.shape[0], 4), device = x.get_device())
        #x = torch.cat((padding_zeros, x), dim = 1)
        x_i = edge_index[0, :]
        x_j = edge_index[1, :]
        prev_edge_attr = self.edge_embedder(torch.abs(pos_enc[x_i, :] - pos_enc[x_j, :]))
        for idx, (name, gnn_layer) in enumerate(self.named_children()):
            if 'GPS' in name:
                x = gnn_layer((x, edge_index, node_indices), edge_attr = prev_edge_attr, pos_enc = pos_enc)
                offset = self.offset_head(x)
                padding_zeros = padding_zeros + offset
                pos_enc = pos_enc + offset
                pos_enc = pos_enc.detach()
                #print(pos_enc.requires_grad)
                prev_edge_attr = self.edge_embedder(torch.abs(pos_enc[x_i, :] - pos_enc[x_j, :]))
        x = torch.cat((padding_zeros, x), dim = 1)
        del padding_zeros
        return x 


class GPSModel_with_voting_v(nn.Module):#not used
    def __init__(self, cfg):
        super(GPSModel_with_voting_v, self).__init__()
        self.latent_dims = cfg.latent_dims
        self.num_heads = cfg.num_heads
        #self.offset_head = nn.Linear(self.latent_dims[-1], 4)
        self.make(cfg)
    
    def make(self, cfg):
        in_dim = cfg.in_dim
        L = len(self.latent_dims) + 1
        for idx, (latent_dim, num_head) in enumerate(zip(self.latent_dims, self.num_heads)):
            self.add_module(
                f'GPSLayer{idx}', GPSLayer(
                    in_dim, 
                    cfg.local_gnn_type,
                    cfg.global_model_type,
                    num_head,
                    cfg.act_fn,
                    cfg.dropout,
                    cfg.attn_dropout,
                    cfg.layer_norm,
                    cfg.batch_norm
                )
            )
            self.add_module(f'Linear{idx}', nn.Linear(in_dim, 4))
            in_dim = latent_dim
        
    def forward(self, batch): # 
        x, edge_index, node_indices = batch
        #print(torch.max(edges))
        #print(x.shape)
        padding_zeros = torch.zeros((x.shape[0], 4), device = x.get_device())
        #x = torch.cat((padding_zeros, x), dim = 1)
        for idx, (name, layer) in enumerate(self.named_children()):
            if 'GPS' in name:
                x = layer((x, edge_index, node_indices))
            else:
                padding_zeros = padding_zeros + layer(x)
        x = torch.cat((padding_zeros, x), dim = 1)

        del padding_zeros
        return x 

class box_refine_module(nn.Module):
    def __init__(self, cfg):
        super(box_refine_module, self).__init__()
        self.make(cfg)
        self.roi_size = cfg.roi_size
        
    def make(self, cfg):
        self.box_refine_branch = Classifier(cfg.box_refine_branch)
    
    def forward(self, img_tensors, node_indices, box_tensors : torch.Tensor):
        B, C, H, W = img_tensors.shape 
        box_tensors_old = box_tensors
        box_tensors[:, 2 : 4] += box_tensors[:, 0 : 2]
        box_tensors = box_tensors.clamp(0, 1)
        box_tensors *= torch.tensor([W, H, W, H], device = box_tensors.device)
        roi_align_feats = roi_align(
            img_tensors,
            torch.cat([node_indices.unsqueeze(1), box_tensors], dim = 1),
            output_size = self.roi_size,
            sampling_ratio = 2 
        )
        offsets = self.box_refine_branch(roi_align_feats.flatten(1))
        return box_tensors_old + offsets
        

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()   
        self.cls_loss_fn = make_classifier_loss(cfg.cls_loss)
        self.reg_loss_fn = make_regression_loss(cfg.reg_loss)
        self.huber_fn = torch.nn.L1Loss() 

        cfg.gnn_fn.in_dim = cfg.pos_embedder.out_dim
        cfg.cls_fn.in_dim = cfg.gnn_fn.out_dim
        cfg.loc_fn.in_dim = cfg.gnn_fn.out_dim

        if  'voting' in cfg.gnn_fn.gnn_type and 'anchor' not in cfg.gnn_fn.gnn_type:
            #cfg.pos_embedder.out_dim = cfg.pos_embedder.out_dim - 4
            #cfg.type_embedder.out_dim = cfg.type_embedder.out_dim - 4
            #cfg.img_embedder.out_dim  = cfg.img_embedder.out_dim - 4
            #cfg.loc_fn.in_dim = cfg.gnn_fn.out_dim - 4
            print("adding voting mechanism!")
            cfg.cls_fn.in_dim = cfg.gnn_fn.out_dim + 4
            #cfg.loc_fn.in_dim = cfg.gnn_fn.out_dim + 4
             
        self.bbox_regression_type = cfg.bbox_regression_type
        self.bbox_vote_radius = cfg.bbox_vote_radius   #used for vote clustering
        print("the bbox regression type: ", self.bbox_regression_type)

        self.edge_embedder, self.edge_dim = GetPosEmbedder(cfg.edge_embedder.multires, include_input = True)
        print("edge_dim is ", self.edge_dim)
        if self.edge_dim != cfg.pos_embedder.out_dim:
            cfg.gnn_fn.edge_dim = self.edge_dim
        else:
            cfg.gnn_fn.edge_dim = 0
        
        self.feat_embed_module = []
        if not cfg.remove_pos:
            print("not remove pos!")
            self.pos_embedder = PosEmbedder(cfg.pos_embedder)
        if not cfg.remove_type:
            self.type_embedder = TypeEmbedder(cfg.type_embedder)
        if not cfg.remove_img:
            self.img_embedder = ImageEmbedder(cfg.img_embedder)
        #cfg.gnn_fn.edge_dim = self.pos_embedder.in_dim
        
        self.gnn_fn = make_gnn(cfg.gnn_fn.gnn_type)(cfg.gnn_fn)
        self.gnn_fn.set_edge_embedder(self.edge_embedder)
        self.cls_fn = Classifier(cfg.cls_fn)
        self.loc_type = cfg.loc_type # cfg.loc_fn.loc_type
        self.gnn_type = cfg.gnn_fn.gnn_type
        print("loc_fn_type: ", cfg.loc_fn.loc_type)
        self.refine_box_module: box_refine_module = None
        if cfg.loc_fn.loc_type == 'classifier_with_gnn':
            self.loc_fn = Classifier_with_gnn(cfg.loc_fn)
        elif  cfg.loc_fn.loc_type == 'classifier_two_branches':
            self.loc_fn = Classifier_two_branches(cfg.loc_fn)
        else:
            self.loc_fn = Classifier(cfg.loc_fn)
        if cfg.train_mode == 3:
            print("fix cls branch!")
            self.fix_network(self.cls_fn)
        elif cfg.train_mode == 2:
            print("train refine branch only!")
            self.fix_network(self.pos_embedder)
            self.fix_network(self.type_embedder)
            self.fix_network(self.img_embedder)
            self.fix_network(self.gnn_fn)
            self.fix_network(self.cls_fn)
            self.fix_network(self.loc_fn)
            self.refine_box_module = box_refine_module(cfg.box_refine_fn)
            
        elif cfg.train_mode == 1:
            print("fix localization branch!")
            self.fix_network(self.loc_fn)
        else:
            print("train two branches together")
        
    def fix_network(self, model):
        for n, params in model.named_parameters():
            params.requires_grad = False

    def open_network(self, model):
        for n, params in model.named_parameters():
            params.requires_grad = True

    def change_to_mode(self, train_mode: str):
        if train_mode == '1_to_0':
            self.open_network(self.loc_fn)

    def process_output_data(self, output, layer_rects, anchor_box_wh = None):
        logits, local_params, confidence, voting_offset = output

        if self.bbox_regression_type == 'center_regress':
            xy = layer_rects[:, 0 : 2] + layer_rects[:, 2 : 4] * 0.5 + local_params[:, 0 : 2] - local_params[:, 2 : 4] * 0.5
            wh = local_params[:, 2 : 4]
            bboxes = torch.cat((xy, wh), dim=1)
            centroids = layer_rects[:, 0 : 2] + layer_rects[:, 2 : 4] * 0.5 + local_params[:, 0 : 2]
            #pred_bboxes.requires_grad = True
            #local_params[:, 0 : 2] = layer_rects[:, 0 : 2] + layer_rects[:, 2 : 4] * 0.5 + local_params[:, 0 : 2] - local_params[:, 2 : 4] * 0.5
        elif self.bbox_regression_type == 'offset_based_on_layer':
            bboxes = local_params + layer_rects
            centroids = bboxes[:, 0 : 2] + 0.5 * bboxes[:, 2 : 4]
            if voting_offset is not None:
                voting_offset = voting_offset + layer_rects

        elif self.bbox_regression_type == 'voting':
            assert(local_params.shape[1] == 6)
            centroids = layer_rects[:, 0 : 2] + layer_rects[:, 2 : 4] * 0.5 + local_params[:, 0 : 2]
            anchor_bboxes = vote_clustering(centroids, layer_rects, radius = self.bbox_vote_radius)
            #anchor_bboxes = vote_clustering_each_layer(centroids, layer_rects, radius = self.bbox_vote_radius)
            bboxes = local_params[:, 2 : 6] + anchor_bboxes
                        
        elif self.bbox_regression_type == 'step_voting':  #local_params[0:2]:offset, local_params[2:4]: wh
            centroids = layer_rects[:, 0 : 2] + layer_rects[:, 2 : 4] * 0.5 + local_params[:, 0 : 2]
            anchor_bboxes = vote_clustering(centroids, layer_rects, radius = self.bbox_vote_radius)
            #anchor_bboxes = vote_clustering_each_layer(centroids, layer_rects, radius = self.bbox_vote_radius)
            bboxes = local_params[:, 2 : 6] + anchor_bboxes
        
        elif self.bbox_regression_type == 'direct':
            centroids = local_params[:, 0 : 2] + 0.5 * local_params[:, 2 : 4]
            bboxes =  local_params #xywh
        
        elif self.bbox_regression_type == 'offset_based_on_anchor':
            layer_rects_expand = layer_rects.repeat(1, 9)
            layer_rects_expand = layer_rects_expand.reshape(-1, 4)
            anchor_box_wh_expand = anchor_box_wh.repeat(layer_rects_expand.shape[0] // 9, 1)
            anchor_bboxes = torch.cat((layer_rects_expand[:, 0 : 2] + layer_rects_expand[:, 2 : 4] * 0.5 - 0.5 * anchor_box_wh_expand, anchor_box_wh_expand), dim = 1)
            bboxes = anchor_bboxes + local_params.reshape(-1, 4)
            centroids = bboxes[:, 0 : 2] + 0.5 * bboxes[:, 2 : 4]
            confidence = confidence.reshape(-1)

        elif self.bbox_regression_type == 'anchor_encoding':
            layer_rects_expand = layer_rects.repeat(1, 9)
            layer_rects_expand = layer_rects_expand.reshape(-1, 4)
            anchor_box_wh_expand = anchor_box_wh.repeat(layer_rects_expand.shape[0] // 9, 1)
            local_params_reshape = local_params.reshape(-1, 4)
            # (bboxes_xy - layer_xy) / anchor_wh = local_params_xy
            bboxes_xy = local_params_reshape[:, 0 : 2] * anchor_box_wh_expand + layer_rects_expand[:, 0 : 2] + layer_rects_expand[:, 2 : 4] * 0.5 - 0.5 * anchor_box_wh_expand
            bboxes_wh = torch.exp(local_params_reshape[:, 2 : 4]) * anchor_box_wh_expand
            bboxes = torch.cat((bboxes_xy, bboxes_wh), dim = 1)
            centroids = bboxes[:, 0 : 2] + 0.5 * bboxes[:, 2 : 4]
            confidence = confidence.reshape(-1)
            
        elif self.bbox_regression_type == 'anchor_with_voting':
            layer_rects_expand = layer_rects.repeat(1, 9)
            layer_rects_expand = layer_rects_expand.reshape(-1, 4)
            anchor_box_wh_expand = anchor_box_wh.repeat(layer_rects_expand.shape[0] // 9, 1)
            anchor_bboxes = torch.cat((layer_rects_expand[:, 0 : 2] + layer_rects_expand[:, 2 : 4] * 0.5 - 0.5 * anchor_box_wh_expand, anchor_box_wh_expand), dim = 1)
            bboxes = anchor_bboxes + local_params.reshape(-1, 4)
            centroids = bboxes[:, 0 : 2] + 0.5 * bboxes[:, 2 : 4]
            confidence = confidence.reshape(-1)
            if voting_offset is not None:
                voting_offset = voting_offset.reshape(-1, 4) + anchor_bboxes
        else:
            raise(f"No such bbox regression type: {self.bbox_regression_type}")
        
        return (logits, centroids, bboxes, torch.nn.Sigmoid()(confidence) if confidence is not None else None, voting_offset)
        #return (logits, centroids, bboxes, confidence)
    def anchor_process(self, output):
        logits, centroids, bboxes, confidence, voting_offset = output
        confidence_reshape = confidence.reshape(-1, 9)
        confidence, confidence_argmax = torch.max(confidence_reshape, dim = 1)
        centroids = centroids.reshape(-1, 2 * 9)
        bboxes = bboxes.reshape(-1, 4 * 9)
        centroids = torch.cat([torch.gather(centroids, 1, confidence_argmax.unsqueeze(1) * 2 + i) for i in range(2)], dim = 1)
        bboxes = torch.cat([torch.gather(bboxes, 1, confidence_argmax.unsqueeze(1) * 4 + i) for i in range(4)], dim = 1)
        return (logits, centroids, bboxes, confidence, voting_offset)

    def contrasitive_loss(self, gnn_feats, layer_rects, bboxes, labels, node_indices):
        alpha = cfg.alpha
        gnn_feats = F.normalize(gnn_feats, p = 2, dim = 1)
        idx = 0
        prev_idx = 0
        contrasitive_loss = 0.0
        while idx < gnn_feats.shape[0]:
            if idx == node_indices.shape[0] - 1 or node_indices[idx + 1] != node_indices[idx]:
                x = gnn_feats[prev_idx : idx + 1, :]
                bboxes_idx = bboxes[prev_idx : idx + 1, :]
                layer_rects_idx = layer_rects[prev_idx : idx + 1, :]
                labels_idx = labels[prev_idx : idx + 1]
                bboxes_idx = bboxes_idx + layer_rects_idx
                similarity_matrix = torch.sum(torch.abs(bboxes_idx - bboxes_idx[:, None, :].repeat(1, idx + 1 - prev_idx, 1)), dim = 2) # size [N, N]
                mask = similarity_matrix <= 1e-6
                contrasitive_labels = torch.zeros_like(similarity_matrix, dtype = torch.float32, device = similarity_matrix.device)
                contrasitive_labels[mask] = 1.
                label_idx_mask_1 = (labels_idx == 0).reshape(idx + 1 - prev_idx, 1) & (labels_idx == 1).reshape(1, idx + 1 -prev_idx)
                label_idx_mask_2 = (labels_idx == 1).reshape(idx + 1 - prev_idx, 1) & (labels_idx == 0).reshape(1, idx + 1 -prev_idx)
                label_idx_mask = torch.logical_or(label_idx_mask_1, label_idx_mask_2)
                label_idx_mask_0 = (labels_idx == 0).reshape(idx + 1 - prev_idx, 1) & (labels_idx == 0).reshape(1, idx + 1 -prev_idx)
                contrasitive_labels[label_idx_mask] = 0. # we need make sure labels==0 is not related to labels == 1
                contrasitive_labels[label_idx_mask_0] = 1. # labels == 0 are considered to be related
                # contrasitive_labels[contrasitive_labels == 1] = 0.9
                # contrasitive_labels[contrasitive_labels == 0] = 0.1 / torch.sum(contrasitive_labels == 0)
                x = F.normalize(x, p = 2, dim = 1) 
                sim = x @ x.t()
                contrasitive_loss = contrasitive_loss + torch.nn.BCEWithLogitsLoss()(sim, contrasitive_labels)
                prev_idx = idx + 1
            idx = idx + 1
        return contrasitive_loss
    
    def loss(self, output, gt, anchor_box_wh = None):
        
        layer_rects, labels, bboxes = gt
        logits, centroids, local_params, confidence, voting_offset = self.process_output_data(output, layer_rects, anchor_box_wh)
        
        if self.loc_type == 'classifier_with_anchor':
            output = self.anchor_process((logits, centroids, local_params, confidence, voting_offset))
        else:
            output = (logits, centroids, local_params, confidence, voting_offset)
            
        loss_stats = {}
        cls_loss = self.cls_loss_fn(logits, labels)
        assert(torch.sum(bboxes[labels == 0]).item() < 1e-8)
        
        #scores, pred = torch.max(F.softmax(logits, dim = 1), 1)
        #training_mask = torch.logical_or(labels == 1, pred == 1)
        training_mask = (labels == 1)
        
        if torch.sum(training_mask) != 0:
            bboxes = bboxes[training_mask]
            layer_rects = layer_rects[training_mask]

        if 'anchor' in self.bbox_regression_type:
            training_mask = training_mask[:, None].repeat(1, 9).reshape(9 * training_mask.shape[0])

        if torch.sum(training_mask) != 0:
            local_params = local_params[training_mask]
            centroids = centroids[training_mask]
            if confidence is not None:
                confidence = confidence[training_mask]
            if voting_offset is not None:
                voting_offset = voting_offset[training_mask]
            #print(local_params.shape, layer_rects.shape)
            #if_pred_bbox_contain_layer = contains(local_params, layer_rects)
            #print(if_pred_bbox_contain_layer.all())
        #local_params = local_params + layer_rects

        bboxes = bboxes + layer_rects  # because in our dataset we store [bbox_gt - layer_rect]
        bboxes_center = bboxes[:, 0 : 2] + bboxes[:, 2 : 4] * 0.5
        if 'anchor' in self.bbox_regression_type:
            bboxes = bboxes.repeat(1, 9).reshape(-1, 4)
            bboxes_center = bboxes_center.repeat(1, 9).reshape(-1, 2)
        #print(local_params, bboxes)
        
        loss =  cfg.cls_loss.weight * cls_loss
        reg_loss = self.reg_loss_fn(local_params, bboxes)  #reg is short for regression
        offset_loss = None
        if voting_offset is not None:
            offset_loss = self.reg_loss_fn(voting_offset, bboxes)
            loss_stats['gnn_box_loss'] = offset_loss

        if confidence is not None:
            #pervious version of calculating confidence loss
            confidence_gt = IoU(bboxes, local_params)
            confidence_loss = cfg.confidence_weight * self.huber_fn(confidence_gt.detach(), confidence)
            '''confidence_gt = contains_how_much(local_params, bboxes)
            confidence_labels = (confidence_gt > 0.7).float()
            confidence_loss = F.binary_cross_entropy_with_logits(confidence, confidence_labels)'''
            loss_stats['confidence_loss'] = 1000 * confidence_loss
            center_reg_loss = torch.sum(torch.abs(bboxes_center - centroids), dim = 1)
            loss = loss + confidence_loss + cfg.reg_loss.weight * (1 - confidence_gt.detach()) * reg_loss + \
                        cfg.center_reg_loss.weight * (1 - confidence_gt.detach()) * center_reg_loss #+ 0.0 if offset_loss is None else cfg.reg_loss.weight * offset_loss
        else:
            center_reg_loss = self.huber_fn(bboxes_center, centroids)
            loss =  loss + cfg.reg_loss.weight * reg_loss + cfg.center_reg_loss.weight * center_reg_loss

        loss_stats['cls_loss'] = cls_loss
        loss_stats['reg_loss'] = reg_loss
        loss_stats['center_reg_loss'] = 1000 * center_reg_loss
        #loss = 10 * center_reg_loss + cfg.cls_loss.weight * cls_loss
        loss_stats['loss'] =  loss
        return loss, loss_stats, output

    def refine_box_loss(self, refine_boxes, gt):  
        layer_rects, labels, bboxes = gt
        training_mask = (labels == 1)
        
        if torch.sum(training_mask) != 0:
            bboxes = bboxes[training_mask]
            layer_rects = layer_rects[training_mask]
            refine_boxes = refine_boxes[training_mask]
        
        loss_stats = {}
        
        bboxes = bboxes + layer_rects # problems in development history: we actually store box_size - layer_rect in the dataset, so here we need to add the layer_rect back to get correct box_sizes
        loss = self.reg_loss_fn(refine_boxes, bboxes)
        loss_stats['refine_box_loss'] = loss
        return loss, loss_stats

    def prepare_img_tensors(self, device, file_list):
        img_tensor_list = []
        for path in file_list:
            dataset_dir, artboard_name = os.path.split(path)
            artboard_name = artboard_name.split(".")[0]
            if '-' not in artboard_name: # if artboard_name is artboard_id.
                # artboard_name = artboard_name.split("-")[0]
                jdx = 0
                while True:
                    img_tensor_path = os.path.join(dataset_dir, f'{artboard_name}-{jdx}.pt')
                    if not os.path.exists(img_tensor_path):
                        break
                    img_tensor_list.append(torch.load(img_tensor_path, map_location = device))
                    jdx += 1
            else: # if artboard_name is artboard_id-idx
                img_tensor_path = os.path.join(dataset_dir, f'{artboard_name}.pt')
                assert(os.path.exists(img_tensor_path))
                img_tensor_list.append(torch.load(img_tensor_path, map_location = device))
        return torch.vstack(img_tensor_list)

    def forward(self, batch, anchor_box_wh = None):
        # nodes, edges, types,  img_tensor, labels, bboxes, node_indices, file_list
        layer_rect, edges, classes, images, labels, bboxes, _, node_indices, file_list = batch

        batch_embedding = 0.0
        if not cfg.remove_pos:
            pos_embedding = self.pos_embedder(layer_rect)
            batch_embedding = batch_embedding + pos_embedding
        if not cfg.remove_type:
            type_embedding = self.type_embedder(classes)
            batch_embedding = batch_embedding + type_embedding
        if not cfg.remove_img:
            img_embedding = self.img_embedder(images)
            batch_embedding = batch_embedding + img_embedding
        # print(pos_embedding.shape, type_embedding.shape, img_embedding.shape)
        # batch_embedding = self.pos_embedder(layer_rect)+self.type_embedder(classes)+self.img_embedder(images)
        # batch_embedding = pos_embedding + type_embedding + img_embedding
        # batch_embedding = type_embedding + img_embedding

        edge_attr = None
        if cfg.gnn_fn.local_gnn_type == 'GINEConv' or cfg.gnn_fn.local_gnn_type == 'GATConv' or cfg.gnn_fn.local_gnn_type == 'PNAConv':
            x_i = edges[0, :]
            x_j = edges[1, :]
            #edge_attr = pos_embedding[x_i, :] - pos_embedding[x_j, :] + img_embedding[x_i, :] - img_embedding[x_j, :]
            #edge_attr = self.pos_embedder.embed(torch.abs(layer_rect[x_i, :] - layer_rect[x_j, :]))
            edge_attr = self.edge_embedder(torch.abs(layer_rect[x_i, :] - layer_rect[x_j, :]))
        
        confidence = None
        voting_offset = None
        
        gnn_out = self.gnn_fn((batch_embedding, edges, node_indices), edge_attr, layer_rect)
        
        if 'anchor_voting' in self.gnn_type:
            gnn_out, voting_offset = gnn_out #voting_offset: [N, 4 * 9]
            
        logits = self.cls_fn(gnn_out)
        
        if 'voting' in self.gnn_type and 'anchor' not in self.gnn_type:
            gnn_out = gnn_out[:, 4:]
            voting_offset = gnn_out[:, :4] 

        contrasitive_loss = self.contrasitive_loss(gnn_out, layer_rect, bboxes, labels, node_indices)
        
        if self.loc_type == 'classifier_with_gnn':
            loc_params = self.loc_fn(gnn_out, edges, node_indices)

        elif self.loc_type == 'classifier_with_confidence':
            out = self.loc_fn(gnn_out)
            assert(out.size(-1) == 5)
            loc_params = out[:, :4]
            confidence = out[:, 4]
            
        elif self.loc_type == 'classifier_with_anchor':
            out = self.loc_fn(gnn_out)
            loc_params = out[:, :36]
            confidence = out[:, 36:]
        else:
            loc_params = self.loc_fn(gnn_out)

        if 'voting' in self.gnn_type: 
            loc_params = voting_offset + loc_params

        #print(logits.shape, loc_params.shape)
        if self.refine_box_module is None:
            loss, loss_stats, output = self.loss([logits, loc_params, confidence, voting_offset], 
                                        [layer_rect, labels, bboxes], anchor_box_wh)
            loss_stats['contrasitvie_loss'] = contrasitive_loss
            loss += contrasitive_loss
            # if self.loc_type == 'classifier_with_anchor':
            #    return self.anchor_process(self.process_output_data((logits, loc_params, confidence, voting_offset), layer_rect, anchor_box_wh)), loss, loss_stats
            #return self.process_output_data((logits, loc_params, confidence, voting_offset), layer_rect, anchor_box_wh), loss, loss_stats
            return output, loss, loss_stats
        else:
            logits, centroids, pred_bboxes, confidence, voting_offset = self.anchor_process(
                self.process_output_data((logits, loc_params, confidence, voting_offset),  layer_rect, anchor_box_wh))
            img_tensors = self.prepare_img_tensors(bboxes.device, file_list)
            # print(img_tensors.requires_grad)
            # print(img_tensors.shape, node_indices.shape, bboxes.shape)
            refine_bboxes = self.refine_box_module(img_tensors, node_indices[labels == 1], pred_bboxes[labels == 1])
            # torch.masked_fill(pred_bboxes, labels == 1, refine_bboxes)
            pred_bboxes[labels == 1, :] = refine_bboxes
            loss, loss_stats = self.refine_box_loss(pred_bboxes, [layer_rect, labels, bboxes])
            return (logits, centroids, pred_bboxes, confidence, voting_offset), loss, loss_stats
            
if __name__=='__main__':
    network = Network()
    print(network)

