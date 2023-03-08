import torch
import torch.nn as nn
from lib.networks.layers import make_fully_connected_layer, GATLayer
from lib.networks.layers import GetPosEmbedder, PosEmbedder, ImageEmbedder, TypeEmbedder
from lib.networks.loss import make_classifier_loss, make_regression_loss
from lib.config import cfg as CFG

cfg = CFG.network

class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.in_dim = cfg.in_dim
        self.latent_dims = cfg.latent_dims
        self.make(cfg)

    def make(self, cfg):
        self.layer_list = []
        in_dim = self.in_dim
        for latent_dim in self.latent_dims:
            self.layer_list.append(make_fully_connected_layer(in_dim,
                                                             latent_dim, 
                                                             cfg.act_fn, 
                                                             cfg.norm_type)
                                  )
            in_dim = latent_dim

        self.layer_list.append(make_fully_connected_layer(self.latent_dims[-1],
                                                          cfg.classes,
                                                          None, None)
                              )
    def forward(self, x):
        for fc_layer in self.layer_list:
            x = fc_layer(x)
        return x

class LayerGNN(nn.Module):
    def __init__(self, cfg):
        super(LayerGNN,self).__init__()
        self.latent_dims = cfg.latent_dims
        self.num_heads = cfg.num_heads
        self.gnn_layer_list = []
        self.make(cfg)

    def make(self, cfg):
        
        in_dim = cfg.in_dim
        assert(len(self.num_heads)==len(self.latent_dims)+1)
        for (latent_dim, num_head) in zip(self.latent_dims, self.num_heads[:-1]):
            self.gnn_layer_list.append(GATLayer(in_dim, latent_dim,
                                                 num_head, cfg.batch_norm, 
                                                 cfg.residual, concat = cfg.concat))
            in_dim = latent_dim * num_head if cfg.concat else latent_dim
        self.gnn_layer_list.append(GATLayer(in_dim, cfg.out_dim, self.num_heads[-1], False, concat=False))
        
    def forward(self, x, edges):
        #print(torch.max(edges))
        #print(x.shape)
        for gnn_layer in self.gnn_layer_list:
            x = gnn_layer(x, edges)
        return x

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
       
        self.pos_embedder = PosEmbedder(cfg.pos_embedder)
        self.type_embedder = TypeEmbedder(cfg.type_embedder)
        self.img_embedder = ImageEmbedder(cfg.img_embedder)
        cfg.gnn_fn.in_dim = cfg.pos_embedder.out_dim
        cfg.cls_fn.in_dim = cfg.gnn_fn.out_dim
        cfg.loc_fn.in_dim = cfg.gnn_fn.out_dim
        self.gnn_fn = LayerGNN(cfg.gnn_fn)
        self.cls_fn = Classifier(cfg.cls_fn)
        self.loc_fn = Classifier(cfg.loc_fn)

    def loss(self, output, gt):
        logits, local_params = output
        labels, bboxes = gt
        cls_loss_fn = make_classifier_loss(cfg.cls_loss)
        reg_loss_fn = make_regression_loss(cfg.reg_loss)
        loss_stats = {}
        cls_loss = cls_loss_fn(logits, labels)
        reg_loss = reg_loss_fn(local_params, bboxes)
        loss_stats['cls_loss'] = cls_loss
        loss_stats['reg_loss'] = reg_loss
        loss_stats['loss'] = cfg.cls_loss.weight * cls_loss \
                                + cfg.reg_loss.weight * reg_loss 
        return loss_stats

    def forward(self, batch):
        #nodes, edges, types,  img_tensor, labels, bboxes, file_list
        layer_rect, edges, classes, images, labels, bboxes, _ = batch
        pos_embedding = self.pos_embedder(layer_rect)
        type_embedding = self.type_embedder(classes)
        img_embedding = self.img_embedder(images)

        #batch_embedding = self.pos_embedder(layer_rect)+self.type_embedder(classes)+self.img_embedder(images)
        batch_embedding = pos_embedding + type_embedding + img_embedding
        gnn_out = self.gnn_fn(batch_embedding, edges)
        logits = self.cls_fn(gnn_out)
        loc_params = self.loc_fn(gnn_out)
        #print(logits.shape, loc_params.shape)

        loss_stats = self.loss([logits, loc_params],[labels,bboxes])
        return (logits, loc_params), loss_stats['loss'], loss_stats



