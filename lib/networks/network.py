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
        layer_list = []
        in_dim = self.in_dim
        
        for idx, latent_dim in enumerate(self.latent_dims):
            self.add_module(f'fc_{idx}',make_fully_connected_layer(in_dim,
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

class LayerGNN(nn.Module):
    def __init__(self, cfg):
        super(LayerGNN,self).__init__()
        self.latent_dims = cfg.latent_dims
        self.num_heads = cfg.num_heads
        self.make(cfg)
        
    def make(self, cfg):
        gnn_layer_list = []
        in_dim = cfg.in_dim
        L = len(self.latent_dims)+1
        assert(len(self.num_heads)==len(self.latent_dims)+1)
        for idx, (latent_dim, num_head) in enumerate(zip(self.latent_dims, self.num_heads[:-1])):
            self.add_module(f'gatlayer_{idx}', GATLayer(in_dim, latent_dim,
                                                 num_head, cfg.batch_norm, 
                                                 cfg.residual, concat = cfg.concat))
            in_dim = latent_dim * num_head if cfg.concat else latent_dim
        self.add_module(f'gatlayer_{L+1}',GATLayer(in_dim, cfg.out_dim, self.num_heads[-1], False, concat=False))
        
    def forward(self, x, edges):
        #print(torch.max(edges))
        #print(x.shape)
        for _, gnn_layer in self.named_children():
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
        layer_rects, labels, bboxes = gt
        cls_loss_fn = make_classifier_loss(cfg.cls_loss)
        reg_loss_fn = make_regression_loss(cfg.reg_loss)
        loss_stats = {}
        cls_loss = cls_loss_fn(logits, labels)
        
        local_params = local_params[labels==1]
        bboxes = bboxes[labels==1]
        layer_rects =  layer_rects[labels==1]

        local_params = local_params + layer_rects
        bboxes = bboxes + layer_rects
        
        #print(local_params, bboxes)
        if local_params.shape[0]==0:
            reg_loss = torch.tensor(0.).to(local_params.get_device())
        else:
            reg_loss = reg_loss_fn(local_params, bboxes)
        
        loss_stats['cls_loss'] = cls_loss
        loss_stats['reg_loss'] = reg_loss
        loss =  cfg.cls_loss.weight * cls_loss \
                    + cfg.reg_loss.weight * reg_loss
        
        loss_stats['loss'] = loss
        return loss, loss_stats

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

        loss, loss_stats = self.loss([logits, loc_params],[layer_rect, labels, bboxes])
        return (logits, loc_params), loss, loss_stats

if __name__=='__main__':
    network = Network()
    print(network)

