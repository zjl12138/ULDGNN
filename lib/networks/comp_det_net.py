import torch
import torch.nn as nn
import torch_geometric
from lib.networks.network import make_gnn
from lib.config import cfg as CFG
from lib.networks.loss import make_classifier_loss, make_regression_loss
from lib.networks.layers import PosEmbedder, ImageEmbedder, GetPosEmbedder, ImgFeatRoiExtractor
from lib.networks.network import Classifier

cfg = CFG.network

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.cls_weights = torch.FloatTensor([4.6026e-04, 1.0917e-03, 3.0843e-04, 6.9754e-04, 1.8872e-04, 5.0046e-03,
                        6.4896e-03, 1.7365e-01, 1.4079e-02, 2.1542e-02, 1.3265e-02, 6.3670e-02,
                        2.6286e-02, 2.0687e-02, 7.7436e-02, 1.4326e-01, 1.4923e-02, 1.2906e-02,
                        2.1200e-03, 5.9179e-04, 1.8020e-02, 2.6179e-04, 2.7287e-01, 1.1020e-01]).to(torch.device(f'cuda:{CFG.train.local_rank}'))
        self.cls_loss_fn = make_classifier_loss(cfg.cls_loss, weight = self.cls_weights)
        self.reg_loss_fn = make_regression_loss(cfg.reg_loss)
        
        cfg.gnn_fn.in_dim = cfg.pos_embedder.out_dim
        cfg.cls_fn.in_dim = cfg.gnn_fn.out_dim
        cfg.loc_fn.in_dim = cfg.gnn_fn.out_dim
        
        self.pos_embedder = PosEmbedder(cfg.pos_embedder)
        print("Layer embedder way: ", cfg.img_embedder.type)
        self.img_embedder_type = cfg.img_embedder.type
        if cfg.img_embedder.type != 'roi':
            self.img_embedder = ImageEmbedder(cfg.img_embedder)
        else:
            self.img_embedder = ImgFeatRoiExtractor(cfg.img_embedder)

        self.edge_embedder, self.edge_dim = GetPosEmbedder(cfg.edge_embedder.multires, include_input = True)
        
        print("edge_dim is ", self.edge_dim)
        if self.edge_dim != cfg.pos_embedder.out_dim:
            cfg.gnn_fn.edge_dim = self.edge_dim
        else:
            cfg.gnn_fn.edge_dim = 0
        
        self.gnn_fn = make_gnn(cfg.gnn_fn.gnn_type)(cfg.gnn_fn)
        
        self.cls_fn = Classifier(cfg.cls_fn)
        self.loc_fn = Classifier(cfg.loc_fn)
        self.fix_network(self.loc_fn)

    def fix_network(self, model):
        for n, params in model.named_parameters():
            params.requires_grad = False

    def loss(self, output, gt):
        labels, bboxes = gt
        logits, local_params = output
        loss_stats = {}
        cls_loss = self.cls_loss_fn(logits, labels)
        
        loss =  cfg.cls_loss.weight * cls_loss
        reg_loss = self.reg_loss_fn(local_params, bboxes, box_format = 'xyxy')  # reg is short for regression
        loss =  loss + cfg.reg_loss.weight * reg_loss
        loss_stats['cls_loss'] = cls_loss
        loss_stats['reg_loss'] = reg_loss
        loss_stats['total_loss'] = loss
        return loss, loss_stats
     
    def forward(self, batch):
        images, layer_rect, edges, bbox, labels, node_indices, _ = batch
        batch_embedding = 0.0
        
        pos_embedding = self.pos_embedder(layer_rect)
        batch_embedding = batch_embedding + pos_embedding
    
        if self.img_embedder_type != 'roi':
            img_embedding = self.img_embedder(images)
        else:
            img_embedding = self.img_embedder(images, torch.cat((node_indices.unsqueeze(1), layer_rect), dim = 1))
        
        batch_embedding = batch_embedding + img_embedding
    
        edge_attr = None
        if cfg.gnn_fn.local_gnn_type == 'GINEConv' or cfg.gnn_fn.local_gnn_type == 'GATConv' or cfg.gnn_fn.local_gnn_type == 'PNAConv':
            x_i = edges[0, :]
            x_j = edges[1, :]
            #edge_attr = pos_embedding[x_i, :] - pos_embedding[x_j, :] + img_embedding[x_i, :] - img_embedding[x_j, :]
            #edge_attr = self.pos_embedder.embed(torch.abs(layer_rect[x_i, :] - layer_rect[x_j, :]))
            edge_attr = self.edge_embedder(torch.abs(layer_rect[x_i, :] - layer_rect[x_j, :]))
        
        gnn_out = self.gnn_fn((batch_embedding, edges, node_indices), edge_attr, layer_rect)
        
        logits = self.cls_fn(gnn_out)
        pred_bboxes = self.loc_fn(gnn_out)
        
        loss, loss_stats = self.loss((logits, pred_bboxes), (labels, bbox))
        return (logits, pred_bboxes), loss, loss_stats
        
        
        
        