import torch
import torch.nn as nn
import torch_geometric
from lib.networks.layers import make_fully_connected_layer, GATLayer, GPSLayer
from lib.networks.layers import GetPosEmbedder, PosEmbedder, ImageEmbedder, TypeEmbedder
from lib.networks.loss import make_classifier_loss, make_regression_loss
from lib.config import cfg as CFG
import importlib
import torch.nn.functional as F
from lib.utils import vote_clustering, vote_clustering_each_layer
cfg = CFG.network

def make_gnn(gnn_type):
    norm_module = importlib.import_module("lib.networks.network")
    return getattr(norm_module,gnn_type)

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
                    cfg.batch_norm
                )
            )
            in_dim = latent_dim
        
    def forward(self, batch):
        x, edge_index, node_indices = batch
        #print(torch.max(edges))
        #print(x.shape)
        x_in = x
        for _, gnn_layer in self.named_children():
            x =  gnn_layer((x, edge_index, node_indices))
        return x

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
                    cfg.batch_norm
                )
            )
            in_dim = latent_dim
        
    def forward(self, batch): # 
        x, edge_index, node_indices = batch
        #print(torch.max(edges))
        #print(x.shape)
        padding_zeros = torch.zeros((x.shape[0], 4), device = x.get_device())
        #x = torch.cat((padding_zeros, x), dim = 1)
        for idx, (name, gnn_layer) in enumerate(self.named_children()):
            if 'GPS' in name:
                x = gnn_layer((x, edge_index, node_indices))
                padding_zeros = padding_zeros + self.offset_head(x)
        x = torch.cat((padding_zeros, x), dim = 1)
        del padding_zeros
        return x 

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()   
        
        cfg.gnn_fn.in_dim = cfg.pos_embedder.out_dim
        cfg.cls_fn.in_dim = cfg.gnn_fn.out_dim
        cfg.loc_fn.in_dim = cfg.gnn_fn.out_dim

        if  cfg.gnn_fn.gnn_type == 'GPSModel_with_voting':
            #cfg.pos_embedder.out_dim = cfg.pos_embedder.out_dim - 4
            #cfg.type_embedder.out_dim = cfg.type_embedder.out_dim - 4
            #cfg.img_embedder.out_dim  = cfg.img_embedder.out_dim - 4
            #cfg.loc_fn.in_dim = cfg.gnn_fn.out_dim - 4
            cfg.cls_fn.in_dim = cfg.gnn_fn.out_dim + 4
            print(cfg.pos_embedder.out_dim)

        self.bbox_regression_type = cfg.bbox_regression_type
        self.bbox_vote_radius = cfg.bbox_vote_radius   #used for vote clustering
        print("the bbox regression type: ", self.bbox_regression_type)

        self.pos_embedder = PosEmbedder(cfg.pos_embedder)
        self.type_embedder = TypeEmbedder(cfg.type_embedder)
        self.img_embedder = ImageEmbedder(cfg.img_embedder)

        self.gnn_fn = make_gnn(cfg.gnn_fn.gnn_type)(cfg.gnn_fn)
        self.cls_fn = Classifier(cfg.cls_fn)
        self.loc_type = cfg.loc_fn.loc_type
        self.gnn_type = cfg.gnn_fn.gnn_type
        print("loc_fn_type: ", cfg.loc_fn.loc_type)
        if cfg.loc_fn.loc_type == 'classifier_with_gnn':
            self.loc_fn = Classifier_with_gnn(cfg.loc_fn) 
        else:
            self.loc_fn = Classifier(cfg.loc_fn)
        if cfg.train_mode == 3:
            print("fix cls branch!")
            self.fix_network(self.cls_fn)
        elif cfg.train_mode == 2:
            print("train localizaiton branch only!")
            self.fix_network(self.pos_embedder)
            self.fix_network(self.type_embedder)
            self.fix_network(self.img_embedder)
            self.fix_network(self.gnn_fn)
            self.fix_network(self.cls_fn)
        elif cfg.train_mode == 1:
            print("fix localization branch!")
            self.fix_network(self.loc_fn)
        else:
            print("train two branches together")
        
    def fix_network(self, model):
        for n, params in model.named_parameters():
            params.requires_grad = False

    def process_output_data(self, output, layer_rects):
        logits, local_params = output
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
        else:
            raise(f"No such bbox regression type: {self.bbox_regression_type}")
        return (logits, centroids, bboxes)

    def loss(self, output, gt):
        
        layer_rects, labels, bboxes = gt
        logits, centroids, local_params = self.process_output_data(output, layer_rects)

        cls_loss_fn = make_classifier_loss(cfg.cls_loss)
        reg_loss_fn = make_regression_loss(cfg.reg_loss)
        huber_fn = torch.nn.L1Loss() 
        loss_stats = {}
        cls_loss = cls_loss_fn(logits, labels)
        assert(torch.sum(bboxes[labels == 0]).item() < 1e-8)
        
        scores, pred = torch.max(F.softmax(logits, dim = 1), 1)
        #training_mask = torch.logical_or(labels == 1, pred == 1)
        '''training_mask = (labels == 1)
        local_params = local_params[training_mask]
        bboxes = bboxes[training_mask]
        layer_rects = layer_rects[training_mask]
        centroids = centroids[training_mask]
        #local_params = local_params + layer_rects'''
        bboxes = bboxes + layer_rects
        bboxes_center = bboxes[:, 0 : 2] + bboxes[:, 2 : 4] * 0.5
        center_reg_loss = huber_fn(bboxes_center, centroids)
        #print(local_params, bboxes)
        if local_params.shape[0] == 0:
            reg_loss = torch.tensor(0., requires_grad = True).to(local_params.get_device())
        else:
            reg_loss = reg_loss_fn(local_params, bboxes)
        
        loss_stats['cls_loss'] = cls_loss
        loss_stats['reg_loss'] = reg_loss
        loss_stats['center_reg_loss'] = 1000 * center_reg_loss
        loss =  cfg.cls_loss.weight * cls_loss \
                    + cfg.reg_loss.weight * reg_loss #+ 100 * center_reg_loss
        #loss = 10 * center_reg_loss + cfg.cls_loss.weight * cls_loss
        loss_stats['loss'] =  loss
        return loss, loss_stats

    def forward(self, batch):
        #nodes, edges, types,  img_tensor, labels, bboxes, node_indices, file_list
        layer_rect, edges, classes, images, labels, bboxes, node_indices, _ = batch
        pos_embedding = self.pos_embedder(layer_rect)
        type_embedding = self.type_embedder(classes)
        img_embedding = self.img_embedder(images)
        #print(pos_embedding.shape, type_embedding.shape, img_embedding.shape)
        #batch_embedding = self.pos_embedder(layer_rect)+self.type_embedder(classes)+self.img_embedder(images)
        batch_embedding = pos_embedding + type_embedding + img_embedding
        gnn_out = self.gnn_fn((batch_embedding, edges, node_indices))
        logits = self.cls_fn(gnn_out)

        if self.gnn_type == 'GPSModel_with_voting':
            gnn_out = gnn_out[:, 4:]
            center_offset = torch.nn.Tanh()(gnn_out[:, :4])

        if self.loc_type == 'classifier_with_gnn':
            loc_params = self.loc_fn(gnn_out, edges, node_indices)
        else:
            loc_params = self.loc_fn(gnn_out)

        if self.gnn_type == 'GPSModel_with_voting': 
            loc_params = center_offset + loc_params
        #print(logits.shape, loc_params.shape)
        loss, loss_stats = self.loss([logits, loc_params], [layer_rect, labels, bboxes])
        return self.process_output_data((logits, loc_params), layer_rect), loss, loss_stats

if __name__=='__main__':
    network = Network()
    print(network)

