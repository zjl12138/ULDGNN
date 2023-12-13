from lib.datasets import make_data_loader
from lib.config import cfg
from lib.visualizers import comp_det_visualizer
from lib.networks import make_network
import torch
from lib.train import make_optimizer, make_recorder, make_scheduler
from lib.train import make_trainer
from lib.evaluators.comp_det_eval import Evaluator
from lib.utils import load_model, save_model, load_network
import os
from PIL import Image
import json
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision

class data_loader:
    def __init__(self):
        
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def read_img(self, img_path):
        layer_assets = self.img_transform(Image.open(img_path).convert('RGB'))
        layer_assets = list(torch.split(layer_assets, 64, dim = 1))
        return torch.stack(layer_assets)

    def read_json(self, json_path):
        # content['layer_rect'],content['edges'], content['bbox'], content['labels']
        with open(json_path, 'r') as f:
            content = json.load(f)
        return content['layer_rect'], content['edges'], content['bbox'], content['ids']

    def load(self, assets_img_path, graph_data_path):
        img_path = assets_img_path
        assets_img = self.read_img(img_path)
        json_path = graph_data_path
        layer_rect, edges, bbox, ids= self.read_json(json_path)
        layer_rect = torch.FloatTensor(layer_rect) 
        edges = torch.LongTensor(edges).transpose(1,0)
        bbox = torch.FloatTensor(bbox)
        node_indices = torch.zeros(layer_rect.shape[0], dtype = torch.int64)
        return [assets_img, layer_rect, edges, bbox, node_indices], ids

def to_cuda(batch, device):
    for k, _ in enumerate(batch):
        if isinstance(batch[k], tuple) or isinstance(batch[k], list):
            batch[k] = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch[k]]
        else:
            batch[k] = batch[k].to(device) if isinstance( batch[k], torch.Tensor) else batch[k]
    
    return batch 

def test(cfg, network, artboard_name, assets_img_path, graph_data_path, artboard_img_path):
    optimizer = make_optimizer(cfg, network)
    scheduler = make_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg.recorder)
    begin_epoch = load_network(network, 
                            cfg.model_dir)
    loader = data_loader()
    
    device = torch.device(f'cuda:{cfg.train.local_rank}')
    network = network.to(device)
    network.eval()
    data, ids = loader.load(assets_img_path, graph_data_path)
    data = to_cuda(data, device)
    assets_img, layer_rect, edges, bbox, node_indices = data

    output = network(data)
    logits, pred_bboxes, = output #output: (logits, bboxes)

    _, label_pred = torch.max(F.softmax(logits,dim=1), 1)

    vis = comp_det_visualizer(cfg.visualizer)
    img_tensor, id_to_labelstr = vis.visualize_layer_inference(layer_rect, label_pred, ids, artboard_img_path)
    torchvision.utils.save_image(img_tensor, f"/media/sda1/ljz-workspace/code/ui_to_code/materials/{artboard_name}/comp_det_inference.png")
    json.dump(id_to_labelstr, open(f"/media/sda1/ljz-workspace/code/ui_to_code/materials/{artboard_name}/comp_det_results.json", 'w'), indent=4)

if __name__=='__main__':
   
    #cfg.test.vis_bbox = True
    cfg.mode = "test"
    cfg.train.is_distributed = False
    cfg.train.local_rank = 0
    cfg.test.vis_bbox = True
    cfg.test.eval_merge = False
    cfg.test.eval_ap = False
    cfg.test.val_nms = False
    # cfg.test_dataset.rootDir = '../../dataset/EGFE_graph_dataset_refine'
    # cfg.test_dataset.index_json = 'index_testv2.json'
    # cfg.test_dataset.bg_color_mode = 'bg_color_orig'
    print(cfg.test_dataset.index_json)
    print(cfg.test_dataset.rootDir)
    network = make_network(cfg.network)
    # begin_epoch = load_network(network, cfg.model_dir, map_location = f'cuda:{cfg.train.local_rank}')
    # network.begi n_update_edge_attr()
    # print("begin epoch: ", begin_epoch)
    artboard_name = 74
    assets_img_path = f"/media/sda1/ljz-workspace/code/ui_to_code/materials/{artboard_name}/comp_det_data/{artboard_name}-assets.png"
    graph_data_path = f"/media/sda1/ljz-workspace/code/ui_to_code/materials/{artboard_name}/comp_det_data/{artboard_name}.json"
    artboard_img_path = f"/media/sda1/ljz-workspace/code/ui_to_code/materials/{artboard_name}/artboard.png"
    test(cfg, network, artboard_name, assets_img_path, graph_data_path, artboard_img_path)