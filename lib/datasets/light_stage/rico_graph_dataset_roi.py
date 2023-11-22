import json
import torch
import numpy as np
from torch.utils import data
import os
import torchvision.transforms as T
from PIL import Image

class Dataset(data.Dataset):
    def __init__(self, cfg):
        print("[MESSAGE] rico_graph dataset init!")
        self.root = cfg.rootDir
        self.index_json = cfg.index_json
        self.train_list = json.load(open(os.path.join(self.root, self.index_json),'r'))
        #self.train_list = self.train_list[:20]
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.artboard_img_root = "/media/sda2/ljz_dataset/combined"
    
    def __len__(self):
        return len(self.train_list)

    def read_img(self, img_path):
        artboard_img = self.img_transform(Image.open(img_path).convert('RGB').resize((270, 480)))
        return artboard_img.unsqueeze(0) # [1, 3, 960, 540]

    def read_json(self, json_path):
        # content['layer_rect'],content['edges'], content['bbox'], content['labels']
        with open(json_path, 'r') as f:
            content = json.load(f)
        return content['layer_rect'],content['edges'], content['bbox'], content['labels']

    def collate_fn(self, batch):
        # assets_img, layer_rect, edges, bbox, labels, node_indices, artboard_id
        img_list = [b[0]for b in batch]
        img_tensor = torch.cat(img_list, dim = 0)
        
        nodes_list = [b[1] for b in batch]
        nodes = torch.cat(nodes_list, dim=0)
        
        nodes_lens = np.fromiter(map(lambda l: l.shape[0], nodes_list), dtype=np.int64)
        nodes_inds = np.cumsum(nodes_lens)
        nodes_num = nodes_inds[-1]
        nodes_inds = np.insert(nodes_inds, 0, 0)
        nodes_inds = np.delete(nodes_inds, -1)
        edges_list = [b[2] for b in batch]
        edges_list = [e+int(i) for e, i in zip(edges_list, nodes_inds)]
        edges = torch.cat(edges_list, dim=1)
    
        bboxes = [b[3]for b in batch]
        bboxes = torch.cat(bboxes,dim=0)
        
        labels = [b[4] for b in batch]
        labels = torch.cat(labels, dim=0)
        
        node_ind_list = [b[5] for b in batch]
        #node_ind_list = [cur + torch.max(i)for prev, cur in zip(node_ind_list[:-1],node_ind_list[1:])]
        graph_lens = np.fromiter(map(lambda l: l[-1]+1, node_ind_list), dtype=np.int64)
        graph_inds = np.cumsum(graph_lens)
        
        graph_inds = np.insert(graph_inds, 0, 0)
        graph_inds = np.delete(graph_inds, -1)
        
        node_ind_list = [e + int(i) for e, i in zip(node_ind_list, graph_inds)]
        node_indices = torch.cat(node_ind_list, dim=0)
        file_list = [b[6] for b in batch]
            
        return img_tensor, nodes, edges, bboxes, labels, node_indices, file_list
    
    def __getitem__(self, index):
        artboard_id = self.train_list[index]
        img_path = os.path.join(self.artboard_img_root, f'{artboard_id}.jpg')
        assets_img = self.read_img(img_path)
        json_path = os.path.join(self.root, str(artboard_id), f'{artboard_id}.json')
        layer_rect, edges, bbox, labels = self.read_json(json_path)
        layer_rect = torch.FloatTensor(layer_rect) 
        edges = torch.LongTensor(edges).transpose(1,0)
        bbox = torch.FloatTensor(bbox)
        labels = torch.LongTensor(labels)
        node_indices = torch.zeros(layer_rect.shape[0], dtype = torch.int64)
        
        return assets_img, layer_rect, edges, bbox, labels, node_indices, artboard_id
        
