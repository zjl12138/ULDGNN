
from torch.utils import data
import json
import os
import glob
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
from lib.config.yacs import CfgNode as CN

def collate_fn(batch, collate_img_path=True):
    #layer_rect, edges, types, layer_img, labels, bbox
    nodes_list = [b[0] for b in batch]
    nodes = torch.cat(nodes_list, dim=0)
    nodes_lens = np.fromiter(map(lambda l: l.shape[0], nodes_list), dtype=np.int64)
    nodes_inds = np.cumsum(nodes_lens)
    nodes_num = nodes_inds[-1]
    nodes_inds = np.insert(nodes_inds, 0, 0)
    nodes_inds = np.delete(nodes_inds, -1)
    edges_list = [b[1] for b in batch]
    print(nodes_inds)
    edges_list = [e+int(i) for e, i in zip(edges_list, nodes_inds)]
    edges = torch.cat(edges_list, dim=1)
    types = [b[2] for b in batch]
    types = torch.cat(types, dim=0)
    labels = [b[4] for b in batch]
    labels = torch.cat(labels, dim=0)
    bboxes = [b[5]for b in batch]
    bboxes = torch.cat(bboxes,dim=0)
    img_list = [b[3]for b in batch]
    img_tensor = torch.cat(img_list, dim=0)
    
    if collate_img_path:
        file_list = [b[6] for b in batch]
        return  nodes, edges, types,  img_tensor, labels, bboxes, file_list   
    return nodes, edges, types,  img_tensor, labels, bboxes

def read_graph_json(path):
    content = json.load(open(path,"r"))
    return content['layer_rect'],content['edges'], content['bbox'],content['types'],content['labels']

class Dataset(data.Dataset):
    def __init__(self, cfg):
        self.root = cfg.rootDir
        self.index_json = cfg.index_json
        self.train_list = json.load(open(os.path.join(self.root,self.index_json),'r'))
       
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def __len__(self):
        return len(self.train_list)

    def read_img(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, index) :
        
        train_artboard = self.train_list[index]
        json_name, assets_img = train_artboard['json'], train_artboard['layerassets']
        
        artboard_idx = json_name.split(".")[0]
        graphs_in_this_arboard = glob.glob(os.path.join(self.root, artboard_idx)+"/*.json")
        layer_assets = self.img_transform(self.read_img(os.path.join(self.root, artboard_idx,assets_img)))
        layer_assets = list(torch.split(layer_assets,64,dim=1))
        
        artboard_img_path = os.path.join(self.root, artboard_idx, train_artboard['image'])

        batch = []
        img_split_last_idx = 0
        nodes = 0
        for idx, graph_path in enumerate(graphs_in_this_arboard):
            #content['layer_rect'],content['edges'], content['bbox'],content['types'],content['labels']

            layer_rect, edges, bbox, types, labels = read_graph_json(graph_path)
            layer_rect = torch.FloatTensor(layer_rect)
            edges = torch.LongTensor(edges).transpose(1,0)
            
            bbox = torch.FloatTensor(bbox)
            types = torch.LongTensor(types)
            labels = torch.LongTensor(labels)
            #print(layer_rect.shape[0])
            nodes += layer_rect.shape[0]
            layer_img = torch.stack(layer_assets[img_split_last_idx:img_split_last_idx+layer_rect.shape[0]])
            #nodes, edges, types,  img_tensor, labels, bboxes, file_list
            batch.append([layer_rect, edges, types, layer_img, labels, bbox])
            img_split_last_idx += layer_rect.shape[0]
        
        return *collate_fn(batch, False), artboard_img_path

if __name__=='__main__':
    cfg = CN()
    cfg.rootDir='/media/sda1/ljz-workspace/dataset/graph_dataset'
    cfg.train_json = 'index_train.json'
    dataset = Dataset(cfg)
    
    for i, batch in enumerate(dataset):
        layer_rects, collate_edges, collate_types, collate_labls,bboxes, layer_assets=batch
        print(layer_rects.shape)

