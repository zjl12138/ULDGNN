
from typing import List
from torch.utils import data
import json
import os
import glob
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
from torchvision.transforms.transforms import ToTensor
from lib.config.yacs import CfgNode as CN
import colorsys
from torchvision.utils import save_image
from lib.utils import get_the_bbox_of_cluster

def read_graph_json(path): 
    '''
    output: bbox stores (delta_x, delta_y, delta_w, delta_h), bbox + layer_rect is the real bbox size
    '''
    content = json.load(open(path,"r"))
    return content['layer_rect'],content['edges'], content['bbox'],content['types'],content['labels']

def find_contrast_color(rgb):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h = (h + 0.5) % 1.0
    #s = 1.0 - s   # 或者使用 
    s = max(s, 0.5)
    v = max(v, 0.5)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return r , g , b

def freq_max(arr):
    modes = []
    for i in range(arr.shape[1]):
        values, counts = np.unique(arr[:,i], return_counts=True)
        modes.append(values[np.argmax(counts)])
    return modes

class Dataset(data.Dataset):
    def __init__(self, cfg):
        print("using small batch datset!")
        self.root = cfg.rootDir
        self.index_json = cfg.index_json
        self.train_list = json.load(open(os.path.join(self.root,self.index_json),'r'))
        self.small_batch_data_list = []
        self.get_small_batch_data()
        #self.train_list = self.train_list[:20]
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.bg_color_mode = cfg.bg_color_mode
        self.normalize_coord = False
    
    def get_small_batch_data(self):
        for train_artboard in self.train_list:
            json_name, assets_img, artboard_img = train_artboard['json'], train_artboard['layerassets'], train_artboard['image']
            #print(assets_img)
            artboard_idx = json_name.split(".")[0]
            iter_idx = 0
            while True:
                iter_json_path = os.path.join(self.root, artboard_idx, f"{artboard_idx}-{iter_idx}.json")
                iter_img_path = os.path.join(self.root, artboard_idx, f"{artboard_idx}-{iter_idx}.png")
                if not os.path.exists(iter_json_path):
                    break
                self.small_batch_data_list.append([iter_json_path, iter_img_path])
                iter_idx += 1
    
    def __len__(self):
        return len(self.small_batch_data_list)

    def read_img_naive(self, path):
        return Image.open(path).convert("RGB")

    def read_img_keep_alpha(self, path):
        return Image.open(path).convert("RGBA")
    
    def read_img(self, path):
        #return Image.open(path).convert("RGB")
        if self.bg_color_mode == 'none':
            img_tensor = T.ToTensor()(Image.open(path).convert('RGB'))
        else:
            img_tensor = T.ToTensor()(Image.open(path).convert('RGBA'))
            if self.bg_color_mode == 'black':
                bg_color = 0.0
            elif self.bg_color_mode == 'grey':
                bg_color = 0.5
            elif self.bg_color_mode == 'white':
                bg_color = 1.0
            img_tensor = bg_color * (1 - img_tensor[3:,...]) + img_tensor[3:, ...] * img_tensor[:3, ...] 
        img_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
        return img_tensor

    def __getitem__(self, index) :
        
        graph_path, layer_img_path = self.small_batch_data_list[index]
        layer_img = self.img_transform(Image.open(layer_img_path).convert('RGB'))
        layer_img = layer_img.reshape(3, -1, 64, 64).transpose(1, 0)
        
        batch = []
        img_split_last_idx = 0
        nodes = 0
        
        layer_rect, edges, bbox, types, labels = read_graph_json(graph_path)
        layer_rect = torch.FloatTensor(layer_rect)
        orig_layer_rect = layer_rect.clone()
          
        edges = torch.LongTensor(edges).transpose(1,0)
        bbox = torch.FloatTensor(bbox)
        types = torch.LongTensor(types)
        labels = torch.LongTensor(labels)
        node_indices = torch.zeros(layer_rect.shape[0], dtype = torch.int64)
       
        nodes += layer_rect.shape[0]
        # nodes, edges, types,  img_tensor, labels, bboxes, orig_layer_rect, file_list
        # batch.append([layer_rect, edges, types, layer_img, labels, bbox, orig_layer_rect])
    
        return [layer_rect, edges, types, layer_img, labels, bbox, orig_layer_rect, node_indices, layer_img_path]

    def collate_fn(self, batch, collate_img_path=True):
    #layer_rect, edges, types, layer_img, labels, bbox, orig_layer_rect
        nodes_list = [b[0] for b in batch]
        nodes = torch.cat(nodes_list, dim=0)
        nodes_lens = np.fromiter(map(lambda l: l.shape[0], nodes_list), dtype=np.int64)
        nodes_inds = np.cumsum(nodes_lens)
        nodes_num = nodes_inds[-1]
        nodes_inds = np.insert(nodes_inds, 0, 0)
        nodes_inds = np.delete(nodes_inds, -1)
        edges_list = [b[1] for b in batch]
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
        
        origin_layer_rect_list = [b[6] for b in batch]
        orig_layer_rect = torch.cat(origin_layer_rect_list, dim=0)
        
        if collate_img_path:
            node_ind_list = [b[7] for b in batch]
            #node_ind_list = [cur + torch.max(i)for prev, cur in zip(node_ind_list[:-1],node_ind_list[1:])]
            graph_lens = np.fromiter(map(lambda l: l[-1]+1, node_ind_list), dtype=np.int64)
            graph_inds = np.cumsum(graph_lens)
            
            graph_inds = np.insert(graph_inds, 0, 0)
            graph_inds = np.delete(graph_inds, -1)
            node_ind_list = [e + int(i) for e, i in zip(node_ind_list, graph_inds)]
            node_indices = torch.cat(node_ind_list, dim=0)
            file_list = [b[8] for b in batch]

            return  nodes, edges, types,  img_tensor, labels, bboxes, orig_layer_rect, node_indices, file_list   
        return nodes, edges, types,  img_tensor, labels, bboxes, orig_layer_rect

if __name__=='__main__':
    cfg = CN()
    cfg.rootDir='/media/sda1/ljz-workspace/dataset/graph_dataset'
    cfg.train_json = 'index_train.json'
    dataset = Dataset(cfg)
    
    for i, batch in enumerate(dataset):
        layer_rects, collate_edges, collate_types, collate_labls,bboxes, layer_assets=batch
        print(layer_rects.shape)

