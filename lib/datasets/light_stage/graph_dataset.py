
from typing import List
from torch.utils import data
import json
import os
import glob
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
from lib.config.yacs import CfgNode as CN
import colorsys
from torchvision.utils import save_image
from lib.utils import get_the_bbox_of_cluster

def collate_fn(batch, collate_img_path=True):
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
        self.root = cfg.rootDir
        self.index_json = cfg.index_json
        self.train_list = json.load(open(os.path.join(self.root,self.index_json),'r'))
        #self.train_list = self.train_list[:20]
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.bg_color_mode = cfg.bg_color_mode
        self.normalize_coord = False
        
    def __len__(self):
        return len(self.train_list)

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
        
        train_artboard = self.train_list[index]
        json_name, assets_img, artboard_img = train_artboard['json'], train_artboard['layerassets'], train_artboard['image']
        #print(assets_img)
        artboard_idx = json_name.split(".")[0]
        iter_idx = 0
        graphs_in_this_arboard = []
        while True:
            iter_json_path = os.path.join(self.root, artboard_idx, f"{artboard_idx}-{iter_idx}.json")
            if not os.path.exists(iter_json_path):
                break
            graphs_in_this_arboard.append(iter_json_path)
            iter_idx += 1
            
        #graphs_in_this_arboard = glob.glob(os.path.join(self.root, artboard_idx)+"/*.json")
        #graphs_in_this_arboard.sort()
        #layer_assets = self.img_transform(self.read_img_naive(os.path.join(self.root, artboard_idx,assets_img)))
        #layer_assets = list(torch.split(layer_assets,64,dim=1))
        if self.bg_color_mode == 'complementray_color':
            if not os.path.exists(os.path.join(self.root, artboard_idx, f"{artboard_idx}-assets_refine.png")):
                print("refine layer assets image!")
                path = os.path.join(self.root, artboard_idx, assets_img)
                img_tensor = T.ToTensor()(Image.open(path).convert('RGBA'))
                layer_assets = list(torch.split(img_tensor, 64, dim = 1))
                bg_color_list = []
                for layer_img in layer_assets:
                    img_array = T.ToPILImage()(layer_img)
                    img_array = np.array(img_array)
                    img_array = img_array[:3, img_array[3, ...] > 0]
                    if img_array.shape[1] == 0:
                        bg_color = (1.0, 1.0, 1.0)
                    else:
                        bg_color = freq_max(img_array.transpose(1, 0))
                        bg_color = find_contrast_color(bg_color)
                    bg_color_list.append(torch.tensor(bg_color, dtype = torch.float32)[:, None, None].repeat(1, 64, 64))

                bg_color_tensor = torch.cat(bg_color_list, dim = 1) #N, 3
                img_tensor = bg_color_tensor * (1 - img_tensor[3:, ...]) + img_tensor[3:, ...] * img_tensor[:3, ...] 
                save_image(img_tensor, os.path.join(self.root, artboard_idx,  f"{artboard_idx}-assets_refine.png"))
            else:
                img_tensor = self.img_transform(self.read_img_naive(os.path.join(self.root, artboard_idx, f"{artboard_idx}-assets_refine.png")))
            layer_assets = list(torch.split(img_tensor, 64, dim = 1))
        #layer_assets = self.img_transform(self.read_img_naive(os.path.join(self.root, artboard_idx,assets_img)))
        elif self.bg_color_mode == 'bg_color_orig':
            if not os.path.exists(os.path.join(self.root, artboard_idx, f"{artboard_idx}-assets_rerefine.png")):
                #assert(os.path.exists(os.path.join(self.root, artboard_idx, f"{artboard_idx}-assets_rerefine.png")))
                path = os.path.join(self.root, artboard_idx, assets_img)
                img_tensor = T.ToTensor()(Image.open(path).convert('RGBA'))
                bg_color_list = []

                artboard_img_tensor = T.ToTensor()(Image.open(os.path.join(self.root, artboard_idx, artboard_img)).convert('RGB'))
                C, H, W = artboard_img_tensor.shape
                for idx, graph_path in enumerate(graphs_in_this_arboard):
                #content['layer_rect'],content['edges'], content['bbox'],content['types'],content['labels']
                    layer_rects, edges, bbox, types, labels = read_graph_json(graph_path) 
                    for layer_rect in layer_rects:
                        x, y, w, h = int(layer_rect[0] * W), int(layer_rect[1] * H), int(layer_rect[2] * W), int(layer_rect[3] * H)
                        if x == W:
                            x = x - 1
                        if y == H:
                            y = y - 1
                        bg_color = artboard_img_tensor[:, y, x]
                        bg_color_list.append(bg_color[:, None, None].repeat(1, 64, 64))
                bg_color_tensor = torch.cat(bg_color_list, dim = 1) #N, 3
                img_tensor = bg_color_tensor * (1 - img_tensor[3:, ...]) + img_tensor[3:, ...] * img_tensor[:3, ...] 
                save_image(img_tensor, os.path.join(self.root, artboard_idx,  f"{artboard_idx}-assets_rerefine.png"))
            else:
                img_tensor = self.img_transform(self.read_img_naive(os.path.join(self.root, artboard_idx, f"{artboard_idx}-assets_rerefine.png")))
            layer_assets = list(torch.split(img_tensor, 64, dim = 1))

        elif self.bg_color_mode == 'keep_alpha':
            layer_assets = T.ToTensor()(self.read_img_keep_alpha(os.path.join(self.root, artboard_idx, assets_img)))
            layer_assets = list(torch.split(layer_assets, 64, dim = 1))
        
        else:   
            layer_assets = self.read_img(os.path.join(self.root, artboard_idx, assets_img))
            layer_assets = list(torch.split(layer_assets, 64, dim = 1))
        
        artboard_img_path = os.path.join(self.root, artboard_idx, train_artboard['image'])

        batch = []
        img_split_last_idx = 0
        nodes = 0
        node_ind_list= []
        for idx, graph_path in enumerate(graphs_in_this_arboard):
            #content['layer_rect'], content['edges'], content['bbox'],content['types'],content['labels']
            layer_rect, edges, bbox, types, labels = read_graph_json(graph_path)
            layer_rect = torch.FloatTensor(layer_rect)
            orig_layer_rect = layer_rect.clone()
            if self.normalize_coord:
                graph_bbox = get_the_bbox_of_cluster(layer_rect)
                layer_rect[:, 0 : 2] -= graph_bbox[0 : 2]
                scaling_tensor = torch.cat( (graph_bbox[2 : 4], graph_bbox[2 : 4]), dim = 0)
                layer_rect /= scaling_tensor
            
            edges = torch.LongTensor(edges).transpose(1,0)
            bbox = torch.FloatTensor(bbox)
            types = torch.LongTensor(types)
            labels = torch.LongTensor(labels)
            node_indices = torch.zeros(layer_rect.shape[0], dtype = torch.int64) + idx
            node_ind_list.append(node_indices)
            #print(layer_rect.shape[0])
            nodes += layer_rect.shape[0]
            layer_img = torch.stack(layer_assets[img_split_last_idx : img_split_last_idx + layer_rect.shape[0]])
            #nodes, edges, types,  img_tensor, labels, bboxes, orig_layer_rect, file_list
            batch.append([layer_rect, edges, types, layer_img, labels, bbox, orig_layer_rect])
            img_split_last_idx += layer_rect.shape[0]
        
        return [*collate_fn(batch, False), torch.cat(node_ind_list, dim = 0), artboard_img_path]

if __name__=='__main__':
    cfg = CN()
    cfg.rootDir='/media/sda1/ljz-workspace/dataset/graph_dataset'
    cfg.train_json = 'index_train.json'
    dataset = Dataset(cfg)
    
    for i, batch in enumerate(dataset):
        layer_rects, collate_edges, collate_types, collate_labls,bboxes, layer_assets=batch
        print(layer_rects.shape)

