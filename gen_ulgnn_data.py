
import math
from lib.utils.nms import nms_merge
from lib.datasets import make_data_loader
from lib.config import cfg
from lib.visualizers import visualizer
from lib.networks import make_network
import torch
from lib.train import make_optimizer, make_recorder, make_scheduler, make_trainer
from lib.evaluators import Evaluator
from lib.utils import load_model, save_model, get_pred_adj, merging_components, get_gt_adj
from tqdm import tqdm
from torch_geometric.utils import to_dense_batch
import os
from typing import List
from sklearn.model_selection import train_test_split
import json
from PIL import Image, ImageDraw
import random
from random import choice
from typing import List, Tuple, AnyStr
import numpy as np
from lib.utils.gen_graphs_utils import bboxTree, bboxNode
from torchvision.utils import save_image

def random_color(n_colors):
    print('generating random color list')
    r_list, g_list, b_list, color_list = [], [], [], []
    levels = range(32, 256)
    for _ in range(n_colors):
        red, green, blue = tuple(choice(levels) for _ in range(3))
        r_list.append(red)
        g_list.append(green)
        b_list.append(blue)
    color_list.append(r_list)
    color_list.append(g_list)
    color_list.append(b_list)
    np.save('color.npy', color_list)
    return color_list

if __name__=='__main__':
    
    cfg.test.batch_size = 1
    mode = 'trainv2'
    print(mode)
    cfg.test_dataset.rootDir = '../../dataset/graph_dataset_rererefine_copy'
    cfg.test_dataset.index_json = f'index_{mode}.json'
    cfg.test_dataset.bg_color_mode = 'bg_color_orig'
    dataloader = make_data_loader(cfg, is_train=False)
    vis = visualizer(cfg.visualizer)
    evaluator = Evaluator()
    print(len(dataloader))
    #trainer.train(0, dataloader, optim, recorder, evaluator )
    positives = 0
    negatives = 0
    num_of_fragments: List = []
    zeros = 0
    merge_acc = 0.0
    precision = 0.0
    recall = 0.0
    acc = 0.
    labels_pred = []
    labels_gt = []
    merge_recall = 0.0
    merge_precision = 0.0 
    dataset_folder = os.path.join("../../dataset/ULDGNN_datasettest/")
    os.makedirs(dataset_folder, exist_ok = True)
    index_list = []
    for batch in tqdm(dataloader):
        # network(batch)
        is_used_as_test_or_train = False
        layer_rects_, edges, types, img_tensor, labels, bboxes, layer_rects, node_indices, file_list  = batch
        bbox_clone = bboxes.clone()
        bboxes = bboxes + layer_rects
        bboxes = bboxes[labels == 1]
        # print(node_indices) 
        if torch.sum(labels) == 0:
           continue
        file_path, artboard_name = os.path.split(file_list[0])
        artboard_name = artboard_name.split(".")[0]
        save_folder = os.path.join(dataset_folder, artboard_name) # save_folder is the folder to dump the data of this artboard
        os.makedirs(save_folder, exist_ok = True)
        save_img_path = os.path.join(save_folder, artboard_name+".png")
        os.system(f"cp {file_list[0]} {save_img_path}")
        
        artboard_img = Image.open(file_list[0]).convert("RGBA")   
        semantic_map = Image.new(artboard_img.mode, artboard_img.size, 'white') 
        
        if not os.path.exists("color.npy"):
            color_list = random_color(10000)
        else:
            color_list = list(np.load("color.npy"))
            
        W, H = artboard_img.size
        semantic_draw = ImageDraw.ImageDraw(semantic_map, semantic_map.mode)
        scaling_factor = torch.tensor([W, H, W, H], dtype = torch.float32)  # used to scale the layer_rect, bboxes to match the artboard size
        layer_rects = layer_rects * scaling_factor
        bbox_clone = bbox_clone * scaling_factor + layer_rects 
        
        single_image_width = W
        single_image_height = int(W / 1)
        # calculate image need to process
        total = H / single_image_height
        box_range: List[Tuple[float, float]] = [
            (0, float(y)) for y in np.arange(0, total, step=0.5)
        ]                          # calculate the scaling factor of each window
        if total < 1:
            # re-calculate split images' width and height
            single_image_width = int(H * 1)
            single_image_height = H
            # calculate image need to process
            total = W / single_image_width
            box_range = [(float(x), 0)
                            for x in np.arange(0, total, step=0.5)]
        width = W
        height = H
        
        layer_list = [[] for i in range(len(box_range))]
            
        # if layer_rect[count] is contained by box_range[idx], we append it to layer_list[idx]
        for count, layer_rect in enumerate(layer_rects.numpy().tolist()):
            l_x, l_y, l_w, l_h = int(layer_rect[0]), int(layer_rect[1]), int(layer_rect[2]), int(layer_rect[3])
            semantic_draw.rectangle(
                (l_x, l_y, l_x + l_w, l_y + l_h), 
                fill = (color_list[0][count],
                        color_list[1][count],
                        color_list[2][count],
                        int(0.3 * 255)
                     ))
            for idx, (x, y) in enumerate(box_range):
                img_box = (math.floor(single_image_width * x),
                            math.floor(single_image_height * y),
                            math.floor(min(single_image_width * (x + 1),
                                            width)),
                            math.floor(
                                min(single_image_height * (y + 1), height)))
                if (img_box[0] <= l_x and img_box[1] <= l_y and img_box[2] >= l_x + l_w and img_box[3] >= l_y + l_h):
                    layer_list[idx].append(count)
                    # check if the merging group of this layer belong to this box_range
                    if labels[count] == 1:
                        box_tmp = bbox_clone[count]
                        box_x1, box_y1, box_w, box_h = box_tmp[0], box_tmp[1], box_tmp[2], box_tmp[3]
                        box_x2, box_y2 = box_x1 + box_w, box_y1 + box_h
                        if not (img_box[0] <= box_x1 and img_box[1] <= box_y1 and img_box[2] >= box_x2 and img_box[3] >= box_y2):
                            labels[count] = 0
        #semantic_map.save(os.path.join(save_img_folder, f"{artboard_name}-filled.png"))
        semantic_map = Image.alpha_composite(artboard_img, semantic_map)
        #.save(os.path.join(save_img_folder, f"{artboard_name}-default.png"))
        
        bboxes, _ = nms_merge(bboxes, torch.ones(bboxes.shape[0]))
        bboxes = [[int(bbox[0].item() * W), int(bbox[1].item() * H), 
                   int(bbox[2].item() * W + bbox[0].item() * W), int(bbox[3].item() * H + bbox[1].item() * H)] for bbox in bboxes]
        

        graph_id = 0
        
        img_assets = []
        
        for window_idx, (x, y) in enumerate(box_range):
                # calculate crop box in order: (left, top, right, bottom)
            img_box = (math.floor(single_image_width * x),
                        math.floor(single_image_height * y),
                        math.floor(min(single_image_width * (x + 1),
                                        width)),
                        math.floor(
                            min(single_image_height * (y + 1), height)))
            
            labelstmp = list(
                    # filter label in crop box
                    filter(
                        lambda target_label:
                        img_box[0] <= target_label[0]  and img_box[1] <= target_label[1] and
                        img_box[2] >= target_label[2] and img_box[3] >= target_label[3],
                        bboxes
                    ))
            if len(labelstmp) > 0:
                
                if len(layer_list[window_idx]) < 5:
                    continue
                
                is_used_as_test_or_train = True
                
                root: bboxTree = bboxTree(0, 0, 2, 2)
                layer_ids = torch.LongTensor(layer_list[window_idx])
                
                layers = layer_rects[layer_ids]
                bbox_in_this_window = bbox_clone[layer_ids]
                
                types_in_this_window = types[layer_ids]
                labels_in_this_window = labels[layer_ids]
                img_assets.append(img_tensor[layer_ids, ...])
                save_image(img_tensor[layer_ids, ...].transpose(1, 0).reshape(3, -1, 64), 
                           os.path.join(save_folder, f'{artboard_name}-{graph_id}.png'))
                
                layers[:, 0 : 2] -= torch.tensor([math.floor(single_image_width * x), math.floor(single_image_height * y)], dtype = torch.float32)                
                layers /= single_image_height
                layers[:, 2 : 4] += layers[:, 0 : 2]
                
                bbox_in_this_window[:, 0 : 2] -= torch.tensor([math.floor(single_image_width * x), math.floor(single_image_height * y)], dtype = torch.float32)                
                bbox_in_this_window /= single_image_height
                
                bbox_in_this_window -= layers
                
                for layer in layers:
                    l_x, l_y, l_w, l_h = layer[0].item(), layer[1].item(), layer[2].item(), layer[3].item()
                    root.insert(bboxNode(root.num, l_x, l_y, l_x + l_w, l_y + l_h))
                edges = root.gen_graph([])
                
                json.dump({'layer_rect': layers.numpy().tolist(),
                    'edges': edges,
                    'bbox': bbox_in_this_window.numpy().tolist(), 
                    'types': types_in_this_window.numpy().tolist(), 'labels': labels_in_this_window.numpy().tolist(), 'artboard_width': W, 
                    'artboard_height': H,
                    'original_artboard': artboard_name,
                    'offset_in_artboard':[math.floor(single_image_width * x), math.floor(single_image_height * y)]},
                    open(os.path.join(save_folder, f"{artboard_name}-{graph_id}.json"), "w"))
                
                json.dump({'W': W, "H": H, 'patch_size': single_image_height}, open(os.path.join(save_folder,"meta.json"), "w"))
                
                labeled_image = Image.new(
                            artboard_img.mode,
                            (single_image_width, single_image_height), "white")
                crop_img_box = artboard_img.crop(img_box)
                labeled_image.paste(crop_img_box, (0, 0))
                
                artboard_label_draw = ImageDraw.ImageDraw(labeled_image, labeled_image.mode)
                for point in labelstmp:
                    label_bbox = np.array(point) - np.array([math.floor(single_image_width * x), math.floor(single_image_height * y),
                                                            math.floor(single_image_height * x), math.floor(single_image_height * y)])
                    artboard_label_draw.rectangle(label_bbox.tolist(), outline = 'red')
                labeled_image.save(os.path.join(os.path.join(save_folder, artboard_name + f'-labeled-{graph_id}.png')))
                graph_id += 1
                
                filled_image = Image.new(
                        semantic_map.mode,
                        (single_image_width, single_image_height), "white")
                filled_image.paste(semantic_map.crop(img_box),
                                    (0, 0))
                filled_image.save(
                    os.path.join(os.path.join(save_folder, artboard_name+f'-{graph_id}-filled.png')))
        
        if is_used_as_test_or_train:
            index_list.append({"json": f"{artboard_name}.json", "layerassets": f"{artboard_name}-assets.png", "image": f"{artboard_name}.png"})
                
        if len(img_assets) != 0:
            save_image(torch.cat(img_assets, dim = 0).transpose(1, 0).reshape(3, -1, 64), os.path.join(save_folder, artboard_name + '-assets_rerefine.png'))
    
    json.dump(index_list, open(os.path.join(dataset_folder, f"index_{mode}.json"), "w"))