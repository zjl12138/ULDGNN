from scipy.sparse import data
from lib.utils.nms import get_the_bbox_of_cluster, nms_merge, get_comp_gt_list
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

import random

def get_merging_components_transformer(pred_label : torch.Tensor):
    N = pred_label.shape[0]
    merging_list = []
    tmp = []
    for i in range(N):
        if pred_label[i] == 2:
            if len(tmp) == 0:
                tmp.append(i)
            else:
                merging_list.append(torch.LongTensor(tmp))
                tmp = []
                tmp.append(i)
        elif pred_label[i] == 1:
            tmp.append(i)
        else:
            continue
    if len(tmp) > 0:
        merging_list.append(torch.LongTensor(tmp))
    return merging_list

if __name__=='__main__':
    
    vis = visualizer(cfg.visualizer)
    rootDir = "/media/sda1/ljz-workspace/dataset/EGFE_data_from_uldgnn_data"
    #rootDir = "../../dataset/transformer_dataset"
    index_test_list = json.load(open(f"{rootDir}/index_tmp.json"))
    positive_labels_nums = 0
    for idx in tqdm(index_test_list):
        artboard_path = os.path.join(rootDir, idx['image'])
        if not os.path.exists(os.path.join(rootDir, idx['json'])):
            continue
        artboard_json = json.load(open(os.path.join(rootDir, idx['json']),"r"))
        merge_group_list = []
        tmp = []
        #if len(artboard_json['layers']) > 150:
        #    print(idx, len(artboard_json['layers']))
        for layer in artboard_json['layers']:
            bbox = layer['rect']
            xywh = [bbox['x'],  bbox['y'],  bbox['width'],  bbox['height']]
            label = layer['label']
            '''
            if layer['label'] == 3:
                label = 1
            elif layer['label'] == 1:
                label = 0
            '''    
            if label == 2:
                positive_labels_nums += 1
                if len(tmp) == 0:
                    tmp.append(torch.tensor(xywh))
                else:
                    merge_bbox = get_the_bbox_of_cluster(torch.vstack(tmp))
                    merge_group_list.append(merge_bbox)
                    tmp = []
                    tmp.append(torch.tensor(xywh))
            elif label == 1:
                positive_labels_nums += 1
                tmp.append(torch.tensor(xywh))
            else:
                continue
        if len(tmp) != 0:
            merge_group_list.append(get_the_bbox_of_cluster(torch.vstack(tmp)))
        vis.visualize_not_scale(merge_group_list, artboard_path)
    print(positive_labels_nums)
        
                
                