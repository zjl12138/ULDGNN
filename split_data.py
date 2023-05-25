from scipy.sparse import data
from lib.utils.nms import nms_merge, get_comp_gt_list
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
    cfg.train.batch_size = 1
    cfg.train_dataset.rootDir = '../../dataset/EGFE_graph_dataset'
    cfg.train_dataset.index_json = 'index.json'
    cfg.train_dataset.bg_color_mode = 'bg_color_orig'
    dataloader = make_data_loader(cfg,is_train=True)
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
    result_transformer = json.load(open("egfe_test_stats.json"))
    labels_pred = []
    labels_gt = []
    merge_recall = 0.0
    merge_precision = 0.0 
    merge_iou = 0.0
    merging_iou_recall = 0
    merging_iou_precision = 0
    
    for batch in tqdm(dataloader):
        #network(batch)
        nodes_, edges, types, img_tensor, labels, bboxes, nodes, node_indices, file_list  = batch
        file_path, artboard_name = os.path.split(file_list[0])
        artboard_name = artboard_name.split(".")[0]
        num_of_fragments.append((torch.sum(labels).item(), {"json": f"{artboard_name}.json", "layerassets":f"{artboard_name}-assets.png", "image":f"{artboard_name}.png"}))
        if torch.sum(labels) == 0:
            zeros += 1
            
    num_of_fragments.sort(key = lambda x: x[0])
    print("number of artboard with no positive samples: ", zeros)

    data_list = [ x[1] for x in num_of_fragments if x[0] > 0 ]
    print(len(data_list) + zeros)
    
    non_valid = [x[1] for x in num_of_fragments if x[0] == 0]

    train_data_list = []
    test_data_list = []
    chunk_size = len(data_list) // 3
    
    first_level_data = data_list[0:chunk_size]
    random.shuffle(first_level_data)

    first_level_train_data, first_level_test_data = train_test_split(first_level_data, test_size = 0.2)
   
    second_level_data = data_list[chunk_size:2*chunk_size]
    random.shuffle(second_level_data)
    second_level_train_data, second_level_test_data = train_test_split(second_level_data, test_size = 0.2)
    
    third_level_data =data_list[2*chunk_size:]
    random.shuffle(third_level_data)
    third_level_train_data, third_level_test_data = train_test_split(third_level_data, test_size = 0.2)

    train_data_list = first_level_train_data + second_level_train_data + third_level_train_data + non_valid
    test_data_list  = first_level_test_data + second_level_test_data + third_level_test_data
    train_data_list_no_zeros = first_level_train_data + second_level_train_data + third_level_train_data
    assert(len(train_data_list) + len(test_data_list) == len(dataloader))
    json.dump(train_data_list,open("index_trainv2.json","w"))
    json.dump(test_data_list,open("index_testv2.json","w"))
    json.dump(train_data_list_no_zeros,open("index_train_no_zeros.json","w"))
    print(num_of_fragments)