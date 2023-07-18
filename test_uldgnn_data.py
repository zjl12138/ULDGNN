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
    cfg.test_dataset.module = 'lib.datasets.light_stage.graph_dataset_new'
    cfg.test_dataset.path = 'lib/datasets/light_stage/graph_dataset_new.py'
    cfg.test.batch_size = 1
    cfg.test_dataset.rootDir = '../../dataset/ULDGNN_graph_dataset'
    cfg.test_dataset.index_json = 'index_testv2.json'
    cfg.test_dataset.bg_color_mode = 'bg_color_orig'
    dataloader = make_data_loader(cfg,is_train=False)
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
    result_transformer = json.load(open("result.json"))
    labels_pred = []
    labels_gt = []
    merge_recall = 0.0
    merge_precision = 0.0 
    merge_iou = 0.0
    for batch in tqdm(dataloader):
        # network(batch)
        nodes_, edges, types, img_tensors, labels, bboxes, nodes, node_indices, file_list  = batch
        # print(node_indices)
        positives += torch.sum(labels)
        negatives += labels.shape[0] - torch.sum(labels)
        # print(positives, negatives)
    
        vis.visualize_recon_artboard(nodes, img_tensors, file_list[0])
        vis.visualize_pred_fraglayers(nodes, file_list[0], save_file=True)
        bboxes = bboxes + nodes
        vis.visualize_nms(bboxes[labels == 1], file_list[0], save_file = True)
        