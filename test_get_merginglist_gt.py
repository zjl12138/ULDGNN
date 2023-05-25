from scipy.sparse import data
from lib.utils.nms import nms_merge, get_comp_gt_list, get_the_bbox_of_cluster
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
    cfg.train_dataset.rootDir = '../../dataset/graph_dataset_rererefine_copy'
    cfg.train_dataset.index_json = 'index_testv2.json'
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
       
        bboxes = bboxes + nodes
        merging_groups_gt = get_comp_gt_list(bboxes, labels)
        #vis.visualize_nms_with_labels(bboxes[labels == 1], torch.ones((bboxes[labels]==1).shape[0]), file_list[0], mode = 'test', save_file = True )
        refine_bbox = []
        for merge_group in merging_groups_gt:
            refine_bbox.append(get_the_bbox_of_cluster(nodes[merge_group, :]))
        vis.visualize_nms(refine_bbox, file_list[0], mode = 'refine')
        