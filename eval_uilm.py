from cProfile import label
from mmdet.core import bbox
from scipy.sparse import data
from lib.utils.nms import get_the_bbox_of_cluster, nms_merge
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
import cv2
import numpy as np
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

def process_bbox_result():
    bbox_dict = json.load(open("bbox_result_egfe_data.json"))
    new_bbox_dict = {}
    for key in bbox_dict.keys():
        artboard_name, _, x_window, y_window = key.split("/")[0].split("-")
        new_bbox_dict.setdefault(artboard_name, [])
        new_bbox_dict[artboard_name].append({x_window + "-" + y_window : bbox_dict[key]})
    json.dump(new_bbox_dict, open("new_bbox_result_egfe_data.json", "w"))
    return new_bbox_dict

if __name__=='__main__':
    cfg.test.batch_size = 1
    cfg.test_dataset.rootDir = '../../dataset/EGFE_graph_dataset'
    cfg.test_dataset.index_json = 'index_test.json'
    cfg.test_dataset.bg_color_mode = 'bg_color_orig'
    dataloader = make_data_loader(cfg,is_train = False)
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
    #result_transformer = json.load(open("bbox_result.json"))
    labels_pred = []
    labels_gt = []
    merge_recall = 0.0
    merge_precision = 0.0 
    new_bbox_results = process_bbox_result()
    merge_iou = 0.0
    print(len(dataloader))
    not_valid_samples = 0
    merge_eval_stats = {}
    for batch in tqdm(dataloader):
        #network(batch)
        nodes_, edges, types, img_tensor, labels, bboxes, nodes, node_indices, file_list  = batch
        if torch.sum(labels) == 0:
            not_valid_samples += 1
            continue
        labels_gt.append(labels)
        #print(node_indices)
        file_path, artboard_name = os.path.split(file_list[0])
        artboard_name = artboard_name.split(".")[0]
        img_1 = cv2.cvtColor(cv2.imread(file_list[0]), cv2.COLOR_BGR2RGB)
        H, W, _ = img_1.shape
        window_size = min(H, W)
        pred_labels = torch.zeros(nodes.shape[0])
        if artboard_name not in new_bbox_results.keys():
            labels_pred.append(pred_labels)
            merge_precision += 0
            merge_recall += 0
            continue
        bbox_result_list = new_bbox_results[artboard_name]
        bbox_pred = []
        
        for bbox_dict in bbox_result_list:
            x_win, y_win = list(bbox_dict.keys())[0].split('-')
            x_win, y_win = float(x_win), float(y_win)
            bbox_list = np.array(list(bbox_dict.values())[0])
            bbox_tensor_for_this_window = torch.tensor(bbox_list, dtype = torch.float32)
            if bbox_tensor_for_this_window.shape[0] == 0:
                continue
            bbox_tensor_for_this_window = bbox_tensor_for_this_window[:, :4]
            bbox_tensor_for_this_window[:, 2 : 4] -= bbox_tensor_for_this_window[:, 0 : 2]
            bbox_tensor_for_this_window[:, 0] += (x_win * window_size)
            bbox_tensor_for_this_window[:, 1] += (y_win * window_size)
            scaling_factor = torch.tensor([W, H, W, H], dtype = torch.float32)
            bbox_tensor_for_this_window /= scaling_factor
            bbox_pred.append(bbox_tensor_for_this_window)
        if len(bbox_pred) == 0:
            labels_pred.append(pred_labels)
            merge_precision += 0
            merge_recall += 0
            continue
        bbox_pred = torch.cat(bbox_pred, dim = 0)
        #vis.visualize_not_scale(bbox_pred, file_list[0])
        vis.visualize_nms(bbox_pred, file_list[0])
        bboxes = bboxes + nodes
        vis.visualize_nms_with_labels(bboxes[labels == 1], torch.ones(bboxes[labels==1].shape[0]), file_list[0], mode = 'test', save_file = True )
        
        bbox_pred, _ = nms_merge(bbox_pred, torch.ones(bbox_pred.shape[0]), threshold = 0.7)
        pred_labels = evaluator.correct_pred_with_nms(pred_labels, bbox_pred, nodes, types)
        assert(pred_labels.shape[0] == nodes.shape[0])
        assert(labels.shape[0] == nodes.shape[0])
        labels_pred.append(pred_labels)
        eval_merging_dict = evaluator.evaluate_merging(bbox_pred, pred_labels, nodes, bboxes, labels)
        
        merging_groups_pred = merging_components(torch.vstack(bbox_pred), nodes, pred_labels)
        refine_bbox = []
        '''for merge_group in merging_groups_pred:
            refine_bbox.append(get_the_bbox_of_cluster(nodes[merge_group, :]))
        vis.visualize_nms(refine_bbox, file_list[0], mode = 'refine')
        '''
        merge_recall += eval_merging_dict['merge_recall']
        merge_precision += eval_merging_dict['merge_precision']
        #merge_iou += eval_merging_dict['merge_iou']
        for k, v in eval_merging_dict.items():
            merge_eval_stats.setdefault(k, 0)
            merge_eval_stats[k] += v
            
    merge_eval_state = []   
    data_size = len(dataloader) - not_valid_samples
    print(data_size)
    for k in merge_eval_stats.keys():
            merge_eval_stats[k] /= data_size
            merge_eval_state.append('{}: {}'.format(k, merge_eval_stats[k]))
    print(merge_eval_state) 
    print(evaluator.evaluate(torch.cat(labels_pred, dim = 0), torch.cat(labels_gt, dim = 0)))


