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
    cfg.test.batch_size = 1
    cfg.test_dataset.rootDir = '../../dataset/EGFE_graph_dataset_refine'
    cfg.test_dataset.index_json = 'index_testv2.json'
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
    result_transformer = json.load(open("eval_egfe_resplit_results.json", "r"))
    labels_pred = []
    labels_gt = []
    merge_recall = 0.0
    merge_precision = 0.0 
    merge_iou = 0.0
    merging_iou_recall = 0
    merging_iou_precision = 0
    not_valid_samples = 0
    text_type = 9
    for batch in tqdm(dataloader):
        #network(batch)
        nodes_, edges, types, img_tensor, labels, bboxes, nodes, node_indices, file_list  = batch
        if torch.sum(labels) == 0:
            not_valid_samples += 1
            continue
        file_path, artboard_name = os.path.split(file_list[0])
        artboard_name = artboard_name.split(".")[0]
        pred_result = torch.LongTensor(result_transformer[artboard_name+".json"])
        # print(artboard_name)
        pred_result_with_text = torch.zeros(nodes.shape[0]).long()
        # print(pred_result.shape, nodes.shape)
        assert(pred_result.shape[0] + torch.sum(types == text_type) == nodes.shape[0])
        pred_result_with_text_3_class = torch.zeros(nodes.shape[0]).long()

        pred_result_clone = pred_result.clone()
        pred_label = torch.masked_fill(pred_result_clone, pred_result_clone != 0, 1)
        pred_result_with_text[types != text_type] = pred_label
        pred_result_with_text_3_class[types != text_type] = pred_result

        labels_pred.append(pred_result_with_text)
        labels_gt.append(labels)

        bboxes = bboxes + nodes
        merging_groups_gt = get_comp_gt_list(bboxes, labels)
        #vis.visualize_nms_with_labels(bboxes[labels == 1], torch.ones((bboxes[labels]==1).shape[0]), file_list[0], mode = 'test', save_file = True )
        adj_gt = get_gt_adj(bboxes, labels)
        merging_list = get_merging_components_transformer(pred_result_with_text_3_class)
        if artboard_name == '347':
            print(pred_label)
            print(merging_list)
        adj_pred = get_pred_adj(merging_list, nodes.shape[0], nodes.device)
        
        merging_iou_recall += evaluator.evaluate_merging_iou(merging_groups_gt, merging_list)
        merging_iou_precision += evaluator.evaluate_merging_iou(merging_list, merging_groups_gt)
        
        vis_bboxes = []
        for merging_group in merging_list:
            vis_bboxes.append(get_the_bbox_of_cluster(bboxes[merging_group, :]))
        vis.visualize_nms(vis_bboxes, file_list[0])
        vis.visualize_pred_fraglayers(nodes[pred_result_with_text == 1], file_list[0], save_file = True)
        #merge_recall += torch.sum(adj_gt[labels==1, :][:, labels==1] == adj_pred[labels==1, :][:, labels==1]).item() / (torch.sum(labels) ** 2)
        tmp_merge_recall = 0
        for merge_comp_gt in merging_groups_gt:
            tmp_merge_recall += torch.sum(adj_gt[merge_comp_gt, :][:, merge_comp_gt] == adj_pred[merge_comp_gt, :][:, merge_comp_gt]).item() / ((merge_comp_gt.shape[0]) ** 2)    
        tmp_merge_recall /= len(merging_groups_gt)
        merge_recall += tmp_merge_recall
        
        if torch.sum(pred_result_with_text) == 0:
            merge_precision += 0.0
        else:
            #merge_precision_mask = (pred_result_with_text == 1)
            #merge_precision += torch.sum(adj_gt[merge_precision_mask, :][:, merge_precision_mask]  ==
            #                             adj_pred[merge_precision_mask, :][:, merge_precision_mask]).item() / (torch.sum(merge_precision_mask) ** 2)
            
            tmp_merge_precision = 0.0
            for merge_comp_pred in merging_list:
                tmp_merge_precision += torch.sum(adj_gt[merge_comp_pred, :][:, merge_comp_pred]
                                          == adj_pred[merge_comp_pred, :][:, merge_comp_pred]).item() / ((merge_comp_pred.shape[0]) ** 2)    
            
            if len(merging_list) != 0:
                tmp_merge_precision /= len(merging_list)
            merge_precision += tmp_merge_precision
            
    data_size = len(dataloader) - not_valid_samples
    print("merging recall: ", merge_recall / data_size)    
    print("merging precision: ", merge_precision / data_size)     
    print("merging iou precision: ", merging_iou_precision / data_size)  
    print("merging iou recall: ", merging_iou_recall / data_size)   
      
    print(evaluator.evaluate(torch.cat(labels_pred, dim = 0), torch.cat(labels_gt, dim = 0)))

    '''bboxes = bboxes + nodes
        file_path, artboard_name = os.path.split(file_list[0])
        artboard_name = artboard_name.split(".")[0]
        adj_gt = get_gt_adj(bboxes, labels)
        new_pred = torch.zeros(nodes.shape[0])

        #num_of_fragments.append((torch.sum(labels).item(), {"json": f"{artboard_name}.json", "layerassets":f"{artboard_name}-assets.png", "image":f"{artboard_name}.png"}))
        bbox, _ = nms_merge(bboxes[labels == 1], torch.ones(bboxes.shape[0])[labels==1])
        
        new_pred = evaluator.correct_pred_with_nms(new_pred, bbox, nodes, types, threshold=0.45)
        #merging_list = merging_components(torch.vstack(bbox), nodes, new_pred)
        #adj_pred = get_pred_adj(merging_list, nodes.shape[0], torch.device("cpu"))
        metrics = evaluator.evaluate(new_pred, labels)
        precision += metrics['precision']
        recall += metrics['recall']
        acc += metrics['accuracy']
        if torch.sum(labels) == 0:
            merge_acc += 1
        #merge_acc += torch.sum(adj_gt[labels==1, :][:, labels==1] == adj_pred[labels==1, :][:, labels==1]).item() / (torch.sum(labels) ** 2)
    print(merge_acc/len(dataloader), precision/len(dataloader), recall/len(dataloader), acc/len(dataloader))
        #if torch.sum(labels) == 0:
        #    zeros += 1'''

    '''positives += torch.sum(labels)
        negatives += labels.shape[0]-torch.sum(labels)
        if nodes.shape[0]>1001:
            print(file_list)'''
    '''
    num_of_fragments.sort(key = lambda x: x[0])
    print(zeros)

    data_list = [ x[1] for x in num_of_fragments if x[0] > 0 ]
    print(len(data_list) + zeros)
    
    non_valid = [x[1] for x in num_of_fragments if x[0] == 0]

    train_data_list = []
    test_data_list = []
    chunk_size = len(data_list) // 3
    print(chunk_size)
    print(data_list[0:chunk_size])
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
    '''