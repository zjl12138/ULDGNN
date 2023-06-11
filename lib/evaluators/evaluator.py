from re import M
from PIL.ImageOps import contain
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from lib.utils.nms import contains_how_much, get_comp_gt_list
from lib.utils import nms_merge, IoU, contains
import torch.nn.functional as F
from lib.utils import get_gt_adj, get_pred_adj, merging_components

class Evaluator:
    def __init__(self):
        pass

    def accuracy(self, pred, target):
        #S = target.cpu().numpy()
        #C = np.argmax( torch.nn.Softmax(dim=1)(logits).cpu().detach().numpy() , axis=1 )
        #print(C)
        CM = confusion_matrix(target,pred).astype(np.float32)
        nb_classes = CM.shape[0]
        nb_non_empty_classes = 0
        pr_classes = np.zeros(nb_classes)
        for r in range(nb_classes):
            cluster = np.where(target==r)[0]
            if cluster.shape[0] != 0:
                pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
                if CM[r,r]>0:
                    nb_non_empty_classes += 1
            else:
                pr_classes[r] = 0.0
        acc = 100.* np.sum(pr_classes)/ float(nb_classes)
        return acc
    
    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):
        C = pred.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        # acc = self.accuracy(C,target)
        acc = np.sum(C == target) / target.shape[0]
        # print(confusion_matrix(target, C).astype(np.float32))
        precision = precision_score(target, C, average = 'macro')
        recall = recall_score(target, C, average = 'macro')
        # recall = np.sum(C[target == 1] == 1) / np.sum(target == 1)
        # precision = np.sum(target[C == 1] == 1) / np.sum(C == 1)
        f1 = f1_score(target, C, average='macro')
        
        return {
                'precision': torch.tensor(precision),
                "recall": torch.tensor(recall),
                "f1-score": torch.tensor(f1),
                "accuracy": torch.tensor(acc)
               }
    
    def correct_pred_with_nms(self, pred,  bbox_results: torch.Tensor, layer_rects:torch.Tensor, types: torch.Tensor, threshold=0.45):
        # not_text = (types != 9) & (types != 13) & (types != 8) & (types != 1) & (types != 0) & (types != 10) & (types != 12)
        not_text = (types != 6)
        '''symbolMaster': 0,
        'group': 1,
        'oval': 2,
        'polygon': 3,
        'rectangle': 4,
        'shapePath': 5,
        'star': 6,
        'triangle': 7,
        'shapeGroup': 8,
        'text': 9,
        'symbolInstance': 10,
        'slice': 11,
        'MSImmutableHotspotLayer': 12,
        'bitmap': 13,'''
        for bbox_result in bbox_results:
            # iou = IoU(bbox_result, layer_rects)
            # inside = contains(bbox_result.unsqueeze(0), layer_rects)
            # if bbox_result[2].item() * bbox_result[3].item() >= (20 * 20) / (750 * 750):
            #     continue 
            inside = contains_how_much(bbox_result.unsqueeze(0), layer_rects)
            #correct_mask = inside > 0.7 
            correct_mask = torch.logical_and(not_text, inside > 0.7)
            #correct_mask = torch.logical_or(correct_mask , iou < threshold)
            pred = torch.masked_fill(pred, correct_mask, 1)
        return pred    
    
    def evaluate_merging(self, merging_groups_nms, pred_labels, layer_rect, bboxes_gt, labels_gt):
        n = layer_rect.shape[0]
        adj_gt = get_gt_adj(bboxes_gt, labels_gt)
        if len(merging_groups_nms) == 0:
            if torch.sum(labels_gt) != 0:
                return {"merge_recall" : 0.0, 'merge_precision' : 0.0, 
                    "merge_iou_recall" : 0.0, "merge_iou_precision": 0.0}
            else:
                return {"merge_recall" : 1.0, 'merge_precision' : 1.0, 
                    "merge_iou_recall" : 1.0, "merge_iou_precision": 1.0}
        merging_groups_pred = merging_components(torch.vstack(merging_groups_nms), layer_rect, pred_labels)
        merging_groups_gt = get_comp_gt_list(bboxes_gt, labels_gt)
        # print(merging_groups_pred)
        adj_pred = get_pred_adj(merging_groups_pred, layer_rect.shape[0], layer_rect.device)
        merging_iou_precision = 0.0
        merging_iou_recall = 0.0
        if torch.sum(labels_gt) == 0:
            assert(torch.sum(labels_gt) != 0)
            merging_iou_recall = 0
            merging_iou_precision = 0.0
            merge_recall = 0
        else:
            #merge_recall = torch.sum(adj_gt[labels_gt == 1, :][:, labels_gt == 1] == adj_pred[labels_gt == 1, :][:, labels_gt == 1]).item() / (torch.sum(labels_gt == 1) ** 2)
            merge_recall = 0
            for merge_comp_gt in merging_groups_gt:
                merge_recall += torch.sum(adj_gt[merge_comp_gt, :][:, merge_comp_gt] == adj_pred[merge_comp_gt, :][:, merge_comp_gt]).item() / ((merge_comp_gt.shape[0]) ** 2)    
           
            if len(merging_groups_gt) == 0:
                merge_recall = 1.0
            else:
                merge_recall /= len(merging_groups_gt)
            
        if torch.sum(pred_labels == 1) == 0:
            merge_precision = 0
            merging_iou_recall = 0
            merging_iou_precision = 0
        else:
            #merge_precision_mask = (pred_labels == 1)
            #merge_precision = torch.sum(adj_gt[merge_precision_mask, :][:, merge_precision_mask]  ==
            #                            adj_pred[merge_precision_mask, :][:, merge_precision_mask]).item() / (torch.sum(merge_precision_mask) ** 2)
            merge_precision = 0.0
            for merge_comp_pred in merging_groups_pred:
                merge_precision += torch.sum(adj_gt[merge_comp_pred, :][:, merge_comp_pred]
                                          == adj_pred[merge_comp_pred, :][:, merge_comp_pred]).item() / ((merge_comp_pred.shape[0]) ** 2)    
            
            if len(merging_groups_pred) != 0:
                merge_precision /= len(merging_groups_pred)
            
            #merging_iou = self.evaluate_merging_iou(merging_groups_pred, meging_groups_gt)
            merging_iou_recall = self.evaluate_merging_iou(merging_groups_gt, merging_groups_pred)
            merging_iou_precision = self.evaluate_merging_iou(merging_groups_pred, merging_groups_gt)
        return {"merge_recall" : merge_recall, 'merge_precision' : merge_precision, 
                "merge_iou_recall": merging_iou_recall, "merge_iou_precision":merging_iou_precision}
        
    def evaluate_bbox(self, merging_groups_nms, bboxes_gt):
        bboxes_gt = nms_merge(bboxes_gt, torch.ones(bboxes_gt.shape[0], device = bboxes_gt.get_device()))
    
    def set_iou(self, set1: torch.Tensor, set2: torch.Tensor):
        set1 = set(set1.cpu().numpy().tolist())
        set2 = set(set2.cpu().numpy().tolist())
        return (1.0 *len(set1 & set2)) / len(set1 | set2)

    def evaluate_merging_iou(self, comp_pred_list, comp_gt_list):
        # print(comp_pred_list, comp_gt_list)
        if len(comp_pred_list) == 0 or len(comp_gt_list)==0:
            return 0.0
        merging_iou = 0.0
        for comp_pred in comp_pred_list:
            max_iou = 0
            for comp_gt in comp_gt_list:
                max_iou = max(max_iou, self.set_iou(comp_pred, comp_gt))
            assert(max_iou >= 0)
            merging_iou += max_iou
        merging_iou /= len(comp_pred_list)
        return merging_iou

    def evaluate_merging_box_iou(self):
        pass

