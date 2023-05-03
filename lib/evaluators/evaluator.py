from PIL.ImageOps import contain
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
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
        #acc = self.accuracy(C,target)
        acc = np.sum(C == target) / target.shape[0]
        #print(confusion_matrix(target,C).astype(np.float32))
        '''precision = precision_score(C, target, average='macro')
        recall = recall_score(C, target, average='macro')'''

        recall = np.sum(C[target == 1] == 1) / np.sum(target == 1)
        precision = np.sum(target[C == 1] == 1) / np.sum(C == 1)
        f1 = f1_score(C,target,average='micro')
        
        return {
                'precision': torch.tensor(precision),
                "recall": torch.tensor(recall),
                "f1-score": torch.tensor(f1),
                "accuracy": torch.tensor(acc)
               }
    
    def correct_pred_with_nms(self, pred,  bbox_results: torch.Tensor, layer_rects:torch.Tensor, types: torch.Tensor, threshold=0.45):

        not_text = (types != 6)
        for bbox_result in bbox_results:
            iou = IoU(bbox_result, layer_rects)
            inside = contains(bbox_result.unsqueeze(0), layer_rects)
            correct_mask = torch.logical_and(not_text, inside)
            #correct_mask = torch.logical_or(correct_mask , iou < threshold)
            pred = torch.masked_fill(pred, correct_mask, 1)
        return pred    
    
    def evaluate_merging(self, merging_groups_nms, pred_labels, layer_rect, bboxes_gt, labels_gt):
        n = layer_rect.shape[0]
        adj_gt = get_gt_adj(bboxes_gt, labels_gt)
        if len(merging_groups_nms) == 0:
            return 1
        merging_list = merging_components(torch.vstack(merging_groups_nms), layer_rect, pred_labels)
        adj_pred = get_pred_adj(merging_list, layer_rect.shape[0], layer_rect.get_device())
        if torch.sum(labels_gt) == 0:
            return 1
        return torch.sum(adj_gt[labels_gt==1, :][:, labels_gt==1] == adj_pred[labels_gt==1, :][:, labels_gt==1]).item() / (torch.sum(labels_gt) ** 2)

    def evaluate_bbox(self, merging_groups_nms, bboxes_gt):
        bboxes_gt = nms_merge(bboxes_gt, torch.ones(bboxes_gt.shape[0], device = bboxes_gt.get_device()))
        