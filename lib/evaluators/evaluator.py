from PIL.ImageOps import contain
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from lib.utils import nms_merge, IoU, contains
import torch.nn.functional as F

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
        acc = self.accuracy(C,target)
        #print(confusion_matrix(target,C).astype(np.float32))
        precision = precision_score(C, target, average='macro')
        recall = recall_score(C, target, average='macro')
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
            inside = contains(bbox_result, layer_rects)
            correct_mask = torch.logical_and(not_text, inside)
            #correct_mask = torch.logical_or(correct_mask , iou < threshold)
            pred = torch.masked_fill(pred, correct_mask, 1)
        return pred    
    