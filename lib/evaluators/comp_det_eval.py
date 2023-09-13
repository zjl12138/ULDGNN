from re import M
from PIL.ImageOps import contain
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from lib.utils.nms import contains_how_much, get_comp_gt_list, get_comp_gt_list_vectorize
from lib.utils import nms_merge, IoU, contains
import torch.nn.functional as F
from lib.utils import get_gt_adj, get_pred_adj, merging_components, get_comp_gt_list_vectorize, get_gt_adj_vectorize

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
        '''precision_macro = precision_score(target, C, average = 'macro')
        recall_macro = recall_score(target, C, average = 'macro')
        # recall = np.sum(C[target == 1] == 1) / np.sum(target == 1)
        # precision = np.sum(target[C == 1] == 1) / np.sum(C == 1)
        f1_macro= f1_score(target, C, average='macro')
        
        precision_weight = precision_score(target, C, average = 'weighted')
        recall_weight = recall_score(target, C, average = 'weighted')
        # recall = np.sum(C[target == 1] == 1) / np.sum(target == 1)
        # precision = np.sum(target[C == 1] == 1) / np.sum(C == 1)
        f1_weight = f1_score(target, C, average='weighted')
        '''
        ''''precision_macro': torch.tensor(precision_macro),
                "recall_macro": torch.tensor(recall_macro),
                "f1-score_macro": torch.tensor(f1_macro),
                
                'precision_weighted': torch.tensor(precision_weight),
                "recall_weighted": torch.tensor(recall_weight),
                "f1-score_weighted": torch.tensor(f1_weight),
                '''
        return {
                "accuracy": torch.tensor(acc)
               }