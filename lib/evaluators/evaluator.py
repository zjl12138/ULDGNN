import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self):
        pass

    def accuracy(self, logits, target):
        S = target.cpu().numpy()
        C = np.argmax( torch.nn.Softmax(dim=1)(logits).cpu().detach().numpy() , axis=1 )
        CM = confusion_matrix(S,C).astype(np.float32)
        nb_classes = CM.shape[0]
        target = target.cpu().numpy()
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
    
    def evaluate(self, output, target):
        logits, _ = output
        acc = self.accuracy(logits,target)
        C = np.argmax( torch.nn.Softmax(dim=1)(logits).cpu().detach().numpy() , axis=1 )
        target = target.cpu().detach().numpy()
        precision = precision_score(C, target, average='macro')
        recall = recall_score(C, target, average='macro')
        f1 = f1_score(C,target,average='macro')
        
        return {
                'precision': torch.tensor(precision),
                "recall": torch.tensor(recall),
                "f1-score": torch.tensor(f1),
                "accuracy": torch.tensor(acc)
               }