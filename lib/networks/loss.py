import torch
import torchvision.ops.focal_loss as focal_loss
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input, target):
        p = torch.nn.Softmax(dim=0)(input)
        ce = F.binary_cross_entropy_with_logits(p,target)
        p_t = p * target + (1-p)*(1-target)
        loss = ce * ((1 - p_t)**self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1-self.alpha)*(1-target)
            loss = alpha_t * loss
        
        if self.reduction is None:
            pass
        elif self.reduction=='mean':
            loss = loss.mean()
        elif self.reduction=='sum':
            loss = loss.sum()
        return loss

def make_classifier_loss(cfg):
    if cfg.type=='focal_loss':
        return FocalLoss(cfg.alpha, cfg.gamma, cfg.reduction)

def make_regression_loss(cfg):
    if cfg.type=='huber_loss':
        return torch.nn.HuberLoss(cfg.reduction, cfg.delta) 
   