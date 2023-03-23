import torch
import torchvision.ops.focal_loss as focal_loss
import torch.nn.functional as F
import torch.nn as nn
import math

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input, target):
        p = F.softmax(input,dim=1)
        p = p[:,1]
        target=target.float()
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

def ciou_loss(box1, box2):
    """
    input:
        box1: shape = [batch_size,  4(xywh)]  xy is the left-up corner of bbox
        box2: shape = [batch_size,  4(xywh)] 
    output:
        ciou: shepe = [batch_size, 1]
    """
    # 计算box的左上角和右下角
    b1_xy = box1[..., :2]
    b1_wh = box1[..., 2:4]
    #b1_wh_half = b1_wh / 2
    b1_mins = b1_xy 
    b1_maxs = b1_xy + b1_wh

    b2_xy = box2[..., :2]
    b2_wh = box2[..., 2:4]
    #b2_wh_half = b2_wh / 2
    b2_mins = b2_xy
    b2_maxs = b2_xy + b2_wh
    # 计算iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxs = torch.min(b1_maxs, b2_maxs)
    intersect_wh = torch.max(intersect_maxs - intersect_mins, torch.zeros_like(intersect_maxs))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area, min=1e-6)
    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), -1)
    # 计算两个框的最小外包矩形的对角线距离
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxs = torch.max(b1_maxs, b2_maxs)
    enclose_wh = torch.max(enclose_maxs - enclose_mins, torch.zeros_like(enclose_maxs))
    enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), -1)

    v = 4 /(math.pi ** 2) * torch.pow(torch.atan(b1_wh[..., 0]/torch.clamp(b1_wh[..., 1], min=1e-6))-torch.atan(b2_wh[..., 0]/torch.clamp(b2_wh[..., 1], min=1e-6)), 2)

    alpha = v / torch.clamp((1 - iou + v), min=1e-6)
    ciou = 1 - iou + 1.0 * center_distance / torch.clamp(enclose_diagonal, min=1e-6) + alpha * v
    return ciou	

def make_classifier_loss(cfg):
    if cfg.type=='focal_loss':
        return FocalLoss(cfg.alpha, cfg.gamma, cfg.reduction)
    if cfg.type=='cross_entropy_loss':
        return torch.nn.CrossEntropyLoss()

def make_regression_loss(cfg):
    if cfg.type=='huber_loss':
        return torch.nn.HuberLoss(cfg.reduction, cfg.delta) 
    elif cfg.type=='ciou_loss':
        return ciou_loss
    elif cfg.type=='mse_loss':
        return torch.nn.MSELoss()
   