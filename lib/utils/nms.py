import torch
import os
import json
'''
def IoU(single_box, box_list):
    left_up_corner = torch.maximum(single_box[0:2], box_list[:,0:2])
    right_down_corner = torch.minimum(single_box[2:4], box_list[:,2:4])
    
    overlap_mask = torch.logical_and(left_up_corner[:,0]<right_down_corner[:,0], left_up_corner[:,1]<right_down_corner[:,1])
    
    
    inter= (right_down_corner[:,1] - left_up_corner[:,1]) \
          * (right_down_corner[:,0] - left_up_corner[:,0])
    
    inter[~overlap_mask] = 0
    
    #print((single_box[2] - single_box[0]))
    #print((single_box[3] - single_box[1]))
    #print((box_list[:,2] - box_list[:,0]))
    #print((box_list[:,3] - box_list[:,1]))

    union = (single_box[2] - single_box[0]) * (single_box[3] - single_box[1]) \
            + (box_list[:,2] - box_list[:,0]) * (box_list[:,3] - box_list[:,1])\
            - inter
    iou = inter / union
 
    return iou'''

def contains(box_large, box_small):
    b1_xy = box_large[:2]
    b1_wh = box_large[2:4]
    #b1_wh_half = b1_wh / 2
    b1_mins = b1_xy 
    b1_maxs = b1_xy + b1_wh

    b2_xy = box_small[..., :2]
    b2_wh = box_small[..., 2:4]
    #b2_wh_half = b2_wh / 2
    b2_mins = b2_xy
    b2_maxs = b2_xy + b2_wh

    check_min = torch.logical_and( (b2_mins[:,0]-b1_mins[0])>=0, (b2_mins[:,1]-b1_mins[1])>=0 )
    check_max = torch.logical_and( (b2_maxs[:,0]-b1_maxs[0])<=0, (b2_maxs[:,1]-b1_maxs[1])<=0 )
    return torch.logical_and(check_min, check_max)

def IoU(box1, box2):
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
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxs = torch.min(b1_maxs, b2_maxs)
    intersect_wh = torch.max(intersect_maxs - intersect_mins, torch.zeros_like(intersect_maxs))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area, min=1e-6)
    return iou

def nms_merge(bboxes:torch.Tensor, scores:torch.Tensor, threshold=0.45):
    '''
    bboxes: [N, 4] (x, y, w, h)
    scores: [N, 1]
    '''
    bbox_results = []
    sort_idx = torch.argsort(scores, dim = 0, descending = True)
    bboxes = bboxes[sort_idx] 
    sum = 0
    while bboxes.shape[0] != 0:
        single_box = bboxes[0, :]
        if bboxes.shape[0] == 1:
            sum += 1
            bbox_results.append(bboxes[0, :])
            break
        bbox_list = bboxes[1:, :]
        #print(single_box.shape, bboxes[1:,:].shape)
        ious = IoU(single_box, bbox_list)
       
        overlapped = (ious > threshold)
        overlapped_bboxes = bbox_list[overlapped, :]
        sum  += overlapped_bboxes.shape[0] + 1
        merged_bboxes = torch.cat((single_box[None, ...], overlapped_bboxes), dim=0)
        final_bbox, _ = torch.median(merged_bboxes, dim=0)
        
        #final_bbox[2:4] = final_bbox[2:4] - final_bbox[0:2]
        
        bbox_results.append(final_bbox)
        bboxes = bbox_list[~overlapped, :]
   
    return bbox_results

def get_the_bbox_of_cluster(bboxes):
    '''
    bboxes: [N, 4] (xywh)
    return: (xywh)
    '''
    b1_mins = bboxes[:, 0:2]
    b1_maxs = bboxes[:, 0:2] + bboxes[:, 2:4]
    xy = torch.min(b1_mins, dim = 0)[0]
    xy2 = torch.max(b1_maxs, dim = 0)[0]
    wh = xy2 - xy
    return torch.cat((xy, wh), dim=0)

def vote_clustering(centroids, layer_rects, radius=0.001):
    '''
    params: centeroids: [N, 2]
            layer_rects: [N, 4] (xywh)
    return: [N, 4]
    description: randomly pick i_th centroid, find layers around it within ball centered at it, the radius of which is less than t
    merging these layers' bboxes to get an anchor box for each layer
    delete these layers and repeat this process until centroids is empty
    '''
    results = torch.zeros_like(layer_rects, device = layer_rects.get_device())
    prev_mask = (torch.zeros(layer_rects.shape[0], device=layer_rects.get_device()) > 1)
    while torch.sum(~prev_mask) != 0:
        centroid_mask = centroids[~prev_mask, :]
        seed = centroid_mask[0]
        dists = torch.sqrt(torch.sum((centroids - seed) ** 2, dim=1))
        cur_mask = torch.logical_and((dists < radius), ~prev_mask)
        cluster_layers = layer_rects[cur_mask, :]
        cluster_bbox = get_the_bbox_of_cluster(cluster_layers)
        results[cur_mask,: ] += cluster_bbox
        prev_mask = torch.logical_or(cur_mask, prev_mask) 
    return results
        
