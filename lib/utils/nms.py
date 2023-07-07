from lib2to3.refactor import get_all_fix_names
from unittest import result
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
    b1_xy = box_large[:, :2]
    b1_wh = box_large[:, 2:4]
    #b1_wh_half = b1_wh / 2
    b1_mins = b1_xy 
    b1_maxs = b1_xy + b1_wh

    b2_xy = box_small[..., :2]
    b2_wh = box_small[..., 2:4]
    #b2_wh_half = b2_wh / 2
    b2_mins = b2_xy
    b2_maxs = b2_xy + b2_wh

    check_min = torch.logical_and( (b2_mins[:,0]-b1_mins[:, 0])>=0, (b2_mins[:,1]-b1_mins[:, 1])>=0 )
    check_max = torch.logical_and( (b2_maxs[:,0]-b1_maxs[:, 0])<=0, (b2_maxs[:,1]-b1_maxs[:, 1])<=0 )
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
    iou = intersect_area / torch.clamp(union_area, min=1e-10)
    return iou

def nms_merge(bboxes:torch.Tensor, scores:torch.Tensor, threshold=0.4, mode = 'mean'):
    '''
    bboxes: [N, 4] (x, y, w, h)
    scores: [N, 1]
    '''
    bbox_results = []
    sort_idx = torch.argsort(scores, dim = 0, descending = True)
    confidence = scores[sort_idx].unsqueeze(1)
    bboxes = bboxes[sort_idx] 
    sum = 0
    confidence_results = []
    while bboxes.shape[0] != 0:
        single_box = bboxes[0, :]
        if bboxes.shape[0] == 1:
            sum += 1
            bbox_results.append(bboxes[0, :])
            confidence_results.append(confidence[0])
            break
        bbox_list = bboxes[1:, :]
        confidence_list = confidence[1:]
        #print(single_box.shape, bboxes[1:,:].shape)
        ious = IoU(single_box, bbox_list)
       
        overlapped = (ious >= threshold)
        overlapped_bboxes = bbox_list[overlapped, :]
        overlapped_bboxes_confidence = confidence_list[overlapped, :]
        sum  += overlapped_bboxes.shape[0] + 1
        merged_bboxes = torch.cat((single_box[None, ...], overlapped_bboxes), dim = 0)
        merged_confidence = torch.cat((confidence[0 : 1, :], overlapped_bboxes_confidence), dim = 0)
        
        if mode == 'median':
            final_bbox, _ = torch.median(merged_bboxes, dim=0)
        elif mode == 'mean':
            final_bbox = torch.mean(merged_bboxes, dim = 0)
        elif mode == 'max':
            max_idx = torch.argsort(merged_confidence, dim = 0, descending = True)[0]
            final_bbox = merged_bboxes[max_idx]
            
        if mode == 'median':
            confidence_results.append(torch.median(merged_confidence))
        elif mode == 'mean':
            confidence_results.append(torch.mean(merged_confidence))
        
        #final_bbox[2:4] = final_bbox[2:4] - final_bbox[0:2]
        
        bbox_results.append(final_bbox)
        bboxes = bbox_list[~overlapped, :]
        confidence = confidence_list[~overlapped]
   
    return bbox_results, confidence_results

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
    prev_mask = (torch.zeros(layer_rects.shape[0], device = layer_rects.get_device()) > 1)
    while torch.sum(~prev_mask) != 0:
        centroid_mask = centroids[~prev_mask, :]
        seed = centroid_mask[0]
        dists = torch.sqrt(torch.sum((centroids - seed) ** 2, dim = 1))
        cur_mask = torch.logical_and((dists < radius), ~prev_mask)
        cluster_layers = layer_rects[cur_mask, :]
        cluster_bbox = get_the_bbox_of_cluster(cluster_layers)
        results[cur_mask,: ] += cluster_bbox
        prev_mask = torch.logical_or(cur_mask, prev_mask) 
    return results

def vote_clustering_each_layer(centroids, layer_rects, radius=0.001):
    '''
    params: centeroids: [N, 2]
            layer_rects: [N, 4] (xywh)
    return: [N, 4]
    description: randomly pick i_th centroid, find layers around it within ball centered at it, the radius of which is less than t
                 merging these layers' bboxes to get an anchor box for each layer
                 delete these layers and repeat this process until centroids is empty
    '''
    results = torch.zeros_like(layer_rects, device = layer_rects.get_device())
    for i in range(layer_rects.shape[0]):
        seed = centroids[i]
        dists = torch.sqrt(torch.sum((centroids - seed) ** 2, dim = 1))
        cur_mask = (dists < radius)
        cluster_layers = layer_rects[cur_mask, :]
        cluster_bbox = get_the_bbox_of_cluster(cluster_layers)
        results[i, :] += cluster_bbox
    return results

def contains_how_much(box1, box2):
    b1_xy = box1[..., :2]
    b1_wh = box1[..., 2:4]
    #b1_wh_half = b1_wh / 2
    b1_mins = b1_xy 
    b1_maxs = b1_xy + b1_wh

    b2_xy = box2[..., :2]
    b2_wh = box2[..., 2 : 4]
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
    iou = intersect_area / torch.clamp(b2_area, min=1e-10)
    return iou

def merging_components(merging_groups_pred_nms,  layer_rects, pred_labels, merging_groups_confidence = None):
   
    areas = merging_groups_pred_nms[:, 2] * merging_groups_pred_nms[:, 3]
    sort_idx = torch.argsort(areas, descending = False)
    merging_groups_pred_nms = merging_groups_pred_nms[sort_idx]  
    results = []
    prev_mask = (torch.zeros(layer_rects.shape[0], device = layer_rects.device) > 1)
    for merging_rect in merging_groups_pred_nms:
        iou = contains_how_much(merging_rect.unsqueeze(0), layer_rects)
        cur_mask = torch.logical_and(~prev_mask, iou > 0.7) & (pred_labels == 1)
        indices = torch.where(cur_mask)[0]
        if indices.shape[0] > 1:
            results.append(indices)
        prev_mask = torch.logical_or(prev_mask, iou > 0.7)
    return results

def refine_merging_bbox(merging_groups_pred_nms, layer_rects, pred_labels, confidence, categories):
    '''
    @params:
        merging_groups_pred_nms: (N, 4)
        layer_rects: (N, 4)
    '''
    areas = merging_groups_pred_nms[:, 2] * merging_groups_pred_nms[:, 3]
    sort_idx = torch.argsort(areas, descending = False)
    merging_groups_pred_nms = merging_groups_pred_nms[sort_idx]  
    results = []
    confidence_list = []
    prev_mask = (torch.zeros(layer_rects.shape[0], device = layer_rects.get_device()) > 1)
    for merging_rect, confid in zip(merging_groups_pred_nms, confidence):
        iou = contains_how_much(merging_rect.unsqueeze(0), layer_rects)
        #cur_mask = torch.logical_and(~prev_mask, iou > 0.7) & (pred_labels == 1)
        cur_mask = torch.logical_and(~prev_mask, iou > 0.7) & torch.logical_or((pred_labels == 1), categories <= 5)

        indices = torch.where(cur_mask)[0]
        if indices.shape[0]:
            results.append(get_the_bbox_of_cluster(layer_rects[indices, :]))
            confidence_list.append(confid)
        prev_mask = torch.logical_or(prev_mask, iou > 0.7)
    if len(results) == 0:
        return results, confidence_list
    return results, torch.hstack(confidence_list)

def get_pred_adj(merging_list, n, device):
    '''
    @params:
        n is the total number of layers in this artboard
        merging_list take the form: [[0, 1, 2...], [3, 4, ... ], ...] meaning that layer_0, 1, 2 form a merging group and layer_3, 4 form the merging group
    @output:
        graph: n * n matrix, for the simple example above, the matrix is like
        [1, 1, 1, 0, 0, ...]
        [1, 1, 1, 0, 0, ...]
        [1, 1, 1, 0, 0, ...]
        [0, 0, 0, 1, 1, ...]
        [0, 0, 0, 1, 1, ...]
        [       .          ]
        [       .          ]
        [       .          ]
    '''
    graph = [[1 if i == j else 0 for i in range(n)] for j in range(n)]
    for merging_group in merging_list:
        for i in merging_group:
            for j in merging_group:
                graph[i][j] = 1
    return torch.tensor(graph, dtype = torch.int64, device = device)

def get_gt_adj(bboxes, labels_gt):
    n = bboxes.shape[0]
    graph = [[0 for i in range(n)] for i in range(n)]
    for idx, bbox in enumerate(bboxes):
        jdx = idx + 1
        graph[idx][idx] = 1
        if labels_gt[idx] == 0:
            continue
        while jdx < n:
            if labels_gt[idx] == 0:
                continue
            if torch.sum(torch.abs(bboxes[idx, :] - bboxes[jdx, :])) <= 1e-6:
                graph[idx][jdx] = 1
                graph[jdx][idx] = 1
            jdx += 1
    return torch.tensor(graph, dtype = torch.int64, device = bboxes.get_device() if bboxes.get_device() >= 0 else torch.device("cpu"))

def get_comp_gt_list(bboxes, labels_gt):
    fragmented_layer_idxs = torch.where(labels_gt)[0]
    comp_gt_list = []
    visited = torch.zeros(fragmented_layer_idxs.shape[0])
    for idx in range(fragmented_layer_idxs.shape[0]):
        if visited[idx] == 0:
            cur_idx = fragmented_layer_idxs[idx]
            visited[idx] = 1
            tmp = []
            for jdx in fragmented_layer_idxs:
                if torch.sum(torch.abs(bboxes[cur_idx, :] - bboxes[jdx, :])) <= 1e-6:
                    tmp.append(jdx)
            assert(cur_idx in tmp)
            assert(len(tmp) > 0)
            if len(tmp) >= 1:
                comp_gt_list.append(torch.LongTensor(tmp))
    return comp_gt_list

def get_gt_adj_vectorize(bboxes, labels_gt):
    N = bboxes.shape[0]
    similarity_matrix = torch.sum(torch.abs(bboxes - bboxes[:, None, :].repeat(1, N, 1)), dim = 2) # size [N, N]
    mask = similarity_matrix <= 1e-6
    contrasitive_labels = torch.zeros_like(similarity_matrix, dtype = torch.int64, device = similarity_matrix.device)
    contrasitive_labels[mask] = 1
    label_idx_mask_1 = (labels_gt == 0).reshape(N, 1) & (labels_gt == 1).reshape(1, N)
    label_idx_mask_2 = (labels_gt == 1).reshape(N, 1) & (labels_gt == 0).reshape(1, N)
    label_idx_mask = torch.logical_or(label_idx_mask_1, label_idx_mask_2)
    label_idx_mask_0 = (labels_gt == 0).reshape(N, 1) & (labels_gt == 0).reshape(1, N)
    contrasitive_labels[label_idx_mask] = 0 # we need make sure labels==0 is not related to labels == 1
    contrasitive_labels[label_idx_mask_0] = 0
    return contrasitive_labels

def get_comp_gt_list_vectorize(contrasitive_labels, labels_gt):
    
    fragmented_idx = torch.where(labels_gt == 1)[0]
    results = []
    for i in range(fragmented_idx.shape[0]):
        merging_tensor = torch.where(contrasitive_labels[fragmented_idx[i], :] != 0)[0]
        #if merging_tensor[merging_tensor.shape[0] - 1] >= fragmented_idx[i]:
        results.append(merging_tensor)
    return results
    