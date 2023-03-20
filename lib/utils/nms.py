import torch

def IoU(single_box, box_list):
    left_up_corner = torch.maximum(single_box[0:2],box_list[:,0:2])
    right_down_corner = torch.minimum(single_box[2:4], box_list[:,2:4])
    inter=(right_down_corner[:,1]-left_up_corner[:,1])*(right_down_corner[:,0]-left_up_corner[:,0])
    union = (single_box[2]  -single_box[0]) * (single_box[3] - single_box[1]) \
            + (box_list[:,2]-box_list[:,0])*(box_list[:,3]-box_list[:,1])\
            - inter
    iou = inter / union
    return iou

def nms_merge(bboxes:torch.Tensor, scores:torch.Tensor, threshold=0.45):
    '''
    bboxes: [N, 4] (x1,y1,x2,y2)
    scores: [N, 1]
    '''
    bbox_results = []
    sort_idx = torch.argsort(scores,dim=0,descending=True)
    
    bboxes = bboxes[sort_idx]
    sum = 0
   
    while bboxes.shape[0]!=0:
        single_box = bboxes[0,:]
        if bboxes.shape[0]==1:
            sum += 1
            bbox_results.append(bboxes[0,:])
            break
        #print(single_box.shape, bboxes[1:,:].shape)
        ious = IoU(single_box, bboxes[1:,:])
        overlapped = (ious > threshold)
        #print(overlapped.all())
        overlapped_bboxes = bboxes[1:,:][overlapped,:]
        sum  += overlapped_bboxes.shape[0]+1
        merged_bboxes = torch.cat((single_box[None,...], overlapped_bboxes),dim=0)
        final_bbox, _ = torch.median(merged_bboxes,dim=0)
        
        final_bbox[2:4] = final_bbox[2:4] - final_bbox[0:2]
        bbox_results.append(final_bbox)
        bboxes = bboxes[1:,:][~overlapped,:]
   
    return bbox_results