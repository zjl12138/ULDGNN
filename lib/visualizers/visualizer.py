import cv2
import torch
import torch.nn.functional as F
import os
from lib.utils import nms_merge
from lib.utils.nms import contains, refine_merging_bbox
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
def clip_val(x, lower, upper):
    x = x if x >= lower else lower
    x = x if x <= upper else upper
    return x

class visualizer:
    def __init__(self, vis_cfg):
        self.vis_dir = vis_cfg.vis_dir

    def scale_to_img(self, x, H, W):
        x[0] = clip_val(x[0], 0, 1)
        x[1] = clip_val(x[1], 0, 1)
        x[2] = clip_val(x[2], 0, 1)
        x[3] = clip_val(x[3], 0, 1)
        return (int(x[0] * W), int(x[1] * H), int(x[2] * W), int(x[3] * H))
        #return ((x[0] * W), (x[1] * H), (x[2] * W), (x[3] * H))
    
    def point_scaled_to_img(self, x, H, W):
        return (int(x[0] * W), int(x[1] * H))

    def visualize_with_labels(self, layer_rects, layer_idxs, img_path):
        file_path, artboard_name = os.path.split(img_path)
        artboard_name = artboard_name.split(".")[0]
        img_1 = cv2.imread(os.path.join(self.vis_dir, f'{artboard_name}-layers.png'))

        H, W, _ = img_1.shape
        for layer_rect, layer_id in zip(layer_rects, layer_idxs):
            cv2.rectangle(img_1, self.scale_to_img(layer_rect, H, W), (0,0,255), 1)
            x, y, w, h = self.scale_to_img(layer_rect, H, W) 
            cv2.putText(img_1, str(layer_id.item()), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, [0, 0, 255],1)
        #return ToTensor()(img_1)
        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-correct.png'), img_1)

    def visualize_pred_fraglayers(self, fragmented_layers_pred, img_path):
        img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        file_path, artboard_name = os.path.split(img_path)
        artboard_name = artboard_name.split(".")[0]
        H, W, _ = img_1.shape
        #print(layer_rects.shape, local_params.shape)
        for layer_rect in fragmented_layers_pred:
            cv2.rectangle(img_1, self.scale_to_img(layer_rect, H, W), (255, 0, 0), 1)

        return ToTensor()(img_1)

    def visualize_pred(self, fragmented_layers_pred, merging_groups_pred, img_path, save_file = True):
        img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_2 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        file_path, artboard_name = os.path.split(img_path)
        artboard_name = artboard_name.split(".")[0]
        H, W, _ = img_1.shape
        #print(layer_rects.shape, local_params.shape)
        for layer_rect, pred_bbox in zip(fragmented_layers_pred, merging_groups_pred):
            cv2.rectangle(img_1, self.scale_to_img(layer_rect, H, W), (255, 0, 0), 1)
            cv2.rectangle(img_2, self.scale_to_img(pred_bbox, H, W), (0, 255, 0), 1)
        
        if not save_file:
            return (ToTensor()(img_1), ToTensor()(img_2))

        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-layers.png'), img_1)
        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-group.png'), img_2)
    
    def visualize_gt(self, fragmented_layers_gt, merging_groups_gt, img_path, save_file = True):
        img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_2 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        file_path, artboard_name = os.path.split(img_path)
        artboard_name = artboard_name.split(".")[0]
        H, W, _ = img_1.shape
        #print(layer_rects.shape, local_params.shape)
        for layer_rect, bbox in zip(fragmented_layers_gt, merging_groups_gt):
            cv2.rectangle(img_1, self.scale_to_img(layer_rect, H, W), (255, 0, 0), 1)
            cv2.rectangle(img_2, self.scale_to_img(bbox, H, W), (0, 255, 0), 1)
        if not save_file:
            return (ToTensor()(img_1), ToTensor()(img_2))
        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-layers_gt.png'), img_1)
        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-group_gt.png'), img_2)

    def visualize_nms(self,  bbox_results:torch.Tensor, img_path, save_file = True):
        img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
        file_path, artboard_name = os.path.split(img_path)
        artboard_name = artboard_name.split(".")[0]
        H, W, _ = img_1.shape
      
        for bbox in bbox_results:
            cv2.rectangle(img_1, self.scale_to_img(bbox, H, W), (0, 255, 0), 1)
        if not save_file:
            return ToTensor()(img_1)
        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-group_nms.png'), img_1)

    def draw_label_type(self, image, bbox, label):
        
        labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        #if bbox[1] - labelSize
        x, y, w, h = bbox
        color = (0, 255, 0) # 绿色
        thickness = 1
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        # 添加目标类别和置信度等信息
        font_scale = 0.25
        font_thickness = 1
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_origin = (x, y - int(label_h))
        cv2.rectangle(image, (text_origin[0], text_origin[1] - label_h),
                    (text_origin[0] + label_w, text_origin[1] + int(0.5 * label_h)), color, -1)
        cv2.putText(image, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        
    def visualize_nms_with_labels(self,  bbox_results:torch.Tensor, confidence:torch.Tensor, img_path, mode = 'group_nms', save_file = True):
        img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        origin_img = img_1
        file_path, artboard_name = os.path.split(img_path)
        artboard_name = artboard_name.split(".")[0]
        H, W, _ = img_1.shape
      
        for bbox, confid in zip(bbox_results, confidence):
            self.draw_label_type(img_1, self.scale_to_img(bbox, H, W), '{:.2f}'.format(confid.item()))
            #cv2.rectangle(img_1, self.scale_to_img(bbox, H, W), (0, 255, 0), 1)
        if not save_file:
            out_img = cv2.addWeighted(origin_img, 0.8, img_1, 0.2, 0)
        return ToTensor()(out_img)
        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-{mode}.png'), out_img)
    
    def visualize_offset_of_centers(self, centers: torch.Tensor, layer_rects: torch.Tensor, img_path, save_file = True):
        layer_center = layer_rects[:, 0:2] + 0.5 * layer_rects[:, 2:4]
        img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
        file_path, artboard_name = os.path.split(img_path)
        artboard_name = artboard_name.split(".")[0]
        H, W, _ = img_1.shape
      
        for start, end in zip(layer_center, centers):
            cv2.arrowedLine(img_1, self.point_scaled_to_img(start, H, W), self.point_scaled_to_img(end, H, W), (0, 255, 0), 1) 
        if  not save_file:
            return ToTensor()(img_1)
        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-center_offset.png'), img_1)
 
    def remove_files(self):
        os.system(f"rm -f {self.vis_dir}/*.png")

    def visualize_overall(self, fetch_data, layer_rects, types, img_path):
        fragmented_layers_gt = fetch_data['fragmented_layers_gt']
        fragmented_layers_pred = fetch_data['fragmented_layers_pred']
        merging_groups_gt = fetch_data['merging_groups_gt']
        merging_groups_pred = fetch_data['merging_groups_pred']
        merging_groups_pred_nms = fetch_data['merging_groups_pred_nms']
        merging_groups_confidence = fetch_data['merging_groups_confidence']
        centers_pred = fetch_data['centers_pred']
        label_pred = fetch_data['label_pred']
        layer_pred_refine_img = self.visualize_pred_fraglayers(layer_rects[label_pred == 1], img_path)
        fragmented_layers_pred_img, bbox_pred_img =  self.visualize_pred(fragmented_layers_pred, merging_groups_pred, img_path, save_file = False)
                    #visualizer.visualize_nms(merging_groups_pred_nms, img_path)
        bbox_nms_pred_img = self.visualize_nms_with_labels(merging_groups_pred_nms, merging_groups_confidence, img_path, save_file = False)
        
        bbox_refine_img = None
        if len(merging_groups_pred_nms) != 0:
            merging_groups_pred_nms, merging_groups_confidence = refine_merging_bbox(torch.vstack(merging_groups_pred_nms), layer_rects, label_pred, merging_groups_confidence, types)
            bbox_refine_img = self.visualize_nms_with_labels(merging_groups_pred_nms, merging_groups_confidence, img_path, mode = "bbox_refine", save_file = False)
            #det_results.append([np.vstack([scale_to_img(t.cpu().numpy(), H, W) for t in merging_groups_pred_nms])])
            
        fragmented_layers_gt_img, bbox_gt_img = self.visualize_gt(fragmented_layers_gt, merging_groups_gt, img_path, save_file = False)
        center_pred_img = self.visualize_offset_of_centers(centers_pred, fragmented_layers_pred, img_path, save_file = False)

        if bbox_refine_img is not None:
            bbox_compare_img = torch.cat((bbox_pred_img, bbox_nms_pred_img, bbox_refine_img, bbox_gt_img), dim = 2)
        else:
            bbox_compare_img = torch.cat((bbox_pred_img, bbox_nms_pred_img, bbox_gt_img), dim = 2)
        
        layer_compare_img = torch.cat((center_pred_img, fragmented_layers_pred_img, layer_pred_refine_img, fragmented_layers_gt_img), dim = 2)
        
        file_path, artboard_name = os.path.split(img_path)
        save_image(bbox_compare_img, os.path.join(self.vis_dir, f'{artboard_name}-bbox_compare.png'))
        save_image(layer_compare_img, os.path.join(self.vis_dir, f'{artboard_name}-layer_compare.png'))
          
