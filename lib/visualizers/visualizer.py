import cv2
import torch
import torch.nn.functional as F
import os
from lib.utils import nms_merge

class visualizer:
    def __init__(self, vis_cfg):
        self.vis_dir = vis_cfg.vis_dir

    def scale_to_img(self, x, H, W):
        return (int(x[0]*W), int(x[1]*H), int(x[2]*W), int(x[3]*H))

    def visualize_with_labels(self, layer_rects, layer_idxs, img_path):
        file_path, artboard_name = os.path.split(img_path)
        artboard_name = artboard_name.split(".")[0]
        img_1 = cv2.imread(os.path.join(self.vis_dir, f'{artboard_name}-layers.png'))
        H, W, _ = img_1.shape
        for layer_rect, layer_id in zip(layer_rects, layer_idxs):
            cv2.rectangle(img_1, self.scale_to_img(layer_rect,H,W), (0,0,255),1)
            x, y, w, h = self.scale_to_img(layer_rect,H,W)
            cv2.putText(img_1, str(layer_id.item()), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, [0, 0, 255],1)

        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-correct.png'), img_1)

    def visualize_pred(self, layer_rects, local_params,  img_path):
        img_1 = cv2.imread(img_path)
        img_2 = cv2.imread(img_path)
        
        file_path, artboard_name = os.path.split(img_path)
        artboard_name = artboard_name.split(".")[0]
        H, W, _ = img_1.shape
        #print(layer_rects.shape, local_params.shape)
        for layer_rect, local_params in zip(layer_rects, local_params):
            cv2.rectangle(img_1, self.scale_to_img(layer_rect,H,W), (255,0,0),1)
            cv2.rectangle(img_2, self.scale_to_img(local_params+layer_rect,H,W),(0,255,0),1)
        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-layers.png'),img_1)
        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-group.png'),img_2)
    
    def visualize_gt(self, layer_rects, bboxes,  img_path):
        img_1 = cv2.imread(img_path)
        img_2 = cv2.imread(img_path)
        file_path, artboard_name = os.path.split(img_path)
        artboard_name = artboard_name.split(".")[0]
        H, W, _ = img_1.shape
        #print(layer_rects.shape, local_params.shape)
        for layer_rect, bbox in zip(layer_rects, bboxes):
            cv2.rectangle(img_1, self.scale_to_img(layer_rect,H,W), (255,0,0),1)
            cv2.rectangle(img_2, self.scale_to_img(bbox+layer_rect,H,W),(0,255,0),1)
        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-layers_gt.png'),img_1)
        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-group_gt.png'),img_2)

    def visualize_nms(self,  bbox_results:torch.Tensor, img_path):
        img_1 = cv2.imread(img_path)
    
        file_path, artboard_name = os.path.split(img_path)
        artboard_name = artboard_name.split(".")[0]
        H, W, _ = img_1.shape
      
        for bbox in bbox_results:
            cv2.rectangle(img_1, self.scale_to_img(bbox,H, W),(0,255,0),1)
        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-group_nms.png'),img_1)

            
        
          
