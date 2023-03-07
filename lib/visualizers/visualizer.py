import cv2
import torch
import torch.nn.functional as F
import os

class visualizer:
    def __init__(self, record_cfg):
        self.vis_dir = record_cfg.vis_dir

    def scale_to_img(self, x, H, W):
        return (int(x[0]*W), int(x[1]*H), int(x[3]*W), int(x[4]*H))

    def visualize(self, logits, layer_rects, local_params, img_path):
        img_1 = cv2.imread(img_path)
        img_2 = cv2.imread(img_path)
        artboard_name = os.path.split(img_path).split(".")[0]
        H, W, _ = img_1.shape
        pred = torch.max(F.softmax(logits,dim=1), 1)[1]
        layer_rects = layer_rects[pred==1]
        local_params = local_params[pred==1]
        for layer_rect, local_params in zip(layer_rects, local_params):
            cv2.rectangle(img_1, self.scale_to_img(layer_rect,H,W), (255,0,0),1)
            cv2.rectangle(img_2, self.scale_to_img(local_params,H,W),(0,255,0),1)
        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-layers.png'),img_1)
        cv2.imwrite(os.path.join(self.vis_dir, f'{artboard_name}-group.png'),img_2)
        
