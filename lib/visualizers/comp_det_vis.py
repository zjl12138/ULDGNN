
import cv2
import torch
import torch.nn.functional as F
import os
from torchvision.transforms.transforms import ToPILImage
from lib.utils import nms_merge
from lib.utils.nms import contains, refine_merging_bbox
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from PIL import Image, ImageDraw

def clip_val(x, lower, upper):
    x = x if x >= lower else lower
    x = x if x <= upper else upper
    return x

class comp_det_visualizer:
    def __init__(self, vis_cfg):
        print('[MESSAGE] Component Detection Visualizer init!')
        self.vis_dir = vis_cfg.vis_dir
        self.img_size = vis_cfg.img_size
        self.img_folder = vis_cfg.img_folder
        self.label_to_str = {-1: 'ROOT',
                            0: 'BACKGROUND',
                            1: 'IMAGE',
                            2: 'PICTOGRAM',
                            3: 'BUTTON',
                            4: 'TEXT',
                            5: 'LABEL',
                            6: 'TEXT_INPUT',
                            7: 'MAP',
                            8: 'CHECK_BOX',
                            9: 'SWITCH',
                            10: 'PAGER_INDICATOR',
                            11: 'SLIDER',
                            12: 'RADIO_BUTTON',
                            13: 'SPINNER',
                            14: 'PROGRESS_BAR',
                            15: 'ADVERTISEMENT',
                            16: 'DRAWER',
                            17: 'NAVIGATION_BAR',
                            18: 'TOOLBAR',
                            19: 'LIST_ITEM',
                            20: 'CARD_VIEW',
                            21: 'CONTAINER',
                            22: 'DATE_PICKER',
                            23: 'NUMBER_STEPPER'}
    def scale_to_img(self, x):
        W, H = self.img_size
        x[0] = clip_val(x[0], 0, 1)
        x[1] = clip_val(x[1], 0, 1)
        x[2] = clip_val(x[2], 0, 1)
        x[3] = clip_val(x[3], 0, 1)
        return (int(x[0] * W), int(x[1] * H), int(x[2] * W), int(x[3] * H))
    
    def visualize_layer(self, layer_rects, pred_label, artboard_id):
        img_1 = Image.open(os.path.join(self.img_folder, f'{artboard_id}.jpg')).resize(self.img_size)
        # draw rectangles in the image
        img_draw = ImageDraw.Draw(img_1)       
        for layer_rect, layer_id in zip(layer_rects, pred_label):
            img_draw.rectangle(self.scale_to_img(layer_rect), outline=(0, 0, 255), width=1)
            x, y, w, h = self.scale_to_img(layer_rect)
            img_draw.text((x, y), self.label_to_str[layer_id.cpu().item()], fill=(0, 0, 255))
        return ToTensor()(img_1)
    
    def visualize_box(self, boxes, artboard_id):
        img_1 = Image.open(os.path.join(self.img_folder, f'{artboard_id}.jpg')).resize(self.img_size)
        # draw rectangles in the image
        img_draw = ImageDraw.Draw(img_1)       
        for box in boxes:
            img_draw.rectangle(self.scale_to_img(box), outline=(0, 255, 0), width=1)
            
        return ToTensor()(img_1)
    
    def visualize_overall(self, layer_rect, pred_label, labels, pred_bboxes, gt_bboxes, artboard_id):
        comp_pred_img = self.visualize_layer(layer_rect, pred_label, artboard_id)
        comp_pred_gt = self.visualize_layer(layer_rect, labels, artboard_id)
        group_pred_img = self.visualize_box(pred_bboxes, artboard_id)
        group_gt_img = self.visualize_box(gt_bboxes, artboard_id)
        compare_img = torch.cat((comp_pred_img, comp_pred_gt, group_pred_img, group_gt_img), dim = 2)
        save_image(compare_img, os.path.join(self.vis_dir, f'{artboard_id}-compare.png'))
        