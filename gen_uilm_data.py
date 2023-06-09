from tkinter.tix import ExFileSelectBox
import math
from scipy.sparse import data
from lib.utils.nms import nms_merge
from lib.datasets import make_data_loader
from lib.config import cfg
from lib.visualizers import visualizer
from lib.networks import make_network
import torch
from lib.train import make_optimizer, make_recorder, make_scheduler, make_trainer
from lib.evaluators import Evaluator
from lib.utils import load_model, save_model, get_pred_adj, merging_components, get_gt_adj
from tqdm import tqdm
from torch_geometric.utils import to_dense_batch
import os
from typing import List
from sklearn.model_selection import train_test_split
import json
from PIL import Image, ImageDraw
import random
from random import choice
from typing import List, Tuple, AnyStr
import numpy as np

def random_color(n_colors):
    print('generating random color list')
    r_list, g_list, b_list, color_list = [], [], [], []
    levels = range(32, 256)
    for _ in range(n_colors):
        red, green, blue = tuple(choice(levels) for _ in range(3))
        r_list.append(red)
        g_list.append(green)
        b_list.append(blue)
    color_list.append(r_list)
    color_list.append(g_list)
    color_list.append(b_list)
    np.save('color.npy', color_list)
    return color_list


if __name__=='__main__':
    cfg.test.batch_size = 1
    mode = 'testv2'
    cfg.test_dataset.rootDir = '../../dataset/EGFE_graph_dataset'
    cfg.test_dataset.index_json = f'index_{mode}.json'
    cfg.test_dataset.bg_color_mode = 'keep_alpha'
    dataloader = make_data_loader(cfg, is_train=False)
    vis = visualizer(cfg.visualizer)
    evaluator = Evaluator()
    print(len(dataloader))
    #trainer.train(0, dataloader, optim, recorder, evaluator )
    positives = 0
    negatives = 0
    num_of_fragments: List = []
    zeros = 0
    merge_acc = 0.0
    precision = 0.0
    recall = 0.0
    acc = 0.
    labels_pred = []
    labels_gt = []
    merge_recall = 0.0
    merge_precision = 0.0 
    save_img_folder = os.path.join("../../dataset/UILM_from_egfe_resplit/", mode, "images")
    save_label_folder = os.path.join("../../dataset/UILM_from_egfe_resplit/", mode, "labels")
    labeld_artboard_folder = os.path.join("../../dataset/UILM_from_egfe_resplit/tmp", "images")
    os.makedirs(save_img_folder, exist_ok = True)
    os.makedirs(save_label_folder, exist_ok = True)
    os.makedirs(labeld_artboard_folder, exist_ok = True)
    for batch in tqdm(dataloader):
        #network(batch)
        layer_rects_, edges, types, img_tensor, labels, bboxes, layer_rects, node_indices, file_list  = batch
        #print(node_indices)
        if torch.sum(labels) == 0:
           continue
        file_path, artboard_name = os.path.split(file_list[0])
        artboard_name = artboard_name.split(".")[0]

        artboard_img = Image.open(file_list[0]).convert("RGBA")
        semantic_map = Image.new(artboard_img.mode, artboard_img.size, 'white')
        
        if not os.path.exists("color.npy"):
            color_list = random_color(10000)
        else:
            color_list = list(np.load("color.npy"))
        W, H = artboard_img.size
        semantic_draw = ImageDraw.ImageDraw(semantic_map, semantic_map.mode)
        
        for count, layer_rect in enumerate(layer_rects.numpy().tolist()):
            x, y, w, h = int(layer_rect[0] * W), int(layer_rect[1] * H), int(layer_rect[2] * W), int(layer_rect[3] * H)
            semantic_draw.rectangle(
                (x, y, x + w, y + h), 
                fill = (color_list[0][count],
                        color_list[1][count],
                        color_list[2][count],
                        int(0.3 * 255)
                     ))
        #semantic_map.save(os.path.join(save_img_folder, f"{artboard_name}-filled.png"))
        semantic_map = Image.alpha_composite(artboard_img, semantic_map)
        #.save(os.path.join(save_img_folder, f"{artboard_name}-default.png"))

        bboxes = bboxes + layer_rects
        bboxes = bboxes[labels == 1]
        
        bboxes, _ = nms_merge(bboxes, torch.ones(bboxes.shape[0]))
        bboxes = [[int(bbox[0].item() * W), int(bbox[1].item() * H), 
                   int(bbox[2].item() * W + bbox[0].item() * W), int(bbox[3].item() * H + bbox[1].item() * H)] for bbox in bboxes]
    
        '''labeled_artboard_image = Image.new(
                            artboard_img.mode,
                            artboard_img.size, "white")
                
        labeled_artboard_image.paste(artboard_img, (0, 0))
                
        artboard_label_draw = ImageDraw.ImageDraw(labeled_artboard_image, labeled_artboard_image.mode)
        for point in bboxes:
            artboard_label_draw.rectangle(point, outline = 'red')
        labeled_artboard_image.save(os.path.join(labeld_artboard_folder, artboard_name+".png"))
        '''
        single_image_width = W
        single_image_height = int(W / 1)
        # calculate image need to process
        total = H / single_image_height
        box_range: List[Tuple[float, float]] = [
            (0, float(y)) for y in np.arange(0, total, step=0.5)
        ]
        if total < 1:
            # re-calculate split images' width and height
            single_image_width = int(H * 1)
            single_image_height = H
            # calculate image need to process
            total = W / single_image_width
            box_range = [(float(x), 0)
                            for x in np.arange(0, total, step=0.5)]
        width = W
        height = H
        for (x, y) in box_range:
                # calculate crop box in order: (left, top, right, bottom)
            img_box = (math.floor(single_image_width * x),
                        math.floor(single_image_height * y),
                        math.floor(min(single_image_width * (x + 1),
                                        width)),
                        math.floor(
                            min(single_image_height * (y + 1), height)))
            file_base_name = f'{artboard_name}-1-{x}-{y}'
            labels = list(
                    # filter label in crop box
                    filter(
                        lambda target_label:
                        img_box[0] <= target_label[0]  and img_box[1] <= target_label[1] and
                        img_box[2] >= target_label[2] and img_box[3] >= target_label[3],
                        bboxes
                    ))
            if len(labels) > 0:
                os.makedirs(os.path.join(save_img_folder, file_base_name), exist_ok = True)
                
                labeled_image = Image.new(
                            artboard_img.mode,
                            (single_image_width, single_image_height), "white")
                crop_img_box = artboard_img.crop(img_box)
                labeled_image.paste(crop_img_box, (0, 0))
                
                artboard_label_draw = ImageDraw.ImageDraw(labeled_image, labeled_image.mode)
                for point in labels:
                    label_bbox = np.array(point) - np.array([math.floor(single_image_width * x), math.floor(single_image_height * y),
                                                            math.floor(single_image_height * x), math.floor(single_image_height * y)])
                    artboard_label_draw.rectangle(label_bbox.tolist(), outline = 'red')
                labeled_image.save(os.path.join(os.path.join(save_img_folder, file_base_name), 'labeled.png'))
                
                labels = list(
                        map(
                            # convert (left, top, right, bottom) to (x_center, y_center, width, height) format
                            # remember to multiply scale to match real pixel
                            lambda points: " ".join(["0"] + list(
                                map(
                                    lambda float_value: "{:.6f}".format(
                                        float_value),
                                    [((points[0] + points[2]) / 2 
                                      - math.floor(single_image_width * x) ) /
                                     (single_image_width ),
                                     ((points[1] + points[3]) / 2 
                                      - math.floor(single_image_height * y)) /
                                     (single_image_width ),
                                     ((points[2] - points[0])) /
                                     (single_image_width ),
                                     ((points[3] - points[1]) ) /
                                     (single_image_width)]))) + "\n",
                            labels))
                
                filled_image = Image.new(
                        semantic_map.mode,
                        (single_image_width, single_image_height), "white")
                filled_image.paste(semantic_map.crop(img_box),
                                    (0, 0))
                filled_image.save(
                    os.path.join(os.path.join(save_img_folder, file_base_name), 'filled.png'))
                
                default_image = Image.new(
                            artboard_img.mode,
                            (single_image_width, single_image_height), "white")
                crop_img_box = artboard_img.crop(img_box)
                default_image.paste(crop_img_box, (0, 0))
                default_image.save(os.path.join(save_img_folder, file_base_name, 'default.png'))
                
                with open(
                            os.path.join(save_label_folder, file_base_name + '.txt'),
                            'w+') as label_file:
                        label_file.writelines(labels)