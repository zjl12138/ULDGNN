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
label_to_str = {
                            
                          
                        17: 0,
                        18: 1,
                        19: 2,
                        20: 3,
                        21: 4,
                    }

if __name__=='__main__':
    cfg.test.batch_size = 1
    mode = 'tmp'
    cfg.test_dataset.rootDir = '../../dataset/rico_graph'
    cfg.test_dataset.module = 'lib.datasets.light_stage.rico_graph_dataset'
    cfg.test_dataset.path = 'lib/datasets/light_stage/rico_graph_dataset.py'
    cfg.test_dataset.index_json = f'index_{mode}.json'
    cfg.test_dataset.bg_color_mode = 'keep_alpha'
    cfg.visualizer.img_folder = '/media/sda2/ljz_dataset/combined'

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
    save_img_folder = os.path.join("../../dataset/UILM_rico/", mode, "images")
    save_label_folder = os.path.join("../../dataset/UILM_rico/", mode, "labels")
    labeld_artboard_folder = os.path.join("../../dataset/UILM_rico/tmp", "images")
    os.makedirs(save_img_folder, exist_ok = True)
    os.makedirs(save_label_folder, exist_ok = True)
    os.makedirs(labeld_artboard_folder, exist_ok = True)
    container_labels = [20, 21, 17, 19, 18]
    for batch in tqdm(dataloader):
        assets_img, layer_rects, edges, bbox, labels, node_indices, artboard_ids = batch
        artboard_id = artboard_ids[0]
        artboard_img = Image.open(os.path.join(cfg.visualizer.img_folder, f'{artboard_id}.jpg')).convert("RGBA").resize(cfg.visualizer.img_size)
        W, H = artboard_img .size

        semantic_map = Image.new(artboard_img.mode, artboard_img.size, (1, 1, 1, 1))
        if not os.path.exists("color.npy"):
            color_list = random_color(10000)
        else:
            color_list = list(np.load("color.npy"))
        semantic_draw = ImageDraw.ImageDraw(semantic_map, semantic_map.mode)
        container_type = []
        container_bounding_box = []
        for count, layer_rect in enumerate(layer_rects.numpy().tolist()):
            x, y, w, h = int(layer_rect[0] * W), int(layer_rect[1] * H), int(layer_rect[2] * W), int(layer_rect[3] * H)
            if labels[count] not in container_labels:
                semantic_draw.rectangle(
                    (x, y, w, h), 
                    fill = (color_list[0][count],
                            color_list[1][count],
                            color_list[2][count],
                            int(0.3 * 255)
                        ))
            else:
                container_type.append(labels[count])
                container_bounding_box.append([layer_rect[0], layer_rect[1], layer_rect[2], layer_rect[3]])
        if len(container_bounding_box) ==  0:
            continue
        #semantic_map = Image.blend(artboard_img, semantic_map, alpha=0.3)
        semantic_map = Image.alpha_composite(artboard_img, semantic_map)
        file_base_name = f'{artboard_id}'
        os.makedirs(os.path.join(save_img_folder, file_base_name), exist_ok = True)
        semantic_map.save(os.path.join(save_img_folder, file_base_name, "filled.png"))
        artboard_img.save(os.path.join(save_img_folder, file_base_name, "default.png"))
        labeled_img = Image.new(artboard_img.mode, artboard_img.size, 'white')
        labeled_img.paste(artboard_img, (0, 0))
        rect_draw = ImageDraw.ImageDraw(labeled_img, labeled_img.mode)
        labels_str = ""
        for count, layer_rect in enumerate(container_bounding_box):
            x, y, x1, x2 = int(layer_rect[0] * W), int(layer_rect[1] * H), int(layer_rect[2] * W), int(layer_rect[3] * H)
            rect_draw.rectangle(
                (x, y, x1, x2), 
                outline = "red")
            x1_float, y1_float, x2_float, y2_float = layer_rect[0], layer_rect[1], layer_rect[2], layer_rect[3]
            labels_str += "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(label_to_str[container_type[count].item()], 
                                                                    (x1_float + x2_float)/2, 
                                                                    (y1_float + y2_float)/2,
                                                                    x2_float - x1_float,
                                                                    y2_float - y1_float)

        labeled_img.save(os.path.join(save_img_folder, file_base_name, "labeled.png"))
        with open(
                os.path.join(save_label_folder, file_base_name + '.txt'),
                'w+') as label_file:
            label_file.writelines(labels_str)