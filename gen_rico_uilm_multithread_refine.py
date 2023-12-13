# This file generate the UILM dataset based on the rico graph dataset
# We refine the bounding box labels by removing repeated bounding boxes

import os
import torch
from threading import Thread, Lock
import json
from PIL import Image, ImageDraw
from lib.config import cfg
import numpy as np
from random import choice
from lib.visualizers import visualizer
from queue import Queue
from typing import List
from tqdm import tqdm


cfg.test.batch_size = 1
mode = 'train_new'
cfg.test_dataset.rootDir = '../../dataset/rico_graph'
cfg.test_dataset.module = 'lib.datasets.light_stage.rico_graph_dataset'
cfg.test_dataset.path = 'lib/datasets/light_stage/rico_graph_dataset.py'
cfg.test_dataset.index_json = f'index_{mode}.json'
cfg.test_dataset.bg_color_mode = 'keep_alpha'
cfg.visualizer.img_folder = '/media/sda2/ljz_dataset/combined'

vis = visualizer(cfg.visualizer)

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
dataset_name = 'UILM_rico_refine'
save_img_folder = os.path.join(f"../../dataset/{dataset_name}/", mode, "images")
save_label_folder = os.path.join(f"../../dataset/{dataset_name}/", mode, "labels")
labeld_artboard_folder = os.path.join(f"../../dataset/{dataset_name}/tmp", "images")
os.makedirs(save_img_folder, exist_ok = True)
os.makedirs(save_label_folder, exist_ok = True)
os.makedirs(labeld_artboard_folder, exist_ok = True)
container_labels = [20, 21, 17, 19, 18]
label_to_str = {
                17: 0,
                18: 1,
                19: 2,
                20: 3,
                21: 4,
                }

labelstring = {
                17: 'NAVIGATION_BAR',
                18: 'TOOLBAR',
                19: 'LIST_ITEM',
                20: 'CARD_VIEW',
                21: 'CONTAINER'
              }

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

filter_artboard_ids = []
filter_artboard_ids_lock = Lock()

class processor(Thread):
    def __init__(self, cfg, thread_id, queue, pbar):
        super().__init__()
        self.thread_id = thread_id
        self.queue = queue
        self.root = cfg.test_dataset.rootDir
        self.pbar = pbar
        #self.train_list = self.train_list[:20]
    

    def read_json(self, json_path):
        # content['layer_rect'],content['edges'], content['bbox'], content['labels']
        with open(json_path, 'r') as f:
            content = json.load(f)
        return content['layer_rect'],content['edges'], content['bbox'], content['labels']


    def read_data(self, artboard_id):
        
        json_path = os.path.join(self.root, str(artboard_id), f'{artboard_id}.json')
        layer_rect, edges, bbox, labels = self.read_json(json_path)
        layer_rect = torch.FloatTensor(layer_rect) 
        edges = torch.LongTensor(edges).transpose(1,0)
        bbox = torch.FloatTensor(bbox)
        labels = torch.LongTensor(labels)
        node_indices = torch.zeros(layer_rect.shape[0], dtype = torch.int64)
        
        return layer_rect, edges, bbox, labels, node_indices, artboard_id

    def is_child(self, parent, child):
        if parent['bounds'][0] <= child['bounds'][0] and parent['bounds'][1] <= child['bounds'][1] \
            and parent['bounds'][2] + parent['bounds'][0]>= child['bounds'][2] + child['bounds'][0] and parent['bounds'][3] +  parent['bounds'][1] \
                    >= child['bounds'][3] + child['bounds'][1] :
            return True
        else:
            return False

    def build_tree(self, component_list):
        component_list.sort(key=lambda x: x['bounds'][2] * x['bounds'][3], reverse = True)
        root = component_list[0]
        for i, node in enumerate(component_list):
            if i == 0:
                continue
            for j in range(i - 1, -1, -1):
                if self.is_child(component_list[j], node):
                    component_list[j].setdefault('children', [])
                    component_list[j]['children'].append(node)
                    break
        return root

    def get_container_boxes (self, root, container_type, container_bounding_box):        
        if 'children' in root and len(root['children']) > 1 and root['class'] != -1:
            x, y, w, h = root['float_bounds']
            container_bounding_box.append([x, y, x+w, y+h])
            label_int = root['class']  
            if label_int not in container_labels:
                label_int = torch.LongTensor([21])
            container_type.append(label_int)
        if 'children' in root:
            for child in root['children']:
                self.get_container_boxes(child, container_type, container_bounding_box)


    def run(self):
        while True:
            if self.queue.empty():
                return
            artboard_id = self.queue.get()
            
            batch = self.read_data(artboard_id)

            layer_rects, edges, bbox, labels, node_indices, artboard_ids = batch
            
            artboard_img = Image.open(os.path.join(cfg.visualizer.img_folder, f'{artboard_id}.jpg')).convert("RGBA").resize(cfg.visualizer.img_size)
            W, H = artboard_img.size

            semantic_map = Image.new(artboard_img.mode, artboard_img.size, (1, 1, 1, 1))
            if not os.path.exists("color.npy"):
                color_list = random_color(10000)
            else:
                color_list = list(np.load("color.npy"))
            semantic_draw = ImageDraw.ImageDraw(semantic_map, semantic_map.mode)
            container_type = []
            container_bounding_box = []
            component_list = []
            component_list.append({'class': -1, 'bounds': [0, 0, W, H]})
            for count, layer_rect in enumerate(layer_rects.numpy().tolist()):
                x, y, w, h = int(layer_rect[0] * W), int(layer_rect[1] * H), int(layer_rect[2] * W), int(layer_rect[3] * H)
                component_list.append({'float_bounds':[layer_rect[0], layer_rect[1], 
                                                 layer_rect[2] - layer_rect[0], 
                                                 layer_rect[3] - layer_rect[1],
                                                ], 'class': labels[count],
                                                'bounds': [x, y, w - x, h - y]})
                if labels[count] not in container_labels:
                    semantic_draw.rectangle(
                        (x, y, w, h), 
                        fill = (color_list[0][count],
                                color_list[1][count],
                                color_list[2][count],
                                int(0.3 * 255)
                            ))
            
            root = self.build_tree(component_list)
            self.get_container_boxes(root, container_type, container_bounding_box)
            
            if len(container_bounding_box) ==  0:
                self.pbar.update()
                self.queue.task_done()
                filter_artboard_ids_lock.acquire()
                filter_artboard_ids.append(artboard_id)
                filter_artboard_ids_lock.release()
                continue
            # Check whether each layer are bounded by these 5 types of containers
            '''
            flag = False
            for count, layer_rect in enumerate(layer_rects.numpy().tolist()):
                x1, y1, x2, y2 = layer_rect
                if labels[count] not in container_labels:
                    for box in container_bounding_box:
                        box_x1, box_y1, box_x2, box_y2 = box
                        if not (box_x1 <= x1 and box_y1 <= y1 and box_x2 >= x2 and box_y2 >= y2):
                            flag = True
                            break
                if flag:
                    break
            if flag:
                filter_artboard_ids_lock.acquire()
                filter_artboard_ids.append(artboard_id)
                filter_artboard_ids_lock.release()

                self.pbar.update()
                self.queue.task_done()
                continue
            '''
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
            
                rect_draw.text((x, y - 10), text = labelstring[container_type[count].item()], fill = (0, 0, 0))
                rect_draw.rectangle(
                    (x, y, x1, x2), 
                    outline = (color_list[0][container_type[count] - 17],
                                color_list[1][container_type[count] - 17],
                                color_list[2][container_type[count]] - 17), width=2)

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
            self.pbar.update()
            self.queue.task_done()


def generate(max_threads):
    root = cfg.test_dataset.rootDir 
    index_json = cfg.test_dataset.index_json
    artboard_list = json.load(open(os.path.join(root, index_json),'r'))

    pbar = tqdm(total = len(artboard_list))
    artboard_queue = Queue()
    for i, id in enumerate(artboard_list):
        # if i > 12000:
        artboard_queue.put(id)

    for i in range(max_threads):
        thread = processor(cfg, i, artboard_queue, pbar)
        thread.daemon = True
        thread.start()
    artboard_queue.join()

generate(32)
print(len(filter_artboard_ids))
json.dump(filter_artboard_ids, open(f"filter_artboard_ids_{mode}.json", 'w'))