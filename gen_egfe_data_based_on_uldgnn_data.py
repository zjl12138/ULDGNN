
from scipy.sparse import data
from lib.utils.nms import get_the_bbox_of_cluster, nms_merge
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
import cv2
import numpy as np
import random

def get_merging_components_transformer(pred_label : torch.Tensor):
    N = pred_label.shape[0]
    merging_list = []
    tmp = []
    for i in range(N):
        if pred_label[i] == 2:
            if len(tmp) == 0:
                tmp.append(i)
            else:
                merging_list.append(torch.LongTensor(tmp))
                tmp = []
                tmp.append(i)
        elif pred_label[i] == 1:
            tmp.append(i)
        else:
            continue
    if len(tmp) > 0:
        merging_list.append(torch.LongTensor(tmp))
    return merging_list

def deal_with_class_(str):
    str = str[:1].lower() + str[1:]
    if str == 'hotspot':
        str = 'MSImmutableHotspotLayer'
    return str

LAYER_CLASS_MAP = {
    'Oval': 0,
    'Polygon': 1,
    'Rectangle': 2,
    'ShapePath': 3,
    'Star': 4,
    'Triangle': 5,
    'Text': 6,
    'SymbolInstance': 7,
    'Slice': 8,
    'Hotspot': 9,
    'Bitmap': 10
}
def process_bbox_result():
    bbox_dict = json.load(open("bbox_result_uldgraph.json"))
    new_bbox_dict = {}
    for key in bbox_dict.keys():
        artboard_name, _, x_window, y_window = key.split("/")[0].split("-")
        new_bbox_dict.setdefault(artboard_name, [])
        new_bbox_dict[artboard_name].append({x_window + "-" + y_window : bbox_dict[key]})
    json.dump(new_bbox_dict, open("new_bbox_result_uldgraph.json", "w"))
    return new_bbox_dict

if __name__=='__main__':
    cfg.test_dataset.module = 'lib.datasets.light_stage.graph_dataset_new'
    cfg.test_dataset.path = 'lib/datasets/light_stage/graph_dataset_new.py'
    
    cfg.test.batch_size = 1
    cfg.test_dataset.rootDir = '../../dataset/ULDGNN_dataset'
    cfg.test_dataset.index_json = 'index_test_based_on_sketch.json'
    cfg.test_dataset.bg_color_mode = 'bg_color_orig'
    outDir = "../../dataset/EGFE_data_from_uldgnn_data"
    os.makedirs(outDir, exist_ok=True)
    dataloader = make_data_loader(cfg,is_train = False)
    vis = visualizer(cfg.visualizer)
    evaluator = Evaluator()
    print(len(dataloader))
    # trainer.train(0, dataloader, optim, recorder, evaluator )
    positives = 0
    negatives = 0
    num_of_fragments: List = []
    zeros = 0
    merge_acc = 0.0
    precision = 0.0
    recall = 0.0
    acc = 0.
    # result_transformer = json.load(open("bbox_result.json"))
    labels_pred = []
    labels_gt = []
    merge_recall = 0.0
    merge_precision = 0.0 
    new_bbox_results = process_bbox_result()
    merge_iou = 0.0
    print(len(dataloader))
    not_valid_samples = 0
    merge_eval_stats = {}
    type_id_to_class = {}
    
    for k, v in LAYER_CLASS_MAP.items():
        type_id_to_class[v] = deal_with_class_(k)
        
    for batch in tqdm(dataloader):
        #network(batch)
        nodes_, edges, types, img_tensor, labels, bboxes, nodes, node_indices, file_list  = batch
        layer_info_list = []
        merge_type = False
        label = 0
        file_path, artboard_name = os.path.split(file_list[0])
        artboard_name = artboard_name.split(".")[0]
        bboxes = bboxes + nodes
        for i in range(nodes.shape[0]):
            class_ = type_id_to_class[types[i].item()]
            label_ = 0
            if labels[i] == 1:
                if not merge_type:
                    merge_type = True
                    label_ = 2
                else:
                    if torch.sum(torch.abs(bboxes[i] - bboxes[i - 1])) < 1e-6:
                        label_ = 1
                    else:
                        label_ = 2
            else:
                merge_type = False
                label_ = 0
            color = [0, 0, 0, 0]
            
            tmp_json = json.load(open(f"/media/sda1/ljz-workspace/dataset/graph_dataset_rererefine_copy/{artboard_name}/{artboard_name}-0.json", "r"))
            
            # original_artboard_json = json.load(open(tmp_json['original_artboard'], "r"))
            
            artboard_height = int(tmp_json['artboard_height']) # artboard_json is a dictionary
            artboard_width = int(tmp_json['artboard_width'])
            rect = {"x": int(nodes[i][0] * artboard_width),
                    "y": int(nodes[i][1] * artboard_height),
                    "width": int(nodes[i][2] * artboard_width),
                    "height": int(nodes[i][3] * artboard_height)}
            layer_info_list.append({
                "id": "None",
                "name": "None",  
                "rect": rect, 
                "_class": class_,
                "label": label_,
                "color": color
            })
        
        json.dump({
                "layers": layer_info_list,
                "width": artboard_width, "height": artboard_height, 
                "layer_width": 64, "layer_height": 64
                }, 
                open(f"{outDir}/{artboard_name}.json","w"))
        os.system(f"cp {cfg.test_dataset.rootDir}/{artboard_name}/{artboard_name}-assets_rerefine.png {outDir}/{artboard_name}-assets.png")
        os.system(f"cp {cfg.test_dataset.rootDir}/{artboard_name}/{artboard_name}.png {outDir}/{artboard_name}.png")
           
        '''
        layer_info = {"id":id,
            "name":name,  
            "rect": rect, 
            "_class": class_,
            "label": label_,
            "color": color}
        '''

