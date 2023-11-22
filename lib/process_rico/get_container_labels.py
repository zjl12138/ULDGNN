import csv
import os
import numpy as np
import torch
import torchvision
from abc import ABC, abstractmethod
from threading import Thread, Lock
from queue import Queue
from typing import List
from tqdm import tqdm
import json
from PIL import Image

# read a rico data json file, return a dict, the key is the component id, the value is the bounds of the component
def read_csv(file_name):
    data = []
    with open(file_name) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    data_dict = {}
    for i in range(len(data)):
        if i != 0:
            data_dict.setdefault(data[i][0], {})
            data_dict[data[i][0]][data[i][1]] = data[i][2]
    return data_dict

root_dir = "/media/sda2/ljz_dataset"
out_dir = "/media/sda1/ljz-workspace/dataset/rico_graph"
label_dict = read_csv("/media/sda1/ljz-workspace/dataset/clay/clay_labels.csv") # map component id to label
filter_artboard_id = []
container_labels = set()
os.makedirs(out_dir, exist_ok = True)
class Processor(Thread):
    def __init__(self,  thread_id, queue, out_dir, dataset_name, pbar):
        super().__init__()
        self.thread_id = thread_id
        self.queue = queue
        self.out_dir = out_dir
        self.dataset_name = dataset_name
        self.pbar = pbar
    
    def parse_rico_json(self, dataset_name, id):
        file_name = os.path.join(root_dir, dataset_name, str(id) + ".json")
        try:
            with open(file_name, 'r') as f:
                data = json.load(f)
        except:
            print("wrong artboard json: ", id)
            filter_artboard_id.append(str(id))
            return None, None
        data_dict = {}
        def dfs_search(cur_node):
            data_dict[cur_node['pointer']] = cur_node['bounds']
            if 'children' not in cur_node.keys() or cur_node['children'] == []:
                return
            for child in cur_node['children']:
                if child is None:
                    continue
                dfs_search(child)
        dfs_search(data["activity"]['root'])
        return data_dict, data["activity"]['root']['bounds']

    def is_child(self, parent, child):
        if parent['bounds'][0] <= child['bounds'][0] and parent['bounds'][1] <= child['bounds'][1] \
            and parent['bounds'][2] >= child['bounds'][2] and parent['bounds'][3] >= child['bounds'][3]:
            return True
        else:
            return False

    def build_tree(self, component_list: List):
        component_list.sort(key=lambda x: (x['bounds'][2] - x['bounds'][0]) * (x['bounds'][3] - x['bounds'][1]), reverse = True)
        root = component_list[0]
        for i, node in enumerate(component_list):
            if i == 0:
                continue
            for j in range(i - 1, -1, -1):
                if self.is_child(component_list[j], node):
                    component_list[j]['children'].append(node)
                    break
        return root

    def generate_assets(self, id, root, node_num):
        # print("nodenum: ", node_num)
        
        def dfs_search(cur_node, parent_bbox = [1., 1., 1., 1.]):
            cur_node_box = [1., 1., 1., 1.]
            cur_index = cur_node['index']
            if cur_node['label'] != -1:
                if 'children' in cur_node.keys() and len(cur_node['children']) != 0:
                    container_labels.add(cur_node['label'])
        
            for child_node in cur_node['children']:
                dfs_search(child_node, cur_node_box)

        dfs_search(root)

    def convert_rico(self, dataset_name, id):
        json_dict, root_bounds = self.parse_rico_json(dataset_name, id)
        if json_dict is None:
            return
        component_list = []
        idx = 0
        for k, v in json_dict.items():
            if k in label_dict[str(id)].keys():
                label_int = int(label_dict[str(id)][k])
                if label_int != -1: # we abandon root node
                    if (v[0] >= root_bounds[0] and v[1] >= root_bounds[1] and v[2] <= root_bounds[2] and v[3] <= root_bounds[3]):
                        if v[2] >= v[0] and v[3] >= v[1]:
                            component_list.append({"id": k, "label": label_int, "bounds": v, 'children': [], "index": idx})
                            idx += 1
                else:
                    component_list.append({"id": k, "label": label_int, "bounds": v, 'children': [], "index": -1})
        if idx < 5:
            filter_artboard_id.append(str(id))
            return
        root = self.build_tree(component_list)
        self.generate_assets(id, root, idx)

    def run(self):
        while True:
            if self.queue.empty():
                return
            id = self.queue.get()
            # print(self.thread_id, "-" * 10, id)
            os.makedirs(os.path.join(self.out_dir, str(id)), exist_ok = True)
            self.convert_rico(self.dataset_name, id)
            self.pbar.update()
            self.queue.task_done()

def generate_graph_dataset(dataset_name, max_threads = 4):
    artboard_list = list(label_dict.keys())
    # artboard_list = [2972]
    pbar = tqdm(total = len(artboard_list))
    artboard_queue = Queue()
    for i, id in enumerate(artboard_list):
        # if i > 12000:
        artboard_queue.put(id)

    for i in range(max_threads):
        thread = Processor(i, artboard_queue, out_dir, dataset_name, pbar)
        thread.daemon = True
        thread.start()
    artboard_queue.join()
    print("finished")

generate_graph_dataset("combined", 32)
print(container_labels)