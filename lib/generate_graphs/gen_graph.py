import cProfile
import logging
from abc import ABC, abstractmethod
from threading import Thread
from queue import Queue
from tqdm import tqdm
import asyncio
import json
from typing import List
import os
from bboxTree import bboxTree, bboxNode
import numpy as np

LAYER_CLASS_MAP = {
    'symbolMaster': 0,
    'group': 1,
    'oval': 2,
    'polygon': 3,
    'rectangle': 4,
    'shapePath': 5,
    'star': 6,
    'triangle': 7,
    'shapeGroup': 8,
    'text': 9,
    'symbolInstance': 10,
    'slice': 11,
    'MSImmutableHotspotLayer': 12,
    'bitmap': 13,
}

def make_logger(logger_name:str,
                logfile_path:str,
                stream_level: int = logging.ERROR,
                file_level: int=logging.DEBUG):
    fomatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(stream_level)
    sh.setFormatter(stream_level)
    logger.addHandler(sh)
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(file_level)
    fh.setFormatter(fomatter)
    logger.addHandler(fh)
    return logger

class LoggingThread(Thread):
    def __init__(self, thread_name:str, logfile_path):
        super().__init__()
        self.thread_name = thread_name
        self.logfile_path = logfile_path
        self.logger = make_logger(thread_name,logfile_path)

class ProfileLoggingThread(LoggingThread,ABC):
    def __init__(self, thread_name:str, logfile_path:str, profile_path:str):
        super().__init__(thread_name,logfile_path)
        self.profile_path = profile_path
    
    @abstractmethod
    def run_impl(self):
        pass 

    def run(self):
        profile = cProfile.Profile()
        try:
            profile.runcall(self.run_impl)
        except Exception as e:
            self.logger.error(f'{str(e)}')
        finally:
            profile.dump_stats(self.profile_path)

async def generate_single_graph(layer_list, output_path, artboard_height, artboard_width):
    def clip_val(x):
        x = abs(x)
        if x>1:
            x=1
        return x
    root = bboxTree(0,0,2,2)

    labels = []
    layer_rect = []
    bbox = []
    types = []
    merge_groups = []
    tmp_merge_group = []
    for idx, layer in enumerate(layer_list):
        x, y, w, h = abs(layer['rect']['x'])/artboard_width, layer['rect']['y']/artboard_height, layer['rect']['width']/artboard_width, layer['rect']['height']/artboard_height
        x, y, w, h = clip_val(x),clip_val(y),clip_val(w),clip_val(h)
        assert(x>=0 and x<=1)
        assert(y>=0 and y<=1)
        assert(w>=0 and w<=1)
        assert(h>=0 and h<=1)
        layer_rect.append([x, y, w, h])
        #print(root.num)
        root.insert(bboxNode(root.num, x, y, x+w, x+h))
        types.append(LAYER_CLASS_MAP[layer['_class']])
        if layer['label']==0 or layer['label']==1:
            labels.append(0)
            bbox.append([0, 0, 0, 0])
        else:
            labels.append(1)
            bbox.append([])
        if layer['label']==2:
            if len(tmp_merge_group)!=0:
                merge_groups.append(tmp_merge_group)
                tmp_merge_group=[]
            tmp_merge_group.append(idx)
        if layer['label']==3:
            tmp_merge_group.append(idx)
        if layer['label']==0 or layer['label']==1:
            merge_groups.append(tmp_merge_group)
            tmp_merge_group=[]
    if len(tmp_merge_group)!=0:
        merge_groups.append(tmp_merge_group)

    for merge_group in merge_groups:
        tmp_bbox = [2,2,0,0]    
        for idx in merge_group:
            x, y, w, h = layer_rect[idx]
            tmp_bbox[0], tmp_bbox[1] = min(tmp_bbox[0],x),min(tmp_bbox[1],y)
            tmp_bbox[2], tmp_bbox[3] = max(tmp_bbox[2],x+w), max(tmp_bbox[3],y+h)
        for idx in merge_group:
            x, y, w, h = layer_rect[idx]
            bbox[idx].extend([tmp_bbox[0]-x,tmp_bbox[1]-y,tmp_bbox[2]-tmp_bbox[0]-w, tmp_bbox[3]-tmp_bbox[1]-h])
    
    assert(len(layer_rect)==len(layer_list))
    assert(len(bbox)==len(layer_list))
    assert(root.num == len(layer_list))
    tmp = []
    edges = root.gen_graph(tmp)
    #print(np.max(np.array(edges)),len(layer_list))
    assert(np.max(np.array(edges))==len(layer_list)-1)
    json.dump({'layer_rect':layer_rect,'edges':edges,
                'bbox':bbox, 
                'types':types, 'labels':labels,'artboard_width':artboard_width, 
                'artboard_height':artboard_height}, open(output_path,"w"))

async def generate_graph(json_path:str, output_dir:str):
    artboard_json = json.load(open(json_path,"r"))
    artboard_height = float(artboard_json['height'])
    artboard_width = float(artboard_json['width'])
    file_name = os.path.split(json_path)[1].split(".")[0]
   
    output_dir = os.path.join(output_dir,file_name)
    os.makedirs(output_dir,exist_ok=True)

    split_layers = []
    tmp_list = []
    split_num = 0
    merge_state = False
    for idx, layer in enumerate(artboard_json['layers']):
        if layer['label'] ==2:
            merge_state=True
        if layer['label']==0 or layer['label']==1:
            merge_state=False
        tmp_list.append(layer)
        split_num += 1
        if split_num >= 20:
            if merge_state or len(artboard_json['layers'])-idx<20:
                continue
            split_layers.append(tmp_list)
            tmp_list=[]
            split_num = 0
        
    if len(tmp_list)!=0:
        split_layers.append(tmp_list)
    sum = 0
    for idx, layer_list in enumerate(split_layers):
        sum += len(layer_list)
        await generate_single_graph(layer_list,
                        os.path.join(output_dir, file_name+f"-{idx}.json"),
                        artboard_height,artboard_width)
  
    assert(sum==len(artboard_json['layers']))
           
def generate_graph_sync(json_path, output_dir):
    return asyncio.run( generate_graph(json_path, output_dir) )

class GenerateGraphsThread(ProfileLoggingThread):
    def __init__(self, 
                json_queue: Queue,
                output_dir,
                thread_name: str,
                logfile_path: str,
                profile_path: str,
                pbar: tqdm):
        super().__init__(thread_name, logfile_path,profile_path)
        self.output_dir = output_dir
        self.json_queue = json_queue
        self.pbar = pbar

    def run_impl(self):
        while True:
            if self.json_queue.empty():
                return  
            json_path = self.json_queue.get()
            self.logger.info(f"Generating graph for {json_path}")
            generate_graph_sync(json_path,self.output_dir)
            self.pbar.update()
            self.json_queue.task_done()

def generate_graphs(json_list: List[str],
                    logfile_folder: str,
                    profile_folder: str,
                    output_dir: str,
                    max_threads: int   
    ):
    pbar = tqdm(total = len(json_list))
    json_queue = Queue()
    os.makedirs(logfile_folder,exist_ok=True)
    os.makedirs(profile_folder, exist_ok=True)
    logfile_path = os.path.join(logfile_folder,"generate_graph.log")
    profile_path = os.path.join(profile_folder,"generate_graph.profile")

    for json_path in json_list:
        json_queue.put(json_path)

    for i in range(max_threads):
        thread_name = f"convert-thread-{i}"
        thread = GenerateGraphsThread(
            json_queue,
            output_dir,
            thread_name,
            logfile_path,
            profile_path,
            pbar
        )
        thread.daemon = True
        thread.start()
    print("finished")
    json_queue.join()

if __name__=='__main__':
    rootdir="/media/sda1/ljz-workspace/dataset/ui_dataset/"
    outDir = "/media/sda1/ljz-workspace/dataset/fewshot_dataset/"
    os.makedirs(outDir, exist_ok=True)
    indexes = 20
    json_list = [] 
    index_train = []
    for idx in range(indexes):
        json_list.append(f'/media/sda1/ljz-workspace/dataset/ui_dataset/{idx}.json')
        index_train.append({"json": f"{idx}.json", "layerassets":f"{idx}-assets.png", "image":f"{idx}.png"})
        
    generate_graphs(json_list, '/media/sda1/ljz-workspace/code/ULGnn/output/log',
                    '/media/sda1/ljz-workspace/code/ULGnn/output/profile',
                    outDir,
                    max_threads=8)
    #json.dump(index_train,open(f"{outDir}/index_train.json","w"))
   
    for i in range(indexes):
        os.system(f"cp {rootdir}/{i}.png {outDir}/{i}/{i}.png")
        os.system(f"cp {rootdir}/{i}-assets.png {outDir}/{i}/{i}-assets.png")
    os.system(f"cp {rootdir}/index_train.json {outDir}/index_train.json")
    os.system(f"cp {rootdir}/index_test.json {outDir}/index_test.json")    
    

