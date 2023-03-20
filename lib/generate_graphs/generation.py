


'''
--dataDir
    |__0
       |————img
       |____json
    |__1
       |____img
       |____json
    |...
    |...
    
--outDir
    |__0
       |____0-assets.png
       |____0.png
       |____0-0.json
       |____0-1.json
       ....
    ...
'''

import cProfile
import logging
from abc import ABC, abstractmethod
from threading import Thread, Lock
from queue import Queue
from tqdm import tqdm
import asyncio
import json
from typing import List
import os
from bboxTree import bboxTree, bboxNode
import numpy as np
from PIL import Image
import sys
sys.setrecursionlimit(100000)
Artboard_index = 0
lock = Lock()

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
    def clip_val_(x):
        x = x if x>0 else 0
        if x>1:
            x=1
        return x
    root:bboxTree = bboxTree(0,0,2,2)

    labels = []
    layer_rect = []
    bbox = []
    types = []
  
    for idx, layer in enumerate(layer_list):
        x, y, w, h = layer['layer_rect']
        x, y, w, h = x / artboard_width, y / artboard_height, w / artboard_width, h / artboard_height
        x, y, w, h = clip_val_(x),clip_val_(y),clip_val_(w),clip_val_(h)
        assert(x>=0 and x<=1)
        assert(y>=0 and y<=1)
        assert(w>=0 and w<=1)
        assert(h>=0 and h<=1)
        #print(root.num)
        root.insert(bboxNode(root.num, x, y, x+w, x+h))
        types.append(LAYER_CLASS_MAP[layer['class']])
        if layer['label'] == 0:
            labels.append(0)
            layer_rect.append([x, y, w, h])
            bbox.append([0, 0, 0, 0])
        else:
            bbox_x, bbox_y, bbox_w, bbox_h = layer['bbox']
            bbox_x, bbox_y, bbox_w, bbox_h = (bbox_x-3) / artboard_width, (bbox_y-3) / artboard_height, (bbox_w+5) / artboard_width, (bbox_h+5) / artboard_height
            bbox_x, bbox_y, bbox_w, bbox_h = clip_val_(bbox_x), clip_val_(bbox_y), clip_val_(bbox_w), clip_val_(bbox_h)
            
            layer_rect.append([x, y, w, h])
        
            labels.append(1)
            bbox.append([bbox_x-x, bbox_y-y,bbox_w-w,bbox_h-h])
        
    assert(len(layer_rect)==len(layer_list))
    assert(len(bbox)==len(layer_list))
    assert(root.num == len(layer_list))
    tmp = []
  
    edges = root.gen_graph(tmp)
    
    assert(np.max(np.array(edges))==len(layer_list)-1)
    if(len(layer_list)<2):
        print("xxxxxxxx",output_path)
    json.dump({'layer_rect':layer_rect,
               'edges':edges,
                'bbox':bbox, 
                'types':types, 'labels':labels,'artboard_width':artboard_width, 
                'artboard_height':artboard_height}, open(output_path,"w"))

def clip_val(x, lower, upper):
    x = x if x >= lower else lower
    x = x if x <=upper else upper
    return x

async def generate_graph(artboard_json, artboard_img:Image, img_path:str, json_path:str, output_dir:str, folder_name):
    global Artboard_index
    
    artboard_height = float(artboard_json['artboard_height'])
    artboard_width = float(artboard_json['artboard_width'])

    output_dir = os.path.join(output_dir,folder_name)
    os.makedirs(output_dir,exist_ok=True)
    file_name = folder_name

    split_layers = []
    tmp_list = []
    split_num = 0
    merge_state = False
    
    assest_image = Image.new("RGBA", (64, 64 * len(artboard_json['layers'])),
                                 (255, 255, 255, 255))
    
    copy_artboard_path = os.path.join(output_dir, file_name+".png")
    os.system(f"cp {img_path} {copy_artboard_path}")
   
    for idx, layer in enumerate(artboard_json['layers']):
        x, y, w, h = layer['layer_rect']
        x1, y1, x2, y2 = clip_val(x, 0,  artboard_width), clip_val(y, 0, artboard_height), clip_val(x+w,0,artboard_width), clip_val(y+h,0,artboard_height)
        if layer['label'] == 1 :
            merge_state=True
            bbox_x, bbox_y, bbox_w, bbox_h = layer['bbox']
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = clip_val(bbox_x, 0,  artboard_width), clip_val(bbox_y, 0, artboard_height), clip_val(bbox_x+bbox_w,0,artboard_width), clip_val(bbox_y+bbox_h,0,artboard_height)
            x1, y1, x2, y2 = clip_val(x1, bbox_x1, bbox_x2),clip_val(y1, bbox_y1, bbox_y2),clip_val(x2, bbox_x1, bbox_x2),clip_val(y2, bbox_y1, bbox_y2)
            layer['layer_rect'] = [x1, y1, x2-x1, y2-y1]
            
        elif layer['label'] == 0 :
            merge_state = False
            
        layer_img = artboard_img.crop((x1,y1,x2,y2)).resize((64,64),resample=Image.BICUBIC)
        assest_image.paste(layer_img, (0, idx*64))
        
        tmp_list.append(layer)
        if len(artboard_json['layers']) - idx < 10:
            continue
        if merge_state:
            continue
        if len(tmp_list)>=20:
            split_layers.append(tmp_list)
            tmp_list = []
            
    assest_image.save(os.path.join(output_dir, file_name+"-assets.png"))
    
    if len(tmp_list)!=0:
        if len(tmp_list)<2:
            print("*****************",len(tmp_list), img_path)
        split_layers.append(tmp_list)
    
    sum = 0
    for idx, layer_list in enumerate(split_layers):
        sum += len(layer_list)
        assert(len(layer_list)!=0)
        await generate_single_graph(layer_list,
                        os.path.join(output_dir, file_name+f"-{idx}.json"),
                        artboard_height,artboard_width)
  
    assert(sum==len(artboard_json['layers']))
          
def generate_graph_sync(artboard_json, artboard_img, img_path, json_path, output_dir,folder_name):
    return asyncio.run( generate_graph(artboard_json, artboard_img, img_path, json_path, output_dir,folder_name) )

class GenerateGraphsThread(ProfileLoggingThread):
    def __init__(self, 
                artboard_queue: Queue,
                output_dir,
                thread_name: str,
                logfile_path: str,
                profile_path: str,
                pbar: tqdm):
        super().__init__(thread_name, logfile_path,profile_path)
        self.output_dir = output_dir
        self.artboard_queue = artboard_queue
        self.pbar = pbar

    def run_impl(self):
        global Artboard_index
        while True:
            if self.artboard_queue.empty():
                return  
            artboard_path = self.artboard_queue.get()
            self.logger.info(f"Generating graph for {artboard_path}")
            json_path = os.path.join(artboard_path,"artboard.json")
            
            img_path = os.path.join(artboard_path,"artboard.png")
            artboard_json = json.load(open(json_path,"r"))
            if len(artboard_json['layers']) < 5:
                pass
    
            else:    
                artboard_img = Image.open(img_path).convert("RGBA")
                if (np.array(artboard_img)==0).all():
                    pass
                else:
                    lock.acquire()
                    folder_name = str(Artboard_index)
                    Artboard_index+=1
                    lock.release()
                    generate_graph_sync(artboard_json, artboard_img, img_path, json_path, self.output_dir, folder_name)
            self.pbar.update()
            self.artboard_queue.task_done()

def generate_graphs(artboard_list: List[str],
                    logfile_folder: str,
                    profile_folder: str,
                    output_dir: str,
                    max_threads: int   
    ):
    pbar = tqdm(total = len(artboard_list))
    artboard_queue = Queue()
    os.makedirs(logfile_folder,exist_ok=True)
    os.makedirs(profile_folder, exist_ok=True)
    logfile_path = os.path.join(logfile_folder,"generate_graph.log")
    profile_path = os.path.join(profile_folder,"generate_graph.profile")

    for json_path in artboard_list:
        artboard_queue.put(json_path)

    for i in range(max_threads):
        thread_name = f"convert-thread-{i}"
        thread = GenerateGraphsThread(
            artboard_queue,
            output_dir,
            thread_name,
            logfile_path,
            profile_path,
            pbar
        )
        thread.daemon = True
        thread.start()
    
    artboard_queue.join()
    print("finished")
    
if __name__=='__main__':
    rootdir="/media/sda1/ljz-workspace/dataset/aliartboards/"
    outDir = "/media/sda1/ljz-workspace/dataset/fewshot_dataset/"
    logdir = 'out'
    os.makedirs(outDir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    
    indexes = 11
    artboard_list = [] 
    index_train = []
    for idx in range(indexes):
        artboard_list.append(f'{rootdir}{idx}')
        #index_train.append({"json": f"{idx}.json", "layerassets":f"{idx}-assets.png", "image":f"{idx}.png"})
        
    generate_graphs(artboard_list, f'{logdir}/log',
                    f'{logdir}/profile',
                    outDir,
                    max_threads=8)
    #json.dump(index_train,open(f"{outDir}/index_train.json","w"))
   
    
            
    
