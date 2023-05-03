


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
import math
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
            self.logger.error(f'{e}')
        finally:
            profile.dump_stats(self.profile_path)

async def generate_single_graph(layer_list : List, output_path, artboard_height, artboard_width, json_path):
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
    
    #areas = np.array([layer['layer_rect'][2] * layer['layer_rect'][3] for layer in layer_list])
    #ind = np.argsort(areas)

    for idx, layer in enumerate(layer_list):
        x, y, w, h = layer['layer_rect']
        x, y, w, h = x / artboard_width, y / artboard_height, w / artboard_width, h / artboard_height
        x, y, w, h = clip_val_(x), clip_val_(y), clip_val_(w), clip_val_(h)
        assert(x >= 0 and x <= 1)
        assert(y >= 0 and y <= 1)
        assert(w >= 0 and w <= 1)
        assert(h >= 0 and h <= 1)
        #print(root.num)
        root.insert(bboxNode(root.num, x, y, x + w, x + h))
        types.append(LAYER_CLASS_MAP[layer['class']])
        if layer['label'] == 0:
            labels.append(0)
            layer_rect.append([x, y, w, h])
            bbox.append([0, 0, 0, 0])
        else:
            bbox_x, bbox_y, bbox_w, bbox_h = layer['bbox']
            bbox_x, bbox_y, bbox_w, bbox_h = (bbox_x - 3) / artboard_width, (bbox_y - 3) / artboard_height, (bbox_w + 5) / artboard_width, (bbox_h + 5) / artboard_height
            bbox_x, bbox_y, bbox_w, bbox_h = clip_val_(bbox_x), clip_val_(bbox_y), clip_val_(bbox_w), clip_val_(bbox_h)
            
            layer_rect.append([x, y, w, h])
        
            labels.append(1)
            bbox.append([bbox_x - x, bbox_y - y,bbox_w - w,bbox_h - h])
        
    assert(len(layer_rect)==len(layer_list))
    assert(len(bbox)==len(layer_list))
    assert(root.num == len(layer_list))
    tmp = []
  
    edges = root.gen_graph(tmp)
    
    assert(np.max(np.array(edges)) == len(layer_list) - 1)
    if(len(layer_list) < 2):
        print("xxxxxxxx", output_path)
    json.dump({'layer_rect': layer_rect,
               'edges': edges,
                'bbox': bbox, 
                'types': types, 'labels': labels, 'artboard_width': artboard_width, 
                'artboard_height': artboard_height,
                'original_artboard': json_path}, open(output_path, "w"))

def clip_val(x, lower, upper):
    x = x if x >= lower else lower
    x = x if x <= upper else upper
    return x

async def generate_graph(artboard_json, artboard_img:Image,  img_path:str, json_path:str, output_dir:str, folder_name):
    global Artboard_index
    assert(len(artboard_json['layers']) >= 10)
    artboard_height = float(artboard_json['artboard_height']) # artboard_json is a dictionary
    artboard_width = float(artboard_json['artboard_width'])

    output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    file_name = folder_name

    split_layers = []
    tmp_list = []
    split_num = 0
    merge_state = False
    
    copy_artboard_path = os.path.join(output_dir, file_name + ".png")
    os.system(f"cp {img_path} {copy_artboard_path}")
   
    remove_non_valid_layers = 0    
    layer_img_list_id = []
    for idx, layer in enumerate(artboard_json['layers']):
        x, y, w, h = layer['layer_rect']
        x1, y1, x2, y2 = clip_val(x, 0,  artboard_width), clip_val(y, 0, artboard_height), clip_val(x + w, 0, artboard_width), clip_val(y + h, 0, artboard_height)
        
        if abs(x2 - x1) < 1 or abs(y2 - y1)<1:
            remove_non_valid_layers += 1
            continue
        
        if layer['label'] == 1 :
            merge_state=True
            bbox_x, bbox_y, bbox_w, bbox_h = layer['bbox']
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = clip_val(bbox_x, 0, artboard_width), clip_val(bbox_y, 0, artboard_height), clip_val(bbox_x+bbox_w,0,artboard_width), clip_val(bbox_y+bbox_h,0,artboard_height)
            x1, y1, x2, y2 = clip_val(x1, bbox_x1, bbox_x2), clip_val(y1, bbox_y1, bbox_y2), clip_val(x2, bbox_x1, bbox_x2), clip_val(y2, bbox_y1, bbox_y2)
            layer['layer_rect'] = [x1, y1, x2 - x1, y2 - y1]
            
        elif layer['label'] == 0 :
            merge_state = False
            
        #layer_img = artboard_img.crop((x1,y1,x2,y2)).resize((64,64),resample=Image.BICUBIC)
        layer_img_list_id.append([layer['id'], (x1, y1, x2, y2)])

        tmp_list.append(layer)
        if len(artboard_json['layers']) - idx < 10:
            continue
        if merge_state:
            continue
        if len(tmp_list) >= 40:
            split_layers.append(tmp_list)
            tmp_list = []
    if 64 * 64 * len(layer_img_list_id)>=89478485:
        print(img_path)
    assest_image = Image.new("RGBA", (64, 64 * len(layer_img_list_id)),
                                 (255, 255, 255, 255))
    
    artboard_folder, _ = os.path.split(img_path)
    for idx, layer_img_id_ in enumerate(layer_img_list_id):
        layer_img_id, bbox = layer_img_id_
        x1, y1, x2, y2 = bbox
        layer_img_path = os.path.join(artboard_folder, 'layer_imgs', layer_img_id + ".png")
        if not os.path.exists(layer_img_path):  #if the sketchtool cannot export layer named layer_img_id, we just use the patch img from original artboard img according to bbox
            layer_img = artboard_img.crop((x1, y1, x2, y2)).resize((64, 64), resample = Image.BICUBIC)
        else:
            layer_img = Image.open(layer_img_path).convert("RGBA").resize((64, 64), resample = Image.BICUBIC)
            '''if (np.array(layer_img) == 0).all():
                #print(layer_img_path, idx, (x1, y1, x2, y2))
                layer_img = artboard_img.crop((x1, y1, x2, y2)).resize((64,64),resample = Image.BICUBIC)'''
        assest_image.paste(layer_img, (0, idx * 64))
          
    assest_image.save(os.path.join(output_dir, file_name+"-assets.png"))
    
    if len(tmp_list) != 0:
        if len(tmp_list) < 10:    # if the number of layers in this list is less than 10 we don't need generate graph for it
            #print("**********", len(tmp_list), len(artboard_json['layers']), json_path)
            if len(split_layers) == 0:
                print(img_path, len(artboard_json['layers']), remove_non_valid_layers, folder_name)
            #assert(len(split_layers) != 0)
            for lay in tmp_list:
                split_layers[-1].append(lay) #    
        else:
            split_layers.append(tmp_list)
        
    sum = 0
    for idx, layer_list in enumerate(split_layers):
        sum += len(layer_list)
        assert(len(layer_list) > 1)
        await generate_single_graph(layer_list,
                        os.path.join(output_dir, file_name+f"-{idx}.json"),
                        artboard_height,artboard_width, json_path)
    #print(remove_non_valid_layers, sum ,len(artboard_json['layers']))
    assert(sum == len(layer_img_list_id))
    assert(sum == len(artboard_json['layers']) - remove_non_valid_layers)
          
def generate_graph_sync(artboard_json, artboard_img, img_path, json_path, output_dir,folder_name):
    return asyncio.run( generate_graph(artboard_json, artboard_img, img_path, json_path, output_dir,folder_name) )

class GenerateGraphsThread(ProfileLoggingThread):
    def __init__(self, 
                idx_mapping,
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
        self.idx_mapping = idx_mapping
        self.filter_artboard = [447, 529, 612, 1422, 2496, 3267, 4201, 286, 1669, 2001, 2441, 2431, 4322, 5462, 5466]

    def run_impl(self):
        global Artboard_index
        while True:
            if self.artboard_queue.empty():
                return  
            idx = self.artboard_queue.get()
            artboard_path = self.idx_mapping[idx]
            artboard_idx = os.path.split(artboard_path)[1]
            if int(artboard_idx) in self.filter_artboard:
                print("filter folder", idx)
                self.pbar.update()
                self.artboard_queue.task_done() 
                continue
            self.logger.info(f"Generating graph for {artboard_path}")
            json_path = os.path.join(artboard_path, "artboard.json")
            
            img_path = os.path.join(artboard_path, "artboard.png")
            artboard_json = json.load(open(json_path, "r"))
            w, h = artboard_json['artboard_width'], artboard_json['artboard_height']
            
            if w * h >= 89478485:
                print(w," ", h, img_path)
            #if w >= 1500 or len(artboard_json['layers']) < 10 or len(artboard_json['layers'])>=1000:
            if len(artboard_json['layers']) < 10 or len(artboard_json['layers']) >= 1000:
                pass
            else:    
            
                artboard_img: Image = Image.open(img_path).convert("RGBA")
                
                if (np.array(artboard_img) == 0).all():
                    pass
                else:
                   
                    folder_name = idx # the folder to store graphs for this artboard
                    generate_graph_sync(artboard_json, artboard_img, img_path, json_path, self.output_dir, folder_name)
               
            self.pbar.update()
            self.artboard_queue.task_done()

def generate_graphs(idx_mapping, artboard_list: List[str],
                    logfile_folder: str,
                    profile_folder: str,
                    output_dir: str,
                    max_threads: int   
    ):
    pbar = tqdm(total = len(artboard_list))
    artboard_queue = Queue()
    os.makedirs(logfile_folder, exist_ok = True)
    os.makedirs(profile_folder, exist_ok = True)
    logfile_path = os.path.join(logfile_folder, "generate_graph.log")
    profile_path = os.path.join(profile_folder, "generate_graph.profile")

    for json_path in artboard_list:
        artboard_queue.put(json_path)

    for i in range(max_threads):
        thread_name = f"convert-thread-{i}"
        thread = GenerateGraphsThread(idx_mapping,
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
    rootdir="/media/sda1/ljz-workspace/dataset/aliartboards_refine/"
    outDir = "/media/sda1/ljz-workspace/dataset/graph_dataset_rererefine"
    logdir = 'out'
    os.makedirs(outDir, exist_ok = True)
    os.makedirs(logdir, exist_ok = True)
    
    indexes = 5503
    artboard_list = [] 
    index_train = []
    filter_idx = [447, 529, 612, 1422, 2496, 3267, 4201, 286, 1669, 2001, 2441, 2431, 4322, 5462, 5466]
    idx_mapping = json.load(open("idx_mapping.json"))
    artboard_list = idx_mapping.keys()
        #index_train.append({"json": f"{idx}.json", "layerassets":f"{idx}-assets.png", "image":f"{idx}.png"})
        
    generate_graphs(idx_mapping, artboard_list, f'{logdir}/log',
                    f'{logdir}/profile',
                    outDir,
                    max_threads = 8)
    #json.dump(index_train,open(f"{outDir}/index_train.json","w"))
   
    
            
    

