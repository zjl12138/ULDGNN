# -*- coding: utf-8 -*-

from ast import AnnAssign
from builtins import isinstance
from cgitb import handler
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from re import L
import threading
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from shapely import affinity
from shapely.geometry import Point, point
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from layer_transform import BoundArea, LayerTransformed, Rotation, Scale, Transform, recursive_remove_layer, \
    transform_layer, get_polygon_exterior
from sketch_utils import from_file
from sketch_utils import AnyLayer, Artboard, Color, Group, Page, SymbolInstance, SymbolMaster, SketchFile
from sketch_utils import SymbolMaster, Group, Oval, Polygon, Rectangle, ShapePath, Star, Triangle, ShapeGroup, Text, SymbolInstance, Slice, Hotspot, Bitmap, Page, Artboard
import json
import traceback

import copy


import cProfile
import logging
from abc import ABC, abstractmethod
from threading import Thread
from queue import Queue
from tqdm import tqdm
import asyncio

from sketchtool.sketchtool_wrapper import SketchToolWrapper

Artboard_index = 0
artboard_lock = threading.Lock()

@dataclass
class ClippingMask:
    """
    A class that can be used to check whether a layer has a clipping mask.
    """
    hasClippingMask: bool
    mask: Optional[BoundArea] = None


@dataclass
class ImageObject:
    """
    A class that store image.
    """
    image: Image.Image
    path: str

    def save(self):
        self.image.save(self.path)


class BaseTraverseHandler(ABC):
    """
    A base class for traverse handler.
    """

    def init(self) -> None:
        """
        initialization.
        do anything like create image or generate list .etc as you want
        """
        pass

    @abstractmethod
    def handle_layer(self, layer: LayerTransformed, layer_stack: Tuple[AnyLayer]) -> None:
        """
        handle a visible layer.
        @param layer: the transformed layer with polygon
        @param layer_stack: the layer stack to reach this layer
        """
        pass

    @abstractmethod
    def if_group_need_handle(self, group: Group) -> bool:
        """
        if a group need extra handle.
        @param group: the group to be checked
        """
        pass

    @abstractmethod
    def handle_group(self, layer: LayerTransformed, layer_stack: Tuple[AnyLayer]) -> None:
        """
        handle a group match :py:func:`BaseTraverseHandler.if_group_need_handle` result.
        @param layer: the transformed group layer with polygon
        @param layer_stack: the layer stack to reach this layer
        """
        pass

    def finish(self) -> None:
        """
        finish traverse
        do anything like save image or save list file .etc as you want
        """
        pass

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

class process_thread(ProfileLoggingThread):
    def __init__(self, sketch_queue:Queue, outDir, thread_name, logfile_path, profile_path,pbar:tqdm):
        super().__init__(thread_name, logfile_path,profile_path)
        self.sketch_queue = sketch_queue
        self.outDir = outDir
        self.pbar = pbar
    def run_impl(self):
        while True:
            if self.sketch_queue.empty():
                break
            sketch_path = self.sketch_queue.get()
            sketch_name = os.path.split(sketch_path)[1].split(".")[0]
            self.logger.info(f"Generating graph for {sketch_path}")
            PreProcessEachSketch(self.outDir,
                        sketch_path, sketch_name).process()
            self.pbar.update()
            self.sketch_queue.task_done()


class PreProcess:
    def __init__(self, outDir:str, sketch_list: List[str], image_list: List[str] = (), scale: int = 2):
        """

        @param sketch_list: sketch 2file path list
        @param image_list: result path for each sketch file
        @param scale: result image scale
        """
        # load sketches from files
        self.outDir = outDir
        self.sketch_queue= Queue()
        for sketch_file_path in sketch_list:
            self.sketch_queue.put(sketch_file_path)
            
        self.image_list = list(image_list)  # result image filename list
        self.count = 0
        self.pbar = tqdm( total= len(sketch_list) )
    def process(self, max_threads=8) -> None:
        """
        traverse sketch list and generate image for each sketch artboard
        """
        # for each sketch
        logfile_folder = os.path.join(self.outDir, "log")
        profile_folder = os.path.join(self.outDir, "profile")
        
        os.makedirs(logfile_folder,exist_ok=True)
        os.makedirs(profile_folder, exist_ok=True)
        logfile_path = os.path.join(logfile_folder,"generate_graph")
        profile_path = os.path.join(profile_folder,"generate_graph")
        
        for i in range(max_threads):
            thread_name = f"thread_{i}"
            thread = process_thread(self.sketch_queue, self.outDir, thread_name,logfile_path+f"_{i}.log", 
                                    profile_path+f"_{i}.profile", self.pbar)
            thread.daemon = True
            thread.start()
        self.sketch_queue.join()
        print("finished")
        
def intersect(pointone, pointtwo):
    b = min(pointone[1]+pointone[3], pointtwo[1]+pointtwo[3])
    t = max(pointone[1], pointtwo[1])
    r = min(pointone[0]+pointone[2], pointtwo[0]+pointtwo[2])
    l = max(pointone[0], pointtwo[0])
    if (b >= t) and (r >= l):
        return b, t, l, r
    return -1, -1, -1, -1

'''
rootDir
    |__sketchname-artboardid
        |__artboard_img
        |__artboard_json

'''

class ReadHandler(BaseTraverseHandler):
    
    def __init__(self, outPath:str, layer:AnyLayer, sketchname:str, artboard_id:str) -> None:
        """
        initialization.
        do anything like create image or generate list .etc as you want
        """
        global Artboard_index
        self.info_list = [] #
        self.layer_belongsTo_group = {}
        #self.dump_folder = os.path.join(outPath, sketchname+"_"+artboard_id)
        
        artboard_lock.acquire()
        self.dump_folder = os.path.join(outPath, str(Artboard_index))
        Artboard_index = Artboard_index + 1
        artboard_lock.release()
        
        os.makedirs(self.dump_folder,exist_ok=True)
        self.outFile = os.path.join(self.dump_folder,"artboard.json")
        self.artboard_width  =layer.frame.width
        self.artboard_height = layer.frame.height
        self.artboard_id = artboard_id
        self.sketchname = sketchname
        self.sketchtool = SketchToolWrapper("/Applications/Sketch.app/")


    def preprocess_layer(self, layerTransformed: LayerTransformed):
        exterior = get_polygon_exterior(layerTransformed.bound_area)
        layer = layerTransformed.origin
        if len(exterior) >= 2:
            min_x = min([point[0] for point in exterior])
            max_x = max([point[0] for point in exterior])
            min_y = min([point[1] for point in exterior])
            max_y = max([point[1] for point in exterior])

            layer.frame.x = min_x
            layer.frame.y = min_y
            layer.frame.width = max_x-min_x
            layer.frame.height = max_y-min_y
        return layer

    def handle_group(self, layerTransformed: LayerTransformed, symbol_master_dict):

        def get_mergeleafs(layer: AnyLayer, leafs: List[AnyLayer], symbol_master_dict):
            if isinstance(layer, Page) or isinstance(layer, Artboard) or isinstance(layer, SymbolMaster):
                for i in layer.layers:
                    get_mergeleafs(i, leafs, symbol_master_dict)
            else:
                if isinstance(layer, Group) or isinstance(layer, ShapeGroup):
                    for i in layer.layers:
                        get_mergeleafs(i, leafs, symbol_master_dict)
                elif isinstance(layer, SymbolInstance) and layer.symbolID in symbol_master_dict.keys():
                    get_mergeleafs(
                        symbol_master_dict[layer.symbolID], leafs, symbol_master_dict)
                else:
                    leafs.append(layer.do_objectID)
                    self.layer_belongsTo_group[layer.do_objectID] = grouplayer

        merge_leafs = []
        grouplayer = self.preprocess_layer(layerTransformed)
        get_mergeleafs(grouplayer, merge_leafs, symbol_master_dict)
    
    def get_type(self, layer: AnyLayer):
        if isinstance(layer, Oval):
            return 'Oval'
        if isinstance(layer, Polygon):
            return 'Polygon'
        if isinstance(layer, Rectangle):
            return 'Rectangle'
        if isinstance(layer, Hotspot):
            return 'Hotspot'
        if isinstance(layer, Bitmap):
            return 'Bitmap'
        if isinstance(layer, ShapePath):
            return 'ShapePath'
        if isinstance(layer, Slice):
            return 'Slice'
        if isinstance(layer, Star):
            return 'Star'
        if isinstance(layer, Text):
            return 'Text'
        if isinstance(layer, Triangle):
            return 'Triangle'
        if isinstance(layer, SymbolInstance):
            return 'SymbolInstance'
    
    def handle_layer(self, layer: LayerTransformed, ifmerge:bool, layer_name:str) -> None:
        
        layer = self.preprocess_layer(layer)
        #if isinstance(layer, Text):
        #    return 
        bbox_layer = self.layer_belongsTo_group[layer.do_objectID] if ifmerge else None
        if ifmerge:
            x, y, w, h = bbox_layer.frame.x, bbox_layer.frame.y, bbox_layer.frame.width, bbox_layer.frame.height
        
        self.info_list.append({
            'layer_rect':[layer.frame.x,layer.frame.y,layer.frame.width,layer.frame.height],
            'id':layer.do_objectID,
            'label': 1 if ifmerge else 0,
            'bbox': [x-layer.frame.x, y-layer.frame.y, w-layer.frame.width, h-layer.frame.height] if ifmerge else [0,0,0,0],
            'name': layer_name,
            'class': self.get_type(layer)
        })

    def if_group_need_handle(self, group: Group or ShapeGroup) -> bool:
        # only process group contains '#merge#' keyword
        return "#merge#" in group.name or isinstance(group,ShapeGroup)

    def finish(self, sketch_path:str, artboard_name:str) -> None:
        """
        finish traverse
        do anything like save image or save list file .etc as you want
        """
        #os.system(f"/Applications/Sketch.app/Contents/MacOS/sketchtool export artboards {sketch_path} --output={self.dump_folder} --item={self.artboard_id} --use-id-for-name=YES")
        asyncio.run( self.sketchtool.export.artboards(sketch_path,items=[self.artboard_id],output=self.dump_folder) )    
        json.dump({'layers':self.info_list,
                   "artboard_width":self.artboard_width,
                   "artboard_height":self.artboard_height,
                   "sketch_name":self.sketchname,
                   "artboard": artboard_name}, open(self.outFile,"w"))

class PreProcessEachSketch:
    def __init__(self, outDir:str, sketch_path: str, sketch_name: str):
        self.outDir=outDir
        self.sketch_path = sketch_path
        self.sketch_file = from_file(sketch_path)
        self.sketch_name = sketch_name
        #os.makedirs(self.sketch_name, exist_ok=True)
        self.symbol_master_dict = {}
        self.layer_stack: List[AnyLayer] = []

    def analyze_symbol_master(self) -> None:
        # load foreign symbol master
        for foreign_symbol in self.sketch_file.contents.document.foreignSymbols:
            self.symbol_master_dict[foreign_symbol.symbolMaster.symbolID] = foreign_symbol.symbolMaster
        # load local symbol master
        for page in self.sketch_file.contents.document.pages:
            #print(f"finding symbol master in page {page.name}")
            # for each sketch page layer (artboard/symbol master)
            for layer in page.layers:
                if isinstance(layer, SymbolMaster):
                    self.symbol_master_dict[layer.symbolID] = layer

    def process(self) -> None:
        """
        traverse sketch and generate image for each sketch artboard
        """
        self.analyze_symbol_master()
        # for each sketch page
        for page in self.sketch_file.contents.document.pages:
            #print(f"processing page {page.name}")
            # remove all invisible layers
            reduced_page = recursive_remove_layer(page)
            # for each sketch page layer (artboard/symbol master)

            for layer in reduced_page.layers:
                # if the layer is an artboard
                if isinstance(layer, Artboard):
                    if layer.frame.width > 1000:
                        continue  # abandon sketches for computer app
                    #print(f"processing artboard {layer.name}")
                    # create a new image
                    artboard_name = layer.name
                    self.layer_stack = []
                    leafs = []
                    handler = ReadHandler(self.outDir, layer, self.sketch_name, layer.do_objectID)
                    handler.init()
                    # traverse the artboard and draw
                    asyncio.run( self.traverse(layer, Point(0, 0), handler, leafs) )
                    #print("*"*80, len(leafs))
                    handler.finish(self.sketch_path, artboard_name) 
    
    async def traverse(self, layer: AnyLayer, parent_xy: Point, traverse_handler: BaseTraverseHandler, leafs=[],
                 transforms: Tuple[Transform] = (), bound_areas: Tuple[BoundArea] = (),
                 opacity: float = 1, tint: Color = None, ifmerge=False) -> ClippingMask:
        """
        traverse sketch list and generate image for each sketch artboard
        @param layer: the layer need to be traversed
        @param traverse_handler:
        @param parent_xy: the parent x and y since all layer"s xy are relative
        @param transforms: the Flip/Rotate inherit from parent
        @param bound_areas: the mask inherit from parent
        @param opacity:
        @param tint:
        @return: if layer has ClippingMask, return the calculated mask area
        """
        if not layer.isVisible:
            return ClippingMask(False)
        # if the layer is a page or artboard or symbol master
        if isinstance(layer, Page) or isinstance(layer, Artboard) or isinstance(layer, SymbolMaster):
            # their x axis and y axis are relative to whole page, ignore them
            # traverse the group and draw
            self.layer_stack.append(layer)
            for i in layer.layers:
                await self.traverse(i, parent_xy, traverse_handler, leafs,
                              transforms, bound_areas, opacity, tint, ifmerge)
            self.layer_stack.pop()
        else:
            # count left top corner point
            xx = parent_xy.x + layer.frame.x
            yy = parent_xy.y + layer.frame.y
            xy = Point(xx, yy)
            # count center point for transformation
            center = Point(xx + layer.frame.width / 2,
                           yy + layer.frame.height / 2)
            # extend inherited transforms
            sub_transforms = transforms
            # since the Rotation is executed before the Flip,
            # and the final transform sequence will be excuted reversedly,
            # we need to reverse the order of the transforms
            if layer.isFlippedHorizontal or layer.isFlippedVertical:
                # since the Flip can be executed before with one time scale,
                # we put horizontal and vertical flip into single object
                sub_transforms = sub_transforms + \
                    (Scale.to_flip(layer.isFlippedHorizontal,
                                   layer.isFlippedVertical, center),)
            if layer.rotation != 0:
                # record the rotation angle and the center point
                # the sketch json"s angle is in count-clockwise, but shapely"s angle is in clockwise
                sub_transforms = sub_transforms + \
                    (Rotation(-layer.rotation, center),)
            sub_opacity = opacity
            if layer.style is not None and layer.style.contextSettings is not None:
                sub_opacity *= layer.style.contextSettings.opacity
            # traverse the group and draw
            if isinstance(layer, Group) or isinstance(layer, ShapeGroup):
                sub_bound_areas = bound_areas
                group_tint = tint
                if group_tint is None and layer.style is not None and layer.style.fills is not None and len(
                        layer.style.fills) > 0:
                    group_tint = layer.style.fills[len(
                        layer.style.fills) - 1].color
                self.layer_stack.append(layer)

                if traverse_handler.if_group_need_handle(layer):
                    transformed_layer = transform_layer(layer, Point(
                        xx, yy), sub_transforms, bound_areas, sub_opacity, group_tint)
                    traverse_handler.handle_group(
                        copy.deepcopy(transformed_layer), self.symbol_master_dict)

                for i in layer.layers:
                    clipping_mask = await self.traverse(
                        i, xy, traverse_handler, leafs, sub_transforms, sub_bound_areas, sub_opacity, group_tint, ifmerge=ifmerge or traverse_handler.if_group_need_handle(layer))
                    # if the layer has a clipping mask, add it to the bound_areas
                    if clipping_mask.hasClippingMask:
                        sub_bound_areas = sub_bound_areas + \
                            (clipping_mask.mask,)
                    # if the layer break the mask, recover all masks created in this group
                    if i.shouldBreakMaskChain:
                        sub_bound_areas = bound_areas
                self.layer_stack.pop()
            elif isinstance(layer, SymbolInstance) and layer.symbolID in self.symbol_master_dict.keys():
                # scale the symbol master
                sub_transforms = sub_transforms + \
                    (Scale(layer.scale if layer.scale is not None else 1, layer.scale if layer.scale is not None else 1, Point(xx, yy)),)
                # load real symbol from symbol link
                await self.traverse(
                    self.symbol_master_dict[layer.symbolID], xy, traverse_handler, leafs, sub_transforms, bound_areas,
                    sub_opacity, tint, ifmerge)
            else:
                # get bounding box

                transformed_layer = transform_layer(layer, Point(
                    xx, yy), sub_transforms, bound_areas, sub_opacity, tint)

                leafs.append(transformed_layer)
                traverse_handler.handle_layer(
                    copy.deepcopy(transformed_layer), ifmerge, layer.name)
                # return the clipping mask if the layer has ClippingMask and the polygon is not empty
                # print("s-----------------",transformed_layer.bound_area)
                if layer.hasClippingMask and len(get_polygon_exterior(transformed_layer.bound_area)) > 2:
                    return ClippingMask(True, transformed_layer.bound_area)
        # default return
        return ClippingMask(False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataDir", type=str, required=True)
    parser.add_argument("--outDir", type=str, required=True)
    args = parser.parse_args()

    import glob
    rootDir = args.dataDir
    os.makedirs(args.outDir, exist_ok=True)
    sketch_lst = glob.glob(rootDir+"/*.sketch")
    print("Total {%d} sketch files"%len(sketch_lst))   
    preProcess = PreProcess(args.outDir, sketch_lst, scale=1)
    preProcess.process()
