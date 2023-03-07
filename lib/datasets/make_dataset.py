from lib.config.config  import cfg
import imp
import time
import torch
from torch.utils.data import RandomSampler, BatchSampler,SequentialSampler,DataLoader
from lib.datasets.light_stage.graph_dataset import collate_fn
import numpy as np

def _dataset_factory(is_train):
    if is_train:
        module = cfg.train_dataset.module
        path = cfg.train_dataset.path
        
    else:
        module = cfg.test_dataset.module
        path = cfg.test_dataset.path
        
    dataset = imp.load_source(module,path).Dataset(cfg.train_dataset)
    return dataset

def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))

def make_dataset(cfg, is_train=True):
    return _dataset_factory(is_train)

def make_sampler(batch_size, is_train, dataset,drop_last):
    if is_train:
        return BatchSampler( RandomSampler(dataset),batch_size, drop_last)
    else:
        return BatchSampler(SequentialSampler(dataset))

def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.train.batch_size
        drop_last = False
        shuffle = cfg.train.shuffle
    else:
        batch_size = cfg.test.batch_size
        drop_last = False
        shuffle = False
    
    dataset = make_dataset(cfg.train_dataset)
    sampler = make_sampler(batch_size, is_train, dataset, drop_last)
    data_loader = DataLoader(dataset, 
                            batch_size,
                            sampler=sampler,
                            collate_fn=collate_fn,
                            worker_init_fn=worker_init_fn)
    return data_loader


