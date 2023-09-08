from lib.config.config  import cfg
import imp
import time
import torch
from torch.utils.data import RandomSampler, BatchSampler, SequentialSampler ,DataLoader, Sampler

import numpy as np
import torch.distributed as dist
import math

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset+self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def _dataset_factory(is_train):
    if is_train:
        module = cfg.train_dataset.module
        path = cfg.train_dataset.path
        print(f"makeing training dataset from {cfg.train_dataset.rootDir}")
        return imp.load_source(module,path).Dataset(cfg.train_dataset)
   
    else:
        print(f"makeing test dataset from {cfg.test_dataset.rootDir}")
        module = cfg.test_dataset.module
        path = cfg.test_dataset.path
        return imp.load_source(module,path).Dataset(cfg.test_dataset)
    return None

def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))

def make_dataset(is_train=True):
    return _dataset_factory(is_train)

def make_sampler(batch_size, is_train, dataset, drop_last, shuffle, is_distributed = False):
    if is_train:
        if is_distributed:
            print("distributed dataset!")
            return BatchSampler( DistributedSampler(dataset, shuffle = shuffle),batch_size, drop_last)
        return BatchSampler( RandomSampler(dataset), batch_size, drop_last)
    else:
        return BatchSampler( SequentialSampler(dataset), batch_size, drop_last)

def make_data_loader(cfg, is_train = True, is_distributed = False):
    if is_train:
        batch_size = cfg.train.batch_size
        drop_last = False
        shuffle = cfg.train.shuffle
        dataset = make_dataset(True)
    else:
        batch_size = cfg.test.batch_size
        drop_last = False
        shuffle = False
        dataset = make_dataset(False)
 
    sampler = make_sampler(batch_size, is_train, dataset, drop_last, shuffle,  is_distributed)
    data_loader = DataLoader(dataset, 
                            batch_sampler=sampler,
                            collate_fn = dataset.collate_fn,
                            worker_init_fn=worker_init_fn)
    return data_loader


