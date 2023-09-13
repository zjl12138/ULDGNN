import sys
sys.path.append("..")

from lib.datasets import make_data_loader
from lib.config import cfg
from lib.visualizers import comp_det_visualizer
from lib.networks import make_network
import torch
from lib.train import make_optimizer, make_recorder, make_scheduler
from lib.train import make_trainer
from lib.evaluators.comp_det_eval import Evaluator
from lib.utils import load_model, save_model
import torch.distributed as dist
import os

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def train(cfg, network, begin_epoch = 0):
    trainer = make_trainer(cfg.train, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg.recorder)
    evaluator = Evaluator()
    
    begin_epoch = load_model(network, 
                            optimizer,
                            scheduler,
                            recorder, 
                            cfg.model_dir, 
                            cfg.train.resume)
    
    train_loader = make_data_loader(cfg, is_train = True, is_distributed = cfg.train.is_distributed)
    print("Training artboards: ", len(train_loader))
    val_loader = make_data_loader(cfg, is_train = False)
    print("validating artboards: ", len(val_loader))
    vis = comp_det_visualizer(cfg.visualizer)
    best_epoch = -1
    best_acc = -1
    best_merging_acc = {"accuracy": -1}
    
    if begin_epoch > 0 and cfg.train.save_best_acc: 
        if cfg.train.is_distributed:
            if cfg.train.local_rank == 0:
                best_epoch = begin_epoch
                val_metric_stats = trainer.val(begin_epoch, val_loader, evaluator, recorder, None, eval_merge = True)
                for k, v in best_merging_acc.items():
                    best_merging_acc[k] = val_metric_stats[k]
        else:
            best_epoch = begin_epoch
            val_metric_stats = trainer.val(begin_epoch, val_loader, evaluator, recorder, None)
            for k, v in best_merging_acc.items():
                best_merging_acc[k] = val_metric_stats[k]
    
    '''
    if not cfg.train.is_distributed or cfg.train.local_rank == 0:
            val_metric_stats = trainer.val(epoch, val_loader, evaluator, recorder,
                                               vis if cfg.test.vis_bbox else None, False)
    '''
    #network.begin_update_edge_attr()
    for epoch in range(begin_epoch, cfg.train.epoch):
        #trainer.val(epoch, val_loader, evaluator, recorder, None)
        '''
        if cfg.train.begin_update_edge_attr_epoch >= 0 and epoch == cfg.train.begin_update_edge_attr_epoch:
            print("start updating edge_attr!")
            network.begin_update_edge_attr()
        '''
        recorder.epoch = epoch
        if cfg.train.is_distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)
        trainer.train(epoch, train_loader, optimizer, recorder, evaluator)
        scheduler.step()
        
        #if (epoch+1) % cfg.train.save_ep == 0:
        #    save_model(network, optimizer,scheduler, recorder, cfg.model_dir,
        #               epoch, True)
        
        if (epoch + 1) % cfg.train.eval_ep == 0:
            if (not cfg.train.is_distributed) or (cfg.train.local_rank == 0):
                val_metric_stats = trainer.val(epoch, val_loader, evaluator, recorder,
                                                vis if cfg.test.vis_bbox else None, False)       
                if cfg.train.save_best_acc:
                    for k, v in best_merging_acc.items():
                        if val_metric_stats[k] >= v:
                            print(f"model with best {k} saving...")
                            best_merging_acc[k] = val_metric_stats[k]
                            save_model(network, optimizer, scheduler, recorder, cfg.model_dir,
                                        epoch, True)

                print("saving model...")
                save_model(network, optimizer, scheduler, recorder, cfg.model_dir, epoch, False)

        #if (epoch+1) % cfg.train.vis_ep == 0:
        #    trainer.val(epoch, val_loader, evaluator, recorder, vis)
        

if __name__ == '__main__':
    if cfg.train.is_distributed:
        print("distributed training! using device ", int(os.environ['RANK']))
        
        cfg.train.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.train.local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://")
        synchronize()
    network = make_network(cfg.network)
    if cfg.train.is_distributed:
        network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
    '''
    print("trained parameters----------------------------")       
    for n, v in network.named_parameters():
        if v.requires_grad:
            print(n)
    '''
    total_sum = sum(p.numel() for p in network.parameters())
    print("Number of parameters: ", total_sum / 1024 / 1024)
    '''
    if cfg.train.load_all_pretrained:
        begin_epoch = load_network(network, cfg.model_dir)
    else:
        begin_epoch = load_partial_network(network, cfg.model_dir)
    '''
    # print(begin_epoch)
    train(cfg, network)