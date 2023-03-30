from lib.utils.net_utils import load_network
from lib.datasets import make_data_loader
from lib.config import cfg
from lib.visualizers import visualizer
from lib.networks import make_network
import torch
from lib.train import make_optimizer, make_recorder, make_scheduler, make_trainer
from lib.evaluators import Evaluator
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

def train(cfg, network, begin_epoch=0):
    trainer = make_trainer(network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg.recorder)
    evaluator = Evaluator()
    
    '''begin_epoch = load_model(network, 
                            optimizer,
                             scheduler,
                            recorder, 
                            cfg.model_dir, 
                            cfg.train.resume)
    '''
    train_loader = make_data_loader(cfg, is_train=True, is_distributed=cfg.train.is_distributed)
    print("Training artboards: ", len(train_loader))
    val_loader = make_data_loader(cfg, is_train=False)
    print("validating artboards: ", len(val_loader))
    vis = visualizer(cfg.visualizer)
    best_epoch = -1
    best_acc = -1
    if begin_epoch > 0 and cfg.train.save_best_acc: 
        best_epoch = begin_epoch
        val_metric_stats = trainer.val(begin_epoch, val_loader, evaluator, recorder, None)
        best_acc = val_metric_stats['accuracy']
        
    for epoch in range(begin_epoch, cfg.train.epoch):
        #trainer.val(epoch, val_loader, evaluator, recorder, None)

        recorder.epoch = epoch
        if cfg.train.is_distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)
        trainer.train(epoch, train_loader, optimizer, recorder, evaluator)
        scheduler.step() 
        #if (epoch+1) % cfg.train.save_ep == 0:
        #    save_model(network, optimizer,scheduler, recorder, cfg.model_dir,
        #               epoch, True)
        if (epoch+1) % cfg.train.eval_ep == 0 and cfg.local_rank == 0:
            if cfg.train.save_best_acc:
                val_metric_stats = trainer.val(epoch, val_loader, evaluator, recorder, None, False)
             
                if val_metric_stats['accuracy'] > best_acc:
                    print("model with best accuracy saving...")
                    best_epoch = epoch
                    best_acc = val_metric_stats['accuracy']
                    save_model(network, optimizer, scheduler, recorder, cfg.model_dir, best_epoch, True)
            else:    
                    print("saving model...")
                    save_model(network, optimizer, scheduler, recorder, cfg.model_dir, best_epoch, True)

        #if (epoch+1) % cfg.train.vis_ep == 0:
        #    trainer.val(epoch, val_loader, evaluator, recorder, vis)
        

if __name__=='__main__':
    '''dataloader = make_data_loader(cfg,is_train=False)
    vis = visualizer(cfg.visualizer)
    network = make_network(cfg.network)
    print(network)
    optim = make_optimizer(cfg, network)
    sched = make_scheduler(cfg, optim)
    trainer = make_trainer(network)
    recorder = make_recorder(cfg.recorder)
    evaluator = Evaluator()
    
    
    trainer.train(0, dataloader, optim, recorder, evaluator )
        #network(batch)
        #vis.visualize(nodes, bboxes, file_list[0])
    '''
    if cfg.train.is_distributed:
        print("distributed training!")
        cfg.train.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.train.local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://")
        synchronize()
    network = make_network(cfg.network)
    if cfg.train.is_distributed:
        network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
            
    for n, v in network.named_parameters():
        if v.requires_grad:
            print(n)
    begin_epoch = load_network(network, cfg.model_dir)
    train(cfg, network, begin_epoch)