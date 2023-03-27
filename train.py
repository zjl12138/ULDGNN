from lib.utils.net_utils import load_network
from lib.datasets import make_data_loader
from lib.config import cfg
from lib.visualizers import visualizer
from lib.networks import make_network
import torch
from lib.train import make_optimizer, make_recorder, make_scheduler, make_trainer
from lib.evaluators import Evaluator
from lib.utils import load_model, save_model

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def train(cfg, network,begin_epoch=0):
    trainer = make_trainer(network)
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

    train_loader = make_data_loader(cfg, is_train=True)
    print("Training artboards: ",len(train_loader))
    val_loader = make_data_loader(cfg, is_train=False)
    print("validating artboards: ",len(val_loader))
    vis = visualizer(cfg.visualizer)
    best_epoch = -1
    best_acc = -1
    if begin_epoch > 0:
        best_epoch = begin_epoch
        val_metric_stats = trainer.val(begin_epoch, val_loader, evaluator, recorder, None)
        best_acc = val_metric_stats['accuracy']
        
    for epoch in range(begin_epoch, cfg.train.epoch):
        #trainer.val(epoch, val_loader, evaluator, recorder, None)

        recorder.epoch = epoch
        trainer.train(epoch, train_loader,optimizer,recorder,evaluator)
        scheduler.step() 
        #if (epoch+1) % cfg.train.save_ep == 0:
        #    save_model(network, optimizer,scheduler, recorder, cfg.model_dir,
        #               epoch, True)
        if (epoch+1) % cfg.train.eval_ep == 0:
            val_metric_stats = trainer.val(epoch, val_loader, evaluator, recorder, None)
            if val_metric_stats['accuracy'] > best_acc:
                print("model with best accuracy saving...")
                best_epoch = epoch
                best_acc = val_metric_stats['accuracy']
                save_model(network, optimizer,scheduler, recorder, cfg.model_dir, best_epoch, True)

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
    network = make_network(cfg.network)
    for n, v in network.named_parameters():
        if v.requires_grad:
            print(n)
    #begin_epoch = load_network(network,cfg.model_dir)
    train(cfg, network)