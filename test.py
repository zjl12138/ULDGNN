from lib.datasets import make_data_loader
from lib.config import cfg
from lib.visualizers import visualizer
from lib.networks import make_network
import torch
from lib.train import make_optimizer, make_recorder, make_scheduler, make_trainer
from lib.evaluators import Evaluator
from lib.utils import load_model, save_model,load_network

def test(cfg, network):
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
    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)
    vis = visualizer(cfg.visualizer)
    
    trainer.val(begin_epoch, val_loader, evaluator, recorder, vis if cfg.test.vis_bbox else None, val_nms=cfg.test.val_nms)

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
    #cfg.test.vis_bbox = True
    cfg.train.is_distributed = False
    cfg.train.local_rank = 2
    cfg.test.vis_bbox = True
    network = make_network(cfg.network)
    begin_epoch = load_network(network, cfg.model_dir)
    network.begin_update_edge_attr()
    print("begin epoch: ", begin_epoch)
    test(cfg, network)