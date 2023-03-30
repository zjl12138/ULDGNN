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
    
    trainer.check_with_human_in_loop(begin_epoch, val_loader, evaluator, recorder, vis, val_nms=cfg.test.val_nms)

if __name__=='__main__':
   
    network = make_network(cfg.network)
    begin_epoch = load_network(network,cfg.model_dir)
    print("begin epoch: ", begin_epoch)
    test(cfg, network)