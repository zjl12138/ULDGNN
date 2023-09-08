from re import T
from lib.datasets import make_data_loader
from lib.config import cfg
from lib.visualizers import visualizer
from lib.networks import make_network
import torch
from lib.train import make_optimizer, make_recorder, make_scheduler, make_trainer
from lib.evaluators import Evaluator
from lib.utils import load_model, save_model,load_network

def test(cfg, network):
    trainer = make_trainer(cfg.train, network)
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
    
    trainer.val(begin_epoch, val_loader, evaluator, None, vis if cfg.test.vis_bbox else None, val_nms=cfg.test.val_nms, eval_merge = cfg.test.eval_merge, eval_ap = cfg.test.eval_ap)

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
    cfg.train.local_rank = 0
    cfg.test.vis_bbox = True
    cfg.test.eval_merge = False
    cfg.test.eval_ap = False
    cfg.test.val_nms = False
    # cfg.test_dataset.rootDir = '../../dataset/EGFE_graph_dataset_refine'
    # cfg.test_dataset.index_json = 'index_testv2.json'
    # cfg.test_dataset.bg_color_mode = 'bg_color_orig'
    print(cfg.test_dataset.index_json)
    print(cfg.test_dataset.rootDir)
    network = make_network(cfg.network)
    begin_epoch = load_network(network, cfg.model_dir, map_location = f'cuda:{cfg.train.local_rank}')
    #network.begi n_update_edge_attr()
    print("begin epoch: ", begin_epoch)
    test(cfg, network)