from lib.datasets import make_data_loader
from lib.config import cfg
from lib.visualizers import visualizer
from lib.networks import make_network
import torch
from lib.train import make_optimizer, make_recorder, make_scheduler, make_trainer
from lib.evaluators import Evaluator
if __name__=='__main__':
    dataloader = make_data_loader(cfg,is_train=False)
    vis = visualizer(cfg.visualizer)
    network = make_network(cfg.network)
    optim = make_optimizer(cfg,network)
    sched = make_scheduler(cfg,optim)
    trainer = make_trainer(network)
    recorder = make_recorder(cfg.recorder)
    evaluator = Evaluator()

    trainer.train(0, dataloader, optim, recorder, evaluator )
        #network(batch)
        #vis.visualize(nodes, bboxes, file_list[0])