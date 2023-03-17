from lib.datasets import make_data_loader
from lib.config import cfg
from lib.visualizers import visualizer
from lib.networks import make_network
import torch
from lib.train import make_optimizer, make_recorder, make_scheduler, make_trainer
from lib.evaluators import Evaluator
from lib.utils import load_model, save_model
from tqdm import tqdm

if __name__=='__main__':
    cfg.test_dataset.rootDir = '../../dataset/graph_dataset_test'
    dataloader = make_data_loader(cfg,is_train=False)
    vis = visualizer(cfg.visualizer)

    #trainer.train(0, dataloader, optim, recorder, evaluator )
    for batch in tqdm(dataloader):
        #network(batch)
        nodes, edges, types,  img_tensor, labels, bboxes, file_list  = batch
        nodes = nodes[labels==1]
        bboxes = bboxes[labels==1]
        vis.visualize_gt(nodes, bboxes, file_list[0])
   