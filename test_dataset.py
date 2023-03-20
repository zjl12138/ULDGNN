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
    cfg.train.batch_size = 1
    cfg.test_dataset.rootDir = '../../dataset/new_graph_dataset'
    dataloader = make_data_loader(cfg,is_train=True)
    vis = visualizer(cfg.visualizer)

    #trainer.train(0, dataloader, optim, recorder, evaluator )
    for batch in tqdm(dataloader):
        #network(batch)
        nodes, edges, types,  img_tensor, labels, bboxes, file_list  = batch
        if(nodes.shape[0]>=2000):
            print(file_list)
   