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
    cfg.train.batch_size = 4
    cfg.test_dataset.rootDir = '../../dataset/new_graph_dataset'
    dataloader = make_data_loader(cfg,is_train=True)
    vis = visualizer(cfg.visualizer)
    print(len(dataloader))
    #trainer.train(0, dataloader, optim, recorder, evaluator )
    positives = 0
    negatives = 0
    for batch in tqdm(dataloader):
        #network(batch)
        nodes, edges, types,  img_tensor, labels, bboxes, file_list  = batch
        positives += torch.sum(labels)
        negatives += labels.shape[0]-torch.sum(labels)
    print("positive: ", positives, "negative: ", negatives)