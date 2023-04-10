from lib.datasets import make_data_loader
from lib.config import cfg
from lib.visualizers import visualizer
from lib.networks import make_network
import torch
from lib.train import make_optimizer, make_recorder, make_scheduler, make_trainer
from lib.evaluators import Evaluator
from lib.utils import load_model, save_model
from tqdm import tqdm
from torch_geometric.utils import to_dense_batch

if __name__=='__main__':
    cfg.train.batch_size = 1
    cfg.train_dataset.rootDir = '../../dataset/graph_dataset_rerefine_large_graph'
    cfg.train_dataset.index_json = 'index.json'
    
    dataloader = make_data_loader(cfg,is_train=True)
    vis = visualizer(cfg.visualizer)
    print(len(dataloader))
    #trainer.train(0, dataloader, optim, recorder, evaluator )
    positives = 0
    negatives = 0
    for batch in tqdm(dataloader):
        #network(batch)
        nodes, edges, types, img_tensor, labels, bboxes, node_indices, file_list  = batch
        #print(node_indices)
        positives += torch.sum(labels)
        negatives += labels.shape[0]-torch.sum(labels)
        if nodes.shape[0]>1001:
            print(file_list)
    print("positive: ", positives, "negative: ", negatives)