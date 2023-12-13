from lib.datasets import make_data_loader
from lib.config import cfg
from tqdm import tqdm 
import torch
from lib.visualizers import comp_det_visualizer
from torch_geometric.data import Data

if __name__=='__main__':
    cfg.test_dataset.module = 'lib.datasets.light_stage.rico_graph_dataset'
    cfg.test_dataset.path = 'lib/datasets/light_stage/rico_graph_dataset.py'
    cfg.test.batch_size = 1
    cfg.test_dataset.rootDir = '../../dataset/rico_graph'
    cfg.test_dataset.index_json = 'index_test.json'
    cfg.test_dataset.bg_color_mode = 'bg_color_orig'
    dataloader = make_data_loader(cfg,is_train=False)
    vis = comp_det_visualizer(cfg.visualizer)
    all_labels = []
    for batch in tqdm(dataloader):
        assets_img, layer_rect, edges, bbox, labels, node_indices, artboard_id = batch
        # print(artboard_id)
        all_labels.append(labels)
        # vis.visualize_recons_artboard(layer_rect, assets_img, artboard_id[0])
        graph_data = Data(layer_rect, edges)
        if graph_data.has_isolated_nodes():
            print(artboard_id)
        if 18 in labels.numpy().tolist():
            print(artboard_id)
    # 
    # all_labels is like [0, 0, 1, 1, ...]
    # I want to know the number of each class label#
    all_labels = torch.cat(all_labels)
    counts = []
    for  i in range(24):
        print(i, (all_labels == i).sum())
        counts.append((all_labels == i).sum())
    
    frequency = torch.tensor(counts).float() / all_labels.size(0)
        
    # Inverting the frequency to use as weights for weighted cross entropy
    weights = 1.0 / frequency
    weights /= weights.sum()
    print(weights, torch.sum(weights))

