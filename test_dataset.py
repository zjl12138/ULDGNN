from lib.datasets.light_stage.graph_dataset import Dataset

from lib.config.yacs  import CfgNode as CN
from torch.utils.data import DataLoader

if __name__=='__main__':
    cfg = CN()
    cfg.rootDir='/media/sda1/ljz-workspace/dataset/graph_dataset'
    cfg.train_json = 'index_train.json'
    dataset = Dataset(cfg)
   
    for i, batch in enumerate(dataset):
        layer_rects, collate_edges, collate_types, collate_labls,bboxes, layer_assets=batch
        print(collate_edges)