from lib.datasets import make_data_loader
from lib.config import cfg
from tqdm import tqdm 

if __name__=='__main__':
    cfg.test_dataset.module = 'lib.datasets.light_stage.rico_graph_dataset'
    cfg.test_dataset.path = 'lib/datasets/light_stage/rico_graph_dataset.py'
    cfg.test.batch_size = 1
    cfg.test_dataset.rootDir = '../../dataset/rico_graph'
    cfg.test_dataset.index_json = 'index_dev.json'
    cfg.test_dataset.bg_color_mode = 'bg_color_orig'
    dataloader = make_data_loader(cfg,is_train=False)
    
    for batch in tqdm(dataloader):
        assets_img, layer_rect, edges, bbox, labels, node_indices, artboard_id = batch
        # print(artboard_id)
        