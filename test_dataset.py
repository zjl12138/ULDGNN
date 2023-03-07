from lib.datasets import make_data_loader
from lib.config import cfg

if __name__=='__main__':
    dataloader = make_data_loader(cfg)
    for i, batch in enumerate(dataloader):
        layer_rects, collate_edges, collate_types, collate_labls,bboxes, layer_assets, _=batch
        