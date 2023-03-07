from lib.datasets import make_data_loader
from lib.config import cfg
from lib.visualizers import visualizer
from lib.networks import make_network

if __name__=='__main__':
    dataloader = make_data_loader(cfg,is_train=False)
    vis = visualizer(cfg.visualizer)
    network = make_network(cfg.network)
    print(len(dataloader))
    for i, batch in enumerate(dataloader):
        nodes, edges, types,  img_tensor, labels, bboxes, file_list = batch  
        print(nodes.shape)
        print(bboxes.shape)
        network(batch)
        #vis.visualize(nodes, bboxes, file_list[0])