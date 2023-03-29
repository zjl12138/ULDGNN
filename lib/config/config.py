from traitlets import default
from .yacs import CfgNode as CN
from . import yacs
import argparse
import os
import yaml

'''
--outDir
  |__exp_name
     |___records
     |___imgs
     |___checkpoints
     |___configs
'''

cfg = CN()

cfg.outDir = 'output'
cfg.exp_name=''
cfg.model_dir = ''

cfg.recorder = CN()
cfg.recorder.record_dir=''
cfg.visualizer = CN()
cfg.visualizer.vis_dir =''

cfg.train_dataset = CN()
cfg.train_dataset.module = 'lib.datasets.light_stage.graph_dataset'
cfg.train_dataset.path = 'lib/datasets/light_stage/graph_dataset.py'
cfg.train_dataset.rootDir = '../../dataset/graph_dataset_rerefine'
cfg.train_dataset.index_json = 'index_train.json'

cfg.test_dataset = CN()
cfg.test_dataset.module = 'lib.datasets.light_stage.graph_dataset'
cfg.test_dataset.path = 'lib/datasets/light_stage/graph_dataset.py'
cfg.test_dataset.rootDir = '../../dataset/graph_dataset_rerefine'
cfg.test_dataset.index_json = 'index_test.json'

cfg.train = CN()
cfg.train.save_best_acc=True
cfg.train.save_ep = 10
cfg.train.eval_ep = 10
cfg.train.vis_ep = 1000
cfg.train.epoch = 1000
cfg.train.lr = 7e-5
cfg.train.weight_decay = 1e-5
cfg.train.optim = 'adamw'
cfg.train.batch_size = 8
cfg.train.local_rank = 3
cfg.train.log_interval = 50
cfg.train.record_interval = 10
cfg.train.shuffle=True
cfg.train.scheduler = 'exponential'
cfg.train.milestones = [80, 120, 200, 240]
cfg.train.decay_epochs = 5
cfg.train.gamma = 0.99
cfg.train.resume = True

cfg.test = CN()
cfg.test.batch_size = 1
cfg.test.vis_bbox = True
cfg.test.val_nms = False

cfg.network = CN()
cfg.network.train_mode = 0  # mode 0: train two branches together, 1: only train cls branch, 2:only train loc branch

cfg.network.network_module = 'lib.networks.network'
cfg.network.network_path = 'lib/networks/network.py'

cfg.network.pos_embedder = CN()
cfg.network.pos_embedder.multires = 9
cfg.network.pos_embedder.out_dim=512

cfg.network.type_embedder = CN()
cfg.network.type_embedder.out_dim = 512
cfg.network.type_embedder.class_num = 11

cfg.network.img_embedder = CN()
cfg.network.img_embedder.outdim = 512
cfg.network.img_embedder.name='resnet50'

cfg.network.gnn_fn = CN()
cfg.network.gnn_fn.gnn_type = 'LayerGNN'
cfg.network.gnn_fn.latent_dims = [256,256]
cfg.network.gnn_fn.num_heads = [4,4,4] #len(num_heads)==len(latent_dims)+1
cfg.network.gnn_fn.out_dim = 256
cfg.network.gnn_fn.concat = True
cfg.network.gnn_fn.residual = True
cfg.network.gnn_fn.batch_norm = True

cfg.network.cls_fn = CN()
cfg.network.cls_fn.latent_dims = [256,256]
#cfg.network.cls_fn.latent_dims = [128,64]
cfg.network.cls_fn.act_fn = 'LeakyReLU'
cfg.network.cls_fn.norm_type = ''
cfg.network.cls_fn.classes = 2

cfg.network.loc_fn = CN()
cfg.network.loc_fn.latent_dims = [256,256]
#cfg.network.loc_fn.latent_dims = [128,64]
cfg.network.loc_fn.act_fn = 'LeakyReLU'
cfg.network.loc_fn.norm_type = ''
cfg.network.loc_fn.classes = 4

cfg.network.cls_loss = CN()
cfg.network.cls_loss.type = 'focal_loss'
cfg.network.cls_loss.alpha = 0.7
cfg.network.cls_loss.reduction = 'mean'
cfg.network.cls_loss.gamma = 2
cfg.network.cls_loss.weight = 1

cfg.network.reg_loss = CN()
cfg.network.reg_loss.type = 'ciou_loss'
cfg.network.reg_loss.reduction = 'mean'
cfg.network.reg_loss.delta = 0.5
cfg.network.reg_loss.weight = 10

def make_cfg(args):
    
    with open(args.cfg_file,'r') as f:
        current_cfg = yacs.load_cfg(f)
    if 'parent_cfg' in current_cfg.keys():
        with open(current_cfg.parent_cfg, 'r') as f:
            parent_cfg = yacs.load_cfg(f)
        cfg.merge_from_other_cfg(parent_cfg)
    cfg.exp_name=args.exp_name
    cfg.network.train_mode = args.train_mode
    cfg.merge_from_other_cfg(current_cfg)
   
    cfg.recorder.record_dir = os.path.join(cfg.outDir, cfg.exp_name, 'records')
    cfg.visualizer.vis_dir = os.path.join(cfg.outDir, cfg.exp_name, 'imgs')
    cfg.model_dir=os.path.join(cfg.outDir, cfg.exp_name,"checkpoints")
    cfg.config_dir = os.path.join(cfg.outDir, cfg.exp_name,"configs")


    print(cfg.model_dir)
    os.makedirs(cfg.recorder.record_dir,exist_ok=True)
    os.makedirs(cfg.visualizer.vis_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.config_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(cfg.config_dir,"config.yaml"),"w"))
    
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default = "configs/default.yaml",type=str)
parser.add_argument("--exp_name",type=str)
parser.add_argument('--train_mode', type=int, default=0)

args = parser.parse_args()
make_cfg(args)