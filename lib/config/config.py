from .yacs import CfgNode as CN
from . import yacs
import argparse
import os
cfg = CN()

'''
--outDir
  |__exp_name
     |___records
     |___imgs
'''

cfg.outDir = ''
cfg.exp_name=''

cfg.recorder = CN()
cfg.recorder.record_dir=''
cfg.visualizer = CN()
cfg.visualizer.vis_dir =''

cfg.train_dataset = CN()
cfg.train_dataset.module = ''
cfg.train_dataset.path = ''
cfg.train_dataset.rootDir = ''
cfg.train_dataset.train_json = ''

cfg.test_dataset = CN()
cfg.test_dataset.module = ''
cfg.test_dataset.path = ''
cfg.test_dataset.rootDir = ''
cfg.test_dataset.train_json = ''

cfg.train = CN()
cfg.train.batch_size = 4
cfg.train.local_rank = 0
cfg.train.log_interval = 10
cfg.train.record_interval = 10

cfg.network = CN()
cfg.network.network_module = ''
cfg.network.network_path = ''
cfg.network.pos_embedder = CN()
cfg.network.pos_embedder.multires=10
cfg.network.pos_embedder.out_dim=256

cfg.network.type_embedder = CN()
cfg.network.type_embedder.out_dim = 256
cfg.network.type_embedder.class_num = 10 

cfg.network.img_embedder = CN()
cfg.network.img_embedder.outdim = 256
cfg.network.img_embedder.name='resnet50'

cfg.network.gnn_fn = CN()
cfg.network.gnn_fn.latent_dims = []
cfg.network.gnn_fn.num_heads = [] #len(num_heads)==len(latent_dims)+1
cfg.network.gnn_fn.out_dim = 256
cfg.network.gnn_fn.concat = True
cfg.network.gnn_fn.residual = True
cfg.network.gnn_fn.batch_norm = True

cfg.network.cls_fn = CN()
cfg.network.cls_fn.latent_dims = []
cfg.network.cls_fn.act_fn = 'LeakyReLU'
cfg.network.cls_fn.norm_type = 'BatchNorm1d'

cfg.network.loc_fn = CN()
cfg.network.loc_fn.latent_dims = []
cfg.network.loc_fn.act_fn = 'LeakyReLU'
cfg.network.loc_fn.norm_type = 'BatchNorm1d'

cfg.network.cls_loss = CN()
cfg.network.cls_loss.type = 'focal_loss'
cfg.network.cls_loss.alpha = 0.5
cfg.network.cls_loss.reduction = 'mean'
cfg.network.cls_loss.gamma = 0.01
cfg.network.cls_loss.weight = 10

cfg.network.reg_loss = CN()
cfg.network.reg_loss.type = 'huber_loss'
cfg.network.reg_loss.reduction = 'mean'
cfg.network.reg_loss.delta = 0.5
cfg.network.reg_loss.weight = 1

def make_cfg(args):
    with open(args.cfg_file,'r') as f:
        current_cfg = yacs.load_cfg(f)
    if 'parent_cfg' in current_cfg.keys():
        with open(current_cfg.parent_cfg, 'r') as f:
            parent_cfg = yacs.load_cfg(f)
        cfg.merge_from_other_cfg(parent_cfg)

    cfg.merge_from_other_cfg(current_cfg)
    cfg.recorder.record_dir = os.path.join(cfg.outDir, cfg.exp_name, 'records')
    cfg.visualizer.vis_dir = os.path.join(cfg.outDir, cfg.exp_name, 'imgs')

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default = "configs/default.yaml",type=str)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
cfg = make_cfg(args)