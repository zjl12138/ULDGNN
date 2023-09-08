import os
import imp

def make_trainer(cfg, network):
    module = cfg.train_module
    path = cfg.train_path
    trainer = imp.load_source(module, path).Trainer(network)
    return trainer