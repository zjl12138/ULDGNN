from lib.datasets import make_data_loader
from lib.config import cfg
from lib.visualizers import visualizer
from lib.networks import make_network
import torch
from lib.train import make_optimizer, make_recorder, make_scheduler, make_trainer
from lib.evaluators import Evaluator
from lib.utils import load_model, save_model

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def train(cfg, network):
    trainer = make_trainer(network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg.recorder)
    evaluator = Evaluator()
    begin_epoch = load_model(network, 
                            optimizer,
                             scheduler,
                            recorder, 
                            cfg.model_dir, 
                            cfg.train.resume)

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)
    vis = visualizer(cfg.visualizer)
    for epoch in range(begin_epoch, cfg.train.epoch):
        #trainer.val(epoch, val_loader, evaluator, recorder, None)

        recorder.epoch = epoch
        trainer.train(epoch, train_loader,optimizer,recorder,evaluator)
        scheduler.step() 
        if (epoch+1) % cfg.train.save_ep == 0:
            save_model(network, optimizer,scheduler, recorder, cfg.model_dir,
                       epoch, False)
        if (epoch+1) % cfg.train.eval_ep == 0:
            trainer.val(epoch, val_loader, evaluator, recorder, None)
        
        elif (epoch+1) % cfg.train.vis_ep == 0:
            trainer.val(epoch, val_loader, evaluator, recorder, vis)
        

if __name__=='__main__':
    '''dataloader = make_data_loader(cfg,is_train=False)
    vis = visualizer(cfg.visualizer)
    network = make_network(cfg.network)
    print(network)
    optim = make_optimizer(cfg, network)
    sched = make_scheduler(cfg, optim)
    trainer = make_trainer(network)
    recorder = make_recorder(cfg.recorder)
    evaluator = Evaluator()
    
    
    trainer.train(0, dataloader, optim, recorder, evaluator )
        #network(batch)
        #vis.visualize(nodes, bboxes, file_list[0])
    '''
    network = make_network(cfg.network)
    train(cfg, network)