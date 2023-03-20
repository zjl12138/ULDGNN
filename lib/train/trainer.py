from torch.nn import DataParallel
import torch
from lib.config import cfg as CFG
from lib.train.recorder import Recorder
import time
import datetime
import tqdm
from lib.evaluators.evaluator import Evaluator
from lib.visualizers.visualizer import visualizer
import torch.nn.functional as F

cfg = CFG.train

class Trainer(object):
    def __init__(self, network):
        self.device = torch.device(f'cuda:{cfg.local_rank}')
        network = network.to(self.device)
        self.network = network
        self.local_rank = cfg.local_rank

    def to_cuda(self, batch):
        for k, _ in enumerate(batch):
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch[k]]
            else:
                batch[k] = batch[k].to(self.device) if isinstance( batch[k], torch.Tensor) else batch[k]
        
        return batch
    
    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def train(self, epoch, data_loader, optimizer, recorder:Recorder, evaluator:Evaluator):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration += 1
            batch = self.to_cuda(list(batch))

            output, loss, loss_stats = self.network(batch)

            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            metric_stats = evaluator.evaluate(output, batch[4])
            recorder.update_loss_stats(metric_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

            if iteration % cfg.record_interval == 0 or iteration == (max_iter - 1):
                # record loss_stats and image_dict
                recorder.record('train')
    
    def val(self, epoch, data_loader, evaluator:Evaluator, recorder:Recorder, visualizer:visualizer=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        val_metric_stats = {}
        pred_list = []
        label_list = []
        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(list(batch))
            layer_rects, edges, types,  img_tensor, labels, bboxes, file_list = batch
            with torch.no_grad():
                output, loss, loss_stats = self.network(batch)
                val_stats = evaluator.evaluate(output, batch[4])
                logits, local_params = output
                '''pred_list.append(logits)
                label_list.append(labels)
                '''
                loss_stats = self.reduce_loss_stats(loss_stats)
                
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v
            for k, v in val_stats.items():
                val_metric_stats.setdefault(k, 0)
                val_metric_stats[k] += v
            
            if visualizer is not None:
                logits, local_params = output
                scores, pred = torch.max(F.softmax(logits,dim=1), 1)
            
                #print(layer_rects.shape, pred.shape, labels)
                pred_fraglayers = layer_rects[labels==1]
                pred_merging_groups = local_params[labels==1]
                scores = scores [labels==1]
                visualizer.visualize_pred(pred_fraglayers, pred_merging_groups,batch[6][0])
                visualizer.visualize_nms(scores.cpu(), pred_fraglayers.cpu(),pred_merging_groups.cpu(),batch[6][0])
                
                fragmented_layers = layer_rects[labels==1]
                merging_groups = bboxes[labels == 1 ]
                visualizer.visualize_gt(fragmented_layers, merging_groups, batch[6][0])
                #visualizer.visualize_nms(scores.cpu(), fragmented_layers.cpu(), merging_groups.cpu(),batch[6][0])
                
        
        #val_metric_stats = evaluator.evaluate((torch.cat(pred_list),None), torch.cat(label_list))

        loss_state = []
        metric_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        for k in val_metric_stats.keys():
            val_metric_stats[k] /= data_size
            metric_state.append('{}: {:.4f}'.format(k, val_metric_stats[k]))
        print(loss_state,metric_state)

        if recorder:
            recorder.record('val', epoch, val_loss_stats)
            recorder.record('val', epoch, val_metric_stats)

def make_trainer(network):
    return Trainer(network)