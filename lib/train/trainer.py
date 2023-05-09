from locale import DAY_1
import json
from symbol import eval_input
from torch.nn import DataParallel
import torch
from lib.utils.nms import contains, refine_merging_bbox
from lib.config import cfg as CFG
from lib.train.recorder import Recorder
import time
import datetime
import tqdm
from lib.evaluators.evaluator import Evaluator
from lib.visualizers.visualizer import visualizer
import torch.nn.functional as F
from lib.utils import nms_merge
import matplotlib.pyplot as plt
import os
from lib.utils import correct_dataset
import numpy as np
cfg = CFG.train

def clip_val(x, lower, upper):
    x = x if x >= lower else lower
    x = x if x <= upper else upper
    return x


def scale_to_img(x , H, W):
    x[0] = clip_val(x[0], 0, 1)
    x[1] = clip_val(x[1], 0, 1)
    x[2] = clip_val(x[2], 0, 1)
    x[3] = clip_val(x[3], 0, 1)
    return [x[0] * W, x[1] * H, x[0] * W + x[2] * W, x[1] * H + x[3] * H]
    #return [int(x[0] * W), int(x[1] * H), int(x[0] * W) + int(x[2] * W), int(x[1] * H) + int(x[3] * H)]
        
class Trainer(object):
    def __init__(self, network):
        
        self.device = torch.device(f'cuda:{cfg.local_rank}')
        network = network.to(self.device)
        self.local_rank = cfg.local_rank
        if cfg.is_distributed:
            print("distributed network!")
            network = torch.nn.parallel.DistributedDataParallel(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank
            )
        self.network = network
        self.anchor_box_wh = torch.tensor([[16.0, 8.0], [16.0, 16.0], [16.0, 32.0],
                                           [64.0, 32.0], [64.0, 64.0], [64.0, 128.0],
                                           [256.0, 128.0], [256.0, 256.0], [512.0, 512.0]], dtype = torch.float32).to(self.device) / 750.0

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

            output, loss, loss_stats = self.network(batch, self.anchor_box_wh)  #output: (logits, centers, bboxes)

            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            logits, _, _, _, _ = output #output: (logits, centers, bboxes, confidence, center_offset)

            _, pred = torch.max(F.softmax(logits,dim=1), 1)

            metric_stats = evaluator.evaluate(pred, batch[4])
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
    
    def process_output_data(self, output, layer_rects_gt, labels, bboxes_gt):
        logits, centers, local_params, confidence, _ = output #output: (logits, centers, bboxes)
        cls_scores, pred = torch.max(F.softmax(logits, dim = 1), 1)

        fragmented_layers_gt = layer_rects_gt[labels == 1]
        fragmented_layers_pred = layer_rects_gt[pred == 1]
        '''
        merging_groups_pred = local_params[labels == 1] 
        scores = scores[labels == 1]
        '''
        merging_groups_pred = local_params[pred == 1] 
        
        merging_groups_pred_mask = contains(merging_groups_pred, layer_rects_gt[pred == 1])

        if confidence is None:
            scores = cls_scores[pred == 1]
            merging_groups_pred_nms, merging_groups_confidence = nms_merge(merging_groups_pred, scores, threshold=0.45)

        else:
            scores = confidence[pred == 1]
            mask = torch.logical_and(scores >= 0.8, merging_groups_pred_mask) 
            merging_groups_pred_nms, merging_groups_confidence = nms_merge(merging_groups_pred[mask], scores[mask], threshold=0.4)

        merging_groups_gt = bboxes_gt[labels == 1] + fragmented_layers_gt
    
        return {
                'fragmented_layers_gt': fragmented_layers_gt,
                'fragmented_layers_pred': fragmented_layers_pred,
                'merging_groups_gt': merging_groups_gt,
                'merging_groups_pred': merging_groups_pred,
                'merging_groups_pred_nms': merging_groups_pred_nms,
                'label_pred': pred,
                'centers_pred': centers[pred==1, :],
                'merging_groups_confidence': merging_groups_confidence
               }

    def val(self, epoch, data_loader, evaluator:Evaluator, recorder:Recorder, visualizer:visualizer=None, val_nms=False, eval_merge = False, eval_ap = False):

        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        val_metric_stats = {}
        pred_list = []
        label_list = []
        check_records = {}
        eval_merging_acc = 0.0

        gt_annotations = []
        det_results = []
        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(list(batch))
            layer_rects, edges, types,  img_tensor, labels, bboxes, node_indices, file_list = batch
            with torch.no_grad():
                output, loss, loss_stats = self.network(batch, self.anchor_box_wh)
                #val_stats = evaluator.evaluate(output, batch[4])
                loss_stats = self.reduce_loss_stats(loss_stats)
                
                for k, v in loss_stats.items():
                    val_loss_stats.setdefault(k, 0)
                    val_loss_stats[k] += v
                
                '''for k, v in val_stats.items():
                    val_metric_stats.setdefault(k, 0)
                    val_metric_stats[k] += v
                '''
                fetch_data = self.process_output_data(output, layer_rects, labels, bboxes)
                fragmented_layers_gt = fetch_data['fragmented_layers_gt']
                fragmented_layers_pred = fetch_data['fragmented_layers_pred']
                merging_groups_gt = fetch_data['merging_groups_gt']
                merging_groups_pred = fetch_data['merging_groups_pred']
                merging_groups_pred_nms = fetch_data['merging_groups_pred_nms']
                merging_groups_confidence = fetch_data['merging_groups_confidence']
                
                label_pred = fetch_data['label_pred']
                merging_groups_gt_nms, _ = nms_merge(merging_groups_gt, torch.ones(merging_groups_gt.shape[0], device = merging_groups_gt.get_device()))
                
                if val_nms:
                    #prev_pred = pred
                    label_pred = evaluator.correct_pred_with_nms(label_pred, merging_groups_pred_nms, layer_rects, types, threshold=0.45) 
                    fetch_data['label_pred'] = label_pred
                    #print(f"correct {torch.sum(pred!=prev_pred)}wrong preditions")
                
                if eval_merge:
                    eval_merging_acc = eval_merging_acc + evaluator.evaluate_merging(merging_groups_pred_nms, label_pred, layer_rects, bboxes + layer_rects, labels)

                if eval_ap:
                    if len(merging_groups_pred_nms) and len(merging_groups_gt_nms):
                        
                        file_path, artboard_name = os.path.split(batch[7][0])
                        artboard_name = artboard_name.split(".")[0]
                        data_dict = json.load(open(f'/media/sda1/ljz-workspace/dataset/graph_dataset_rerefine_copy/{artboard_name}/{artboard_name}-{0}.json'))
                        W, H = data_dict["artboard_width"], data_dict["artboard_height"]

                        gt_info = {
                            'bboxes': np.vstack([np.array(scale_to_img(x.cpu().numpy(), H, W)) for x in merging_groups_gt_nms]),
                            'bboxes_ignore': None,
                            'labels': np.zeros(len(merging_groups_gt_nms), dtype = np.int64),
                            'labels_ignore': None
                        }
                        gt_annotations.append(gt_info)
                        merging_groups_pred_nms_refine, merging_groups_confidence_new = refine_merging_bbox(torch.vstack(merging_groups_pred_nms), layer_rects, label_pred, merging_groups_confidence, categories = types)
                        det_results.append([np.vstack([scale_to_img(t.cpu().numpy(), H, W) for t in merging_groups_pred_nms_refine])])
                        
                        #det_results.append([np.vstack([scale_to_img(t.cpu().numpy(), H, W) for t in merging_groups_pred_nms])])
            
                centers_pred = fetch_data['centers_pred']
                
                pred_list.append(label_pred)
                label_list.append(labels) 

                if visualizer is not None:
                    '''
                    visualizer.visualize_pred(fragmented_layers_pred, merging_groups_pred, batch[7][0])
                    #visualizer.visualize_nms(merging_groups_pred_nms, batch[7][0])
                    visualizer.visualize_nms_with_labels(merging_groups_pred_nms, merging_groups_confidence, batch[7][0])
                    
                    if len(merging_groups_pred_nms) != 0:
                        merging_groups_pred_nms, merging_groups_confidence = refine_merging_bbox(torch.vstack(merging_groups_pred_nms), layer_rects, label_pred, merging_groups_confidence, types)
                        visualizer.visualize_nms_with_labels(merging_groups_pred_nms, merging_groups_confidence, batch[7][0], mode = "bbox_refine")
                        #det_results.append([np.vstack([scale_to_img(t.cpu().numpy(), H, W) for t in merging_groups_pred_nms])])
                        
                    visualizer.visualize_gt(fragmented_layers_gt, merging_groups_gt, batch[7][0])
                    visualizer.visualize_offset_of_centers(centers_pred, fragmented_layers_pred, batch[7][0])
                    '''
                    visualizer.visualize_overall(fetch_data, layer_rects, types, batch[7][0])
        val_metric_stats = evaluator.evaluate(torch.cat(pred_list), torch.cat(label_list))

        loss_state = []
        metric_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        for k in val_metric_stats.keys():
            #val_metric_stats[k] /= data_size
            metric_state.append('{}: {:.4f}'.format(k, val_metric_stats[k]))
        print(loss_state, metric_state, [{"merging_acc": eval_merging_acc / data_size}])
        if recorder:
            recorder.record('val', epoch, val_loss_stats)
            recorder.record('val', epoch, val_metric_stats)

       
        torch.save(gt_annotations, "gt_annotation.pkl")
        torch.save(det_results, "det_results.pkl")
        return val_metric_stats

    '''def val(self, epoch, data_loader, evaluator:Evaluator, recorder:Recorder, visualizer:visualizer=None, val_nms=False):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        val_metric_stats = {}
        pred_list = []
        label_list = []

        check_records = {}

        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(list(batch))
            layer_rects, edges, types,  img_tensor, labels, bboxes, node_indices, file_list = batch
            with torch.no_grad():
                output, loss, loss_stats = self.network(batch)
                #val_stats = evaluator.evaluate(output, batch[4])
                loss_stats = self.reduce_loss_stats(loss_stats)
                
                for k, v in loss_stats.items():
                    val_loss_stats.setdefault(k, 0)
                    val_loss_stats[k] += v
                
               
                #for k, v in val_stats.items():
                #    val_metric_stats.setdefault(k, 0)
                #    val_metric_stats[k] += v
            
                logits, local_params = output
                scores, pred = torch.max(F.softmax(logits,dim=1), 1)
            
                #print(layer_rects.shape, pred.shape, labels)
                pred_fraglayers = layer_rects[pred==1]
                pred_merging_groups = local_params[pred==1]
                scores = scores [pred==1]

                pred_bboxes = pred_merging_groups + pred_fraglayers
                bbox_results = nms_merge(pred_bboxes, scores, threshold=0.45)
                
                if val_nms:
                    #prev_pred = pred
                    pred = evaluator.correct_pred_with_nms(pred, bbox_results, layer_rects, types, threshold=0.45) 
                    #print(f"correct {torch.sum(pred!=prev_pred)}wrong preditions")

                pred_list.append(pred)
                label_list.append(labels) 

                if visualizer is not None:
                    visualizer.visualize_pred(pred_fraglayers, pred_merging_groups,batch[7][0])
                    visualizer.visualize_nms(bbox_results.cpu(),batch[7][0])
                    
                    fragmented_layers = layer_rects[labels==1]
                    merging_groups = bboxes[labels == 1 ]
                    visualizer.visualize_gt(fragmented_layers, merging_groups, batch[7][0])
                    #visualizer.visualize_nms(scores.cpu(), fragmented_layers.cpu(), merging_groups.cpu(),batch[6][0])
                    
        val_metric_stats = evaluator.evaluate(torch.cat(pred_list), torch.cat(label_list))

        loss_state = []
        metric_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        for k in val_metric_stats.keys():
            #val_metric_stats[k] /= data_size
            metric_state.append('{}: {:.4f}'.format(k, val_metric_stats[k]))
        print(loss_state,metric_state)

        if recorder:
            recorder.record('val', epoch, val_loss_stats)
            recorder.record('val', epoch, val_metric_stats)

        return val_metric_stats
    '''
    def check_with_human_in_loop(self, epoch, data_loader, evaluator:Evaluator, 
                                recorder:Recorder, visualizer:visualizer=None, val_nms=False):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {} 
        data_size = len(data_loader)
        val_metric_stats = {}
        pred_list = []
        label_list = []

        check_records = {}

        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(list(batch))
            layer_rects, edges, types, img_tensor, labels, bboxes, node_indices, file_list = batch
            #print(node_indices)
            rootDir, artboard_idx = os.path.split(file_list[0])
            artboard_idx = artboard_idx.split(".")[0]
            print("checking artboard:", artboard_idx,"...")
            with torch.no_grad():

                output, loss, loss_stats = self.network(batch)
                logits, centers, _, confidence = output
                scores, pred = torch.max(F.softmax(logits,dim=1), 1)
                
                correct_mask = torch.logical_and( pred == 1, pred != labels)
                correct_idx = torch.where(correct_mask)[0]
                print(correct_idx)
                fragmented_layers = layer_rects[labels == 1]
                merging_groups = bboxes[labels == 1 ]
                visualizer.visualize_gt(fragmented_layers, merging_groups + fragmented_layers, batch[7][0])
                    
                if correct_idx.shape[0] != 0:
                    correct_layer_rects = layer_rects[correct_idx, :]
                    visualizer.visualize_pred(correct_layer_rects, bboxes[correct_idx, :] + correct_layer_rects, batch[7][0])
                    correct_idx_list = []
                    for i in range(correct_idx.shape[0]):
                        visualizer.visualize_with_labels(correct_layer_rects[i : i+1,:], correct_idx[i : i + 1], batch[7][0])
                        correct_idx_confirm = input(f"accept the correct id {correct_idx[i]} or not [y/n]: ")
                        if correct_idx_confirm=='y':
                            correct_idx_list.append(correct_idx[i].item())
                    print("correction ids: ", correct_idx_list)   
                    correct_idx_confirm = input("accept the correct id_list? [y/n]")
                    
                    if correct_idx_confirm == 'y':
                        if  len(correct_idx_list) and correct_idx_list[0] != '':
                            correct_dataset(rootDir, artboard_idx, node_indices, correct_idx_list)
                    
                    visualizer.remove_files()
        return val_metric_stats
def make_trainer(network):
    return Trainer(network)