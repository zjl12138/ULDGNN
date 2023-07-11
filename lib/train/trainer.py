from logging import root
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
from math import sqrt

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
        # self.anchor_box_wh = torch.tensor([[sqrt(16.0 * 8.0) * 2, sqrt(16.0 * 8.0)], [16.0, 16.0], [sqrt(16.0 * 8.0), 2 * sqrt(16.0 * 8.0)],
        #                                   [sqrt(128.0 * 64.0) * 2, sqrt(128.0 * 64.0)], [128.0, 128.0], [sqrt(128.0 * 64.0), 2 * sqrt(128.0 * 64.0)],
        #                                   [sqrt(128.0 * 256.0) * 2, sqrt(128.0 * 256.0)], [256.0, 256.0], [sqrt(128.0 * 256.0), 2 * sqrt(128.0 * 256.0)]], dtype = torch.float32).to(self.device) / 375.0


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
            CFG.network.alpha = CFG.network.alpha * (epoch * len(data_loader) + iteration) / (2 * len(data_loader))
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
    
    def process_layer_coord(self, orig_layer_rect, patch_size, offsets_in_artboard, artboard_size):
        orig_layer_rect[:, 2 : 4] -= orig_layer_rect[:, 0 : 2]
        orig_layer_rect *= patch_size
        orig_layer_rect[:, 0 : 2] += offsets_in_artboard
        orig_layer_rect /= artboard_size
        return orig_layer_rect
        
    def process_output_data_uldgnn(self, output, layer_rects_gt, labels, bboxes_gt, node_indices, file_list):
        # we need to transforme the coordinates to match original artboard size
        offsets_in_artboard = []
        patch_size = []
        artboard_size = []
        rootDir = '/media/sda1/ljz-workspace/dataset/ULDGNN_graph_dataset'
        meta_infos = []
        for img_path in file_list:
            file_path, artboard_name = os.path.split(img_path)
            artboard_name = artboard_name.split(".")[0]
            
            iter_idx = 0
            meta_info = []
            tmp_dict = json.load(open(os.path.join(rootDir, artboard_name, "meta.json")))
            patch_size_tmp, W, H = tmp_dict['patch_size'], tmp_dict['W'], tmp_dict['H']
            
            while True:
                iter_json_path = os.path.join(rootDir, artboard_name, f"{artboard_name}-{iter_idx}.json")
                if not os.path.exists(iter_json_path):
                    break
                tmp_offset = json.load(open(iter_json_path))['offset_in_artboard']
                meta_info.append([patch_size_tmp, W, H, tmp_offset[0], tmp_offset[1]])
                iter_idx += 1
            meta_infos.append(meta_info)
            
        for idx in node_indices:
            patch_size.append([meta_info[idx][0], meta_info[idx][0], meta_info[idx][0], meta_info[idx][0]])
            offsets_in_artboard.append(meta_info[idx][3 : 5])
            artboard_size.append([meta_info[idx][1], meta_info[idx][2], meta_info[idx][1], meta_info[idx][2]])
        
        patch_size = torch.tensor(patch_size, dtype = torch.float32, device = layer_rects_gt.device)
        offsets_in_artboard = torch.tensor(offsets_in_artboard, dtype = torch.float32, device = layer_rects_gt.device)
        artboard_size = torch.tensor(artboard_size, dtype = torch.float32, device = layer_rects_gt.device)
            
        logits, centers, local_params, confidence, _ = output #output: (logits, centers, bboxes)
        cls_scores, pred = torch.max(F.softmax(logits, dim = 1), 1)
        
        bboxes_gt += layer_rects_gt
        bboxes_gt[:, 2 : 4] += bboxes_gt[:, 0 : 2]
        
        bboxes_gt = self.process_layer_coord(bboxes_gt, patch_size, offsets_in_artboard, artboard_size)
        layer_rects_gt = self.process_layer_coord(layer_rects_gt, patch_size, offsets_in_artboard, artboard_size)

        centers *= patch_size[:, 0:2]
        centers += offsets_in_artboard
        fragmented_layers_gt = layer_rects_gt[labels == 1]
        fragmented_layers_pred = layer_rects_gt[pred == 1]
            
        '''
        merging_groups_pred = local_params[labels == 1] 
        scores = scores[labels == 1]
        '''
        local_params[:, 2 : 4] += local_params[:, 0 : 2]
        local_params = self.process_layer_coord(local_params, patch_size, offsets_in_artboard, artboard_size)
        merging_groups_pred = local_params[pred == 1] 
    
         
        merging_groups_pred_mask = contains(merging_groups_pred, layer_rects_gt[pred == 1])

        if confidence is not None:
            scores = cls_scores[pred == 1]
            merging_groups_pred_nms, merging_groups_confidence = nms_merge(merging_groups_pred, scores, threshold = 0.45)

        else:
            scores = confidence[pred == 1]
            #mask = torch.logical_and(scores >= 0.8, merging_groups_pred_mask) 
            #merging_groups_pred_nms, merging_groups_confidence = nms_merge(merging_groups_pred[mask], scores[mask], threshold=0.4)
            merging_groups_pred_nms, merging_groups_confidence = nms_merge(merging_groups_pred, scores, threshold=0.4)

        merging_groups_gt = bboxes_gt[labels == 1]
    
        return {
                'layer_rects': layer_rects_gt,
                'fragmented_layers_gt': fragmented_layers_gt,
                'fragmented_layers_pred': fragmented_layers_pred,
                'merging_groups_gt': merging_groups_gt,
                'merging_groups_pred': merging_groups_pred,
                'merging_groups_pred_nms': merging_groups_pred_nms,
                'label_pred': pred,
                'centers_pred': centers[pred == 1, :],
                'merging_groups_confidence': merging_groups_confidence,
                'bboxes': bboxes_gt
               }
        pass
    
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

        if confidence is not None:
            scores = cls_scores[pred == 1]
            merging_groups_pred_nms, merging_groups_confidence = nms_merge(merging_groups_pred, scores, threshold = 0.45)

        else:
            scores = confidence[pred == 1]
            #mask = torch.logical_and(scores >= 0.8, merging_groups_pred_mask) 
            #merging_groups_pred_nms, merging_groups_confidence = nms_merge(merging_groups_pred[mask], scores[mask], threshold=0.4)
            merging_groups_pred_nms, merging_groups_confidence = nms_merge(merging_groups_pred, scores, threshold=0.45)

        merging_groups_gt = bboxes_gt[labels == 1] + fragmented_layers_gt
    
        return {
                'fragmented_layers_gt': fragmented_layers_gt,
                'fragmented_layers_pred': fragmented_layers_pred,
                'merging_groups_gt': merging_groups_gt,
                'merging_groups_pred': merging_groups_pred,
                'merging_groups_pred_nms': merging_groups_pred_nms,
                'label_pred': pred,
                'centers_pred': centers[pred==1, :],
                'merging_groups_confidence': merging_groups_confidence,
                'layer_rects': layer_rects_gt,
                'bboxes': bboxes_gt + layer_rects_gt
               }
    
    def val_train(self, epoch, data_loader, evaluator:Evaluator, recorder:Recorder, visualizer:visualizer=None, val_nms=False, eval_merge = False, eval_ap = False):

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
        
        merge_eval_stats = {}
        not_valid_samples = 0
        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(list(batch))
            layer_rects_, edges, types,  img_tensor, labels, bboxes, layer_rects, node_indices, file_list = batch
            if torch.sum(labels) == 0:
                not_valid_samples += 1
                continue
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
                # fetch_data = self.process_output_data(output, layer_rects, labels, bboxes)
                logits, centers, local_params, confidence, _ = output #output: (logits, centers, bboxes)
                cls_scores, label_pred = torch.max(F.softmax(logits, dim = 1), 1)
                pred_list.append(label_pred)
                label_list.append(labels) 

        val_metric_stats = evaluator.evaluate(torch.cat(pred_list), torch.cat(label_list))
        data_size -= not_valid_samples
        loss_state = []
        metric_state = []
        merge_eval_state = []
        print(data_size)
        for k in merge_eval_stats.keys():
            merge_eval_stats[k] /= data_size
            merge_eval_state.append('{}: {}'.format(k, merge_eval_stats[k]))
            
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        
        for k in val_metric_stats.keys():
            #val_metric_stats[k] /= data_size
            metric_state.append('{}: {:.4f}'.format(k, val_metric_stats[k]))
        #print(loss_state, metric_state, [{"merging_acc": eval_merging_acc / data_size}])
        print(loss_state, metric_state, merge_eval_state)
        
        if recorder:
            recorder.record('val', epoch, val_loss_stats)
            recorder.record('val', epoch, val_metric_stats)

        return val_metric_stats
    
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
        
        merge_eval_stats = {}
        not_valid_samples = 0
        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(list(batch))
            layer_rects_, edges, types,  img_tensor, labels, bboxes, layer_rects, node_indices, file_list = batch
            if torch.sum(labels) == 0:
                not_valid_samples += 1
                continue
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
                # fetch_data = self.process_output_data(output, layer_rects, labels, bboxes)
                fetch_data = self.process_output_data_uldgnn(output, layer_rects, labels, bboxes, node_indices, file_list)
                
                fragmented_layers_gt = fetch_data['fragmented_layers_gt']
                fragmented_layers_pred = fetch_data['fragmented_layers_pred']
                merging_groups_gt = fetch_data['merging_groups_gt']
                merging_groups_pred = fetch_data['merging_groups_pred']
                merging_groups_pred_nms = fetch_data['merging_groups_pred_nms']
                merging_groups_confidence = fetch_data['merging_groups_confidence']
                layer_rects = fetch_data['layer_rects']
                label_pred = fetch_data['label_pred']
                bboxes = fetch_data['bboxes']
                merging_groups_gt_nms, _ = nms_merge(merging_groups_gt, torch.ones(merging_groups_gt.shape[0], device = merging_groups_gt.device))
                
                if val_nms:
                    #prev_pred = pred
                    label_pred = evaluator.correct_pred_with_nms(label_pred, merging_groups_pred_nms, layer_rects, types, threshold=0.45) 
                    fetch_data['label_pred'] = label_pred
                    #print(f"correct {torch.sum(pred!=prev_pred)}wrong preditions")
                
                if eval_merge:
                    tmp_stats = evaluator.evaluate_merging(merging_groups_pred_nms, label_pred, layer_rects, bboxes, labels)
                    for k, v in tmp_stats.items():
                        merge_eval_stats.setdefault(k, 0)
                        merge_eval_stats[k] += v
                    #eval_merging_acc = eval_merging_acc + evaluator.evaluate_merging(merging_groups_pred_nms, label_pred, layer_rects, bboxes + layer_rects, labels)

                if eval_ap:
                    if len(merging_groups_pred_nms) and len(merging_groups_gt_nms):
                        
                        file_path, artboard_name = os.path.split(file_list[0])
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
                    visualizer.visualize_pred(fragmented_layers_pred, merging_groups_pred, file_list[0])
                    #visualizer.visualize_nms(merging_groups_pred_nms, file_list[0])
                    visualizer.visualize_nms_with_labels(merging_groups_pred_nms, merging_groups_confidence, file_list[0])
                    
                    if len(merging_groups_pred_nms) != 0:
                        merging_groups_pred_nms, merging_groups_confidence = refine_merging_bbox(torch.vstack(merging_groups_pred_nms), layer_rects, label_pred, merging_groups_confidence, types)
                        visualizer.visualize_nms_with_labels(merging_groups_pred_nms, merging_groups_confidence, file_list[0], mode = "bbox_refine")
                        #det_results.append([np.vstack([scale_to_img(t.cpu().numpy(), H, W) for t in merging_groups_pred_nms])])
                        
                    visualizer.visualize_gt(fragmented_layers_gt, merging_groups_gt, file_list[0])
                    visualizer.visualize_offset_of_centers(centers_pred, fragmented_layers_pred, file_list[0])
                    '''
                    visualizer.visualize_overall(fetch_data, layer_rects, types, file_list[0])
        val_metric_stats = evaluator.evaluate(torch.cat(pred_list), torch.cat(label_list))
        data_size -= not_valid_samples
        loss_state = []
        metric_state = []
        merge_eval_state = []
        print(data_size)
        for k in merge_eval_stats.keys():
            merge_eval_stats[k] /= data_size
            merge_eval_state.append('{}: {}'.format(k, merge_eval_stats[k]))
            
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        
        for k in val_metric_stats.keys():
            #val_metric_stats[k] /= data_size
            metric_state.append('{}: {:.4f}'.format(k, val_metric_stats[k]))
        #print(loss_state, metric_state, [{"merging_acc": eval_merging_acc / data_size}])
        print(loss_state, metric_state, merge_eval_state)
        
        if recorder:
            recorder.record('val', epoch, val_loss_stats)
            recorder.record('val', epoch, val_metric_stats)

        if eval_ap:
            torch.save(gt_annotations, "gt_annotation.pkl")
            torch.save(det_results, "det_results.pkl")
        return val_metric_stats
    
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
            layer_rects_, edges, types, img_tensor, labels, bboxes, layer_rects, node_indices, file_list = batch
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