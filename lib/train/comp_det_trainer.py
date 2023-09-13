import json
import torch
from lib.utils.nms import contains, refine_merging_bbox
from lib.config import cfg as CFG
from lib.train.recorder import Recorder
import time
import datetime
import tqdm
from lib.evaluators.comp_det_eval import Evaluator
from lib.visualizers.comp_det_vis import comp_det_visualizer
import torch.nn.functional as F

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
        print("[MESSAGE] Component Detection Trainer init!")
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
            output, loss, loss_stats = self.network(batch)  #output: (logits, centers, bboxes)

            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            
            if cfg.local_rank > 0:
                continue
            
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            logits, _, = output #output: (logits, bboxes)

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
    
    def val(self, epoch, data_loader, evaluator:Evaluator, recorder:Recorder, visualizer = None, eval_ap = False):

        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        val_metric_stats = {}
        pred_list = []
        label_list = []

        gt_annotations = []
        det_results = []
        
        merge_eval_stats = {}
        not_valid_samples = 0
        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(list(batch))
            assets_img, layer_rect, edges, bbox, labels, node_indices, artboard_id = batch
            
            with torch.no_grad():
                output, loss, loss_stats = self.network(batch)
                #val_stats = evaluator.evaluate(output, batch[4])
                loss_stats = self.reduce_loss_stats(loss_stats)
                
                logits, pred_bboxes, = output #output: (logits, bboxes)

                _, label_pred = torch.max(F.softmax(logits,dim=1), 1)

                for k, v in loss_stats.items():
                    val_loss_stats.setdefault(k, 0)
                    val_loss_stats[k] += v
                
                '''if eval_ap:
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
                '''
                pred_list.append(label_pred)
                label_list.append(labels) 

                if visualizer is not None:
                    visualizer.visualize_overall(layer_rect, label_pred, labels, pred_bboxes, bbox, artboard_id[0])
                    
        val_metric_stats = evaluator.evaluate(torch.cat(pred_list), torch.cat(label_list))
        loss_state = []
        metric_state = []
        
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        
        for k in val_metric_stats.keys():
            #val_metric_stats[k] /= data_size
            metric_state.append('{}: {:.4f}'.format(k, val_metric_stats[k]))
        #print(loss_state, metric_state, [{"merging_acc": eval_merging_acc / data_size}])
        print(loss_state, metric_state)
        
        if recorder:
            recorder.record('val', epoch, val_loss_stats)
            recorder.record('val', epoch, val_metric_stats)

        '''
        if eval_ap:
            torch.save(gt_annotations, "gt_annotation.pkl")
            torch.save(det_results, "det_results.pkl")
        '''
        return val_metric_stats