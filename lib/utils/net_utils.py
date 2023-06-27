import torch
import os

def save_model(net, optim, scheduler, recorder, model_dir, epoch, last=False):
    model = {
        'net': net.state_dict(),
        # 'optim': optim.state_dict(),
        # 'scheduler': scheduler.state_dict(),
        # 'recorder': recorder.state_dict(),
        'epoch': epoch
    }
    if last:
        print("saving model in epoch ", epoch)
        torch.save(model, os.path.join(model_dir, 'latest.pth'))
    else:
        print("saving model in epoch ", epoch)
        torch.save(model, os.path.join(model_dir, '{}.pth'.format(epoch)))

    # remove previous pretrained model if the number of models is too big
    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) <= 20:
        return
    os.system('rm {}'.format(
        os.path.join(model_dir, '{}.pth'.format(min(pths)))))

def load_network(net, model_dir, map_location = 'cuda:0', resume=True, epoch=-1, strict=True):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        print('pretrained model does not exist', 'red')
        return 0

    if os.path.isdir(model_dir):
        pths = [
            int(pth.split('.')[0]) for pth in os.listdir(model_dir)
            if pth != 'latest.pth'
        ]
        if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
            return 0
        if epoch == -1:
            if 'latest.pth' in os.listdir(model_dir):
                pth = 'latest'
            else:
                pth = max(pths)
        else:
            pth = epoch
        model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    else:
        model_path = model_dir

    print('load model: {}'.format(model_path))
    pretrained_model = torch.load(model_path, map_location = map_location)
    net.load_state_dict(pretrained_model['net'], strict=strict)
    return pretrained_model['epoch'] + 1

def load_model(net,
               optim,
               scheduler,
               recorder,
               model_dir,
               resume=True,
               epoch=-1):
    if not resume:
        os.system('rm -rf {}'.format(model_dir))
        os.makedirs(model_dir)
    if not os.path.exists(model_dir):
        return 0

    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
        return 0
    if epoch == -1:
        if 'latest.pth' in os.listdir(model_dir):
            pth = 'latest'
        else:
            pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir,
                                               '{}.pth'.format(pth))))
    pretrained_model = torch.load(
        os.path.join(model_dir, '{}.pth'.format(pth)), 'cpu')
    net.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1

def load_partial_network(net, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        print('pretrained model does not exist', 'red')
        return 0

    if os.path.isdir(model_dir):
        pths = [
            int(pth.split('.')[0]) for pth in os.listdir(model_dir)
            if pth != 'latest.pth'
        ]
        if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
            return 0
        if epoch == -1:
            if 'latest.pth' in os.listdir(model_dir):
                pth = 'latest'
            else:
                pth = max(pths)
        else:
            pth = epoch
        model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    else:
        model_path = model_dir

    print('load partial model: {}'.format(model_path))
    save_dict = torch.load(model_path)
    pretrained_model = save_dict['net']
    model_dict = net.state_dict()
    keys = []
    for k, v in pretrained_model.items():
        keys.append(k)
    i = 0
    for k, v in model_dict.items():
        if k in pretrained_model.keys() and v.size() == pretrained_model[k].size():
            model_dict[k] = pretrained_model[k]
        else:
            print(f"not found {k} or {k}'s size not matched")
    net.load_state_dict(model_dict)
    return save_dict['epoch'] + 1

def average_by_weight(attn_weights, gnn_out):
    '''
    attn_weights: 
    '''
    return attn_weights @ gnn_out