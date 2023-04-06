import torch
import torch.nn as nn
import importlib

class PosEmbedder_impl:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2**0., 2.**max_freq, steps=N_freqs)
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn = p_fn,freq=freq: p_fn(x*freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns],-1)

def GetPosEmbedder(multires, input_dim = 4):
    embed_kwargs = {
        'include_input':True,
        'input_dims':input_dim,
        'max_freq_log2':multires-1,
        'num_freqs':multires,
        'log_sampling':True,
        'periodic_fns':[torch.sin, torch.cos]
    }    
    embed_obj = PosEmbedder_impl(**embed_kwargs)
    embed = lambda x, eo=embed_obj: eo.embed(x)
    return embed, embed_obj.out_dim

class PosEmbedder(nn.Module):
    def __init__(self, cfg):
        super(PosEmbedder, self).__init__()
        self.embed, in_dim = GetPosEmbedder(cfg.multires)
        self.fc_layer = nn.Linear(in_dim, cfg.out_dim)
    def forward(self, x):
        x = self.embed(x)
        #print("embedding dim:",x.shape)
        return self.fc_layer(x)
    
class ImageEmbedder(nn.Module):
    def __init__(self, cfg):
        '''
        cfg is ImageEmbedder cfgNode
        '''
        super(ImageEmbedder, self).__init__()
        self.out_dim = cfg.out_dim    
        self.feature_extractor = getattr(importlib.import_module('torchvision.models'),cfg.name)(pretrained=True)
        for k, p in self.feature_extractor.named_parameters():
            if 'conv' in k or 'bn' in k or "downsample" in k:
                p.requires_grad = False
        num_channels = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_channels, self.out_dim)
    
    def forward(self, imgs):
        return self.feature_extractor(imgs)

class TypeEmbedder(nn.Module):
    def __init__(self, cfg):
        super(TypeEmbedder, self).__init__()
        self.out_dim = cfg.out_dim 
        self.embedding = nn.Embedding(cfg.class_num, self.out_dim)
    def forward(self, input):
        return self.embedding(input)




        
