import torch
import torch.nn as nn
import importlib
import timm
import torchvision

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

def GetPosEmbedder(multires, input_dim = 4, include_input = True):
    embed_kwargs = {
        'include_input':include_input,
        'input_dims':input_dim,
        'max_freq_log2':multires - 1,
        'num_freqs': multires,
        'log_sampling':True,
        'periodic_fns':[torch.sin, torch.cos]
    }    
    embed_obj = PosEmbedder_impl(**embed_kwargs)
    embed = lambda x, eo=embed_obj: eo.embed(x)
    return embed, embed_obj.out_dim

class PosEmbedder(nn.Module):
    def __init__(self, cfg):
        super(PosEmbedder, self).__init__()
        self.embed, self.in_dim = GetPosEmbedder(cfg.multires)
        self.fc_layer = nn.Linear(self.in_dim, cfg.out_dim)
        self.pos_enc = None

    def forward(self, x):
        x = self.embed(x)
        self.pos_enc = x
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
        '''
        for k, p in self.feature_extractor.named_parameters():
            if 'conv' in k or 'bn' in k or "downsample" in k:
                p.requires_grad = False
        '''
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

class ImageSeqEmbedder(nn.Module):
    def __init__(self, cfg):
        super(ImageSeqEmbedder, self).__init__()
        self.vit = timm.create_model(cfg.vit_name, pretrained = True, num_classes = 0)
        self.data_config = timm.data.resolve_model_data_config(self.vit)
        self.transforms = timm.data.create_transform(**self.data_config, is_training=True)
        for k, p in self.vit.named_parameters():
            p.requires_grad = False               # fix vision_transformer

    def forward(self, x):
        '''
        x: [batch, 3, H, W]
        return: [batch, seq_len, dim]
        '''
        x = self.vit.forward_features(x)
        return x[:, 1:, :]

class ImgFeatRoiExtractor(nn.Module):
    def __init__(self, cfg):
        super(ImgFeatRoiExtractor, self).__init__()
        self.roi_out_size = cfg.roi_out_size 
        self.roi_feat_way = cfg.roi_feat_way
        feature_extractor = torchvision.models.resnet50(pretrained = True)
        
        '''for k, p in self.feature_extractor.named_parameters():
            p.requires_grad = False
        '''
        self.pool = feature_extractor.maxpool
        self.layer0 = nn.Sequential(feature_extractor.conv1, 
                                    feature_extractor.bn1, 
                                    feature_extractor.relu)
        self.layer1 = feature_extractor.layer1
        for k, p in self.layer1.named_parameters():
            p.requires_grad = False
        self.layer2 = feature_extractor.layer2
        for k, p in self.layer2.named_parameters():
            p.requires_grad = False
        self.layer3 = feature_extractor.layer3
        for k, p in self.layer3.named_parameters():
            p.requires_grad = False
        self.layer4 = feature_extractor.layer4
       
        self.roi_out_dim = cfg.out_dim   
        if self.roi_feat_way != 'FPN':
            self.final_conv = nn.Conv2d(2048, self.roi_out_dim, kernel_size = 1, stride = 1, padding = 0)
        else:
            print("using FPN!")
            self.fpn = torchvision.ops.FeaturePyramidNetwork([256, 512, 1024, 2048], self.roi_out_dim)
            for k, p in self.fpn.layer_blocks[1].named_parameters():
                p.requires_grad = False
            for k, p in self.fpn.layer_blocks[2].named_parameters():
                p.requires_grad = False
            for k, p in self.fpn.layer_blocks[3].named_parameters():
                p.requires_grad = False
        self.final_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def get_feature_map(self, img_tensors):
        if self.roi_feat_way == 'FPN':
            y = self.layer0(img_tensors)
            y_1 = self.layer1(self.pool(y))
            y_2 = self.layer2(y_1)
            y_3 = self.layer3(y_2)
            y_4 = self.layer4(y_3)
            out = self.fpn({'feat_0':y_1, 'feat_1':y_2, 'feat_2':y_3, 'feat_3':y_4})
            return out['feat_0']
        else:
            y = self.layer0(img_tensors)
            y = self.layer1(self.pool(y))
            y = self.layer2(y)
            y = self.layer3(y)
            y = self.layer4(y)
            return y

    def forward(self, img_tensors, boxes):
        feature_maps = self.get_feature_map(img_tensors)
        roi_features = torchvision.ops.roi_align(feature_maps, boxes, output_size = (self.roi_out_size, self.roi_out_size))
        if self.roi_feat_way != 'FPN':
            roi_features = self.final_conv(roi_features)
        roi_features = self.final_pooling(roi_features)
        roi_features = roi_features.reshape(-1, self.roi_out_dim)
        return roi_features

        
