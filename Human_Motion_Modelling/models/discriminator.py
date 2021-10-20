import sys
import os

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

##############################################################################
# Network helper functions
##############################################################################    

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(True)
    if activation == "gelu":
        return nn.GELU()
    if activation == "selu":
        return nn.SELU(True)
    if activation == "leaky_relu":
        return nn.LeakyReLU(0.2)
    raise RuntimeError(F"activation should be relu/gelu/selu/leaky_relu, not {activation}.")

def _get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'batch_sync':
        norm_layer = BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

##############################################################################
# Discriminator Class
############################################################################# 

class Discriminator_2D(nn.Module):
    def __init__(self, cfg):
        super(Discriminator_2D, self).__init__()

        model = []

        self.channels = cfg.channels
        self.acti = _get_activation_fn(cfg.acti)
        self.norm_layer = _get_norm_layer(cfg.norm)
        self.init_type = cfg.init
        self.use_patch_gan = cfg.use_patch_gan
        self.use_sigmoid = cfg.use_sigmoid

        if type(self.norm_layer) == functools.partial:
            use_bias = self.norm_layer.func == nn.InstanceNorm2d 
        else:
            use_bias = self.norm_layer == nn.InstanceNorm2d 
            
        model.append(nn.ReflectionPad2d(1))
        model.append(nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, padding=0, bias=use_bias))
        if self.norm_layer:
            if cfg.norm == 'group':
                model.append(self.norm_layer(self.channels[1]//8, self.channels[1]))
            else:
                model.append(self.norm_layer(self.channels[1]))
        model.append(self.acti)


        nr_layer = len(self.channels) - 1

        for i in range(1, nr_layer):
            model.append(nn.Conv2d(self.channels[i], self.channels[i+1],
                                   kernel_size=3, padding=1, bias=use_bias))
            if self.norm_layer:
                if cfg.norm == 'group':
                    model.append(self.norm_layer(self.channels[i+1]//8, self.channels[i+1]))
                else:
                    model.append(self.norm_layer(self.channels[i+1]))
            model.append(self.acti)
            model.append(nn.Conv2d(self.channels[i+1], self.channels[i+1],
                                   kernel_size=3, padding=1, bias=use_bias))
            if self.norm_layer:
                if cfg.norm == 'group':
                    model.append(self.norm_layer(self.channels[i+1]//8, self.channels[i+1]))
                else:
                    model.append(self.norm_layer(self.channels[i+1]))
            model.append(self.acti)
            model.append(nn.MaxPool2d(3, stride=2, padding=[1, 1]))

        self.model = nn.Sequential(*model)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.conv = nn.Conv2d(self.channels[-1], 1, kernel_size=1, stride=1, padding=0, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        init_weights(self.model, init_type=self.init_type)

    def forward(self, x):
        feature = self.model(x)
        if not self.use_patch_gan:
            feature = self.global_pool(feature)
        feature = self.conv(feature)
        if self.use_sigmoid:
            feature = torch.sigmoid(feature)
        return feature

    def get_feature(self, x):
        return self.model(x)