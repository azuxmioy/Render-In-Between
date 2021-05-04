# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Conv2dBlock, Res2dBlock, HyperConv2dBlock

from utils.utils import weights_init


class BaseNetwork(nn.Module):
    r"""vid2vid generator."""

    def __init__(self):
        super(BaseNetwork, self).__init__()

    def get_num_filters(self, num_downsamples):
        r"""Get the number of filters at current layer.

        Args:
            num_downsamples (int) : How many downsamples at current layer.
        Returns:
            output (int) : Number of filters.
        """
        return min(self.max_num_filters,
                   self.num_filters * (2 ** num_downsamples))


class Generator(BaseNetwork):
    r"""vid2vid generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg):
        super().__init__()
        self.gen_cfg = gen_cfg
        self.num_frames_G = gen_cfg.num_frames_G
        # Number of residual blocks in generator.
        self.num_layers = num_layers = getattr(gen_cfg, 'num_layers', 7)
        # Number of downsamplings for previous frame.
        self.num_downsamples_img = getattr(gen_cfg, 'num_downsamples_img', 4)
        # Number of filters in the first layer.
        self.num_filters = num_filters = getattr(gen_cfg, 'num_filters', 32)
        self.max_num_filters = getattr(gen_cfg, 'max_num_filters', 1024)
        self.kernel_size = kernel_size = getattr(gen_cfg, 'kernel_size', 3)
        padding = kernel_size // 2


        # Input data params.
        self.num_input_channels = gen_cfg.input_label_nc
        self.num_img_channels = gen_cfg.input_image_nc

        # Label / image embedding network.
        self.emb_cfg = emb_cfg = getattr(gen_cfg, 'embed', None)
        self.use_embed = getattr(emb_cfg, 'use_embed', 'True')
        self.num_downsamples_embed = getattr(emb_cfg, 'num_downsamples', 5)
        if self.use_embed:
            self.ref_embedding = LabelEmbedder(emb_cfg, self.num_img_channels*2)
            self.label_embedding = LabelEmbedder(emb_cfg, self.num_input_channels)

        # Flow network.
        self.mask_cfg = gen_cfg.mask

        #self.img_prev_embedding = LabelEmbedder(emb_cfg, self.num_img_channels + 1)

        # At beginning of training, only train an image generator.
        self.temporal_initialized = False
        # Whether to output hallucinated frame (when training temporal network)
        # for additional loss.
        self.generate_raw_output = False

        # Image generation network.
        weight_norm_type = getattr(gen_cfg, 'weight_norm_type', 'spectral')
        activation_norm_type = gen_cfg.activation_norm_type
        activation_norm_params = gen_cfg.activation_norm_params
        if self.use_embed and \
                not hasattr(activation_norm_params, 'num_filters'):
            activation_norm_params.num_filters = 0
        nonlinearity = 'leakyrelu'

        self.base_res_block = base_res_block = partial(
            Res2dBlock, kernel_size=kernel_size, padding=padding,
            weight_norm_type=weight_norm_type,
            activation_norm_type=activation_norm_type,
            activation_norm_params=activation_norm_params,
            nonlinearity=nonlinearity, order='NACNAC')


        self.base_conv_block = base_conv_block = partial(
                                  Conv2dBlock, kernel_size=kernel_size,
                                  padding=padding,
                                  weight_norm_type=weight_norm_type,
                                  activation_norm_type='instance',
                                  nonlinearity='leakyrelu')

        # Upsampling residual blocks.
        for i in range(self.num_downsamples_img, -1, -1):
            activation_norm_params.cond_dims = self.get_cond_dims(i)
            activation_norm_params.partial = self.get_partial(
                i) if hasattr(self, 'get_partial') else False

            layer = base_res_block(self.get_num_filters(i + 1),
                                   self.get_num_filters(i))
            setattr(self, 'up_%d' % i, layer)

        # Final conv layer.
        self.conv_img = Conv2dBlock(num_filters, self.num_img_channels,
                                    kernel_size, padding=padding,
                                    nonlinearity=nonlinearity, order='AC')

        self.conv_mask = Conv2dBlock(num_filters, 1,
                                    kernel_size, padding=padding,
                                    nonlinearity=nonlinearity, order='AC')

        num_filters = min(self.max_num_filters,
                          num_filters * (2 ** (self.num_layers + 1)))


        # Misc.
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.upsample = partial(F.interpolate, scale_factor=2)
        self.init_temporal_network()


    def init_temporal_network(self, cfg_init=None):
        r"""When starting training multiple frames, initialize the
        downsampling network and flow network.

        Args:
            cfg_init (dict) : Weight initialization config.
        """
        # Number of image downsamplings for the previous frame.
        num_downsamples_img = self.num_downsamples_img
        # Number of residual blocks for the previous frame.
        self.num_res_blocks = int(
            np.ceil((self.num_layers - num_downsamples_img) / 2.0) * 2)

        # First conv layer.
        self.down_first = \
            Conv2dBlock(self.num_input_channels,
                        self.num_filters, self.kernel_size,
                        padding=self.kernel_size // 2)
        if cfg_init is not None:
            self.down_first.apply(weights_init(cfg_init.type, cfg_init.gain))

        # Downsampling residual blocks.
        activation_norm_params = self.gen_cfg.activation_norm_params
        for i in range(num_downsamples_img + 1):
            activation_norm_params.cond_dims = self.get_cond_dims(i)
            layer = self.base_res_block(self.get_num_filters(i),
                                        self.get_num_filters(i + 1))
            if cfg_init is not None:
                layer.apply(weights_init(cfg_init.type, cfg_init.gain))
            setattr(self, 'down_%d' % i, layer)

        # Additional residual blocks.
        res_ch = self.get_num_filters(num_downsamples_img + 1)
        activation_norm_params.cond_dims = \
            self.get_cond_dims(num_downsamples_img + 1)
        for i in range(self.num_res_blocks):
            layer = self.base_res_block(res_ch, res_ch)
            if cfg_init is not None:
                layer.apply(weights_init(cfg_init.type, cfg_init.gain))
            setattr(self, 'res_%d' % i, layer)

        # Mask network.
        self.temporal_initialized = True
        self.flow_network_temp = MaskGenerator(self.gen_cfg)
        if cfg_init is not None:
            self.flow_network_temp.apply(weights_init(cfg_init.type,
                                                      cfg_init.gain))


    def forward(self, label, label_prev, label_ref, img_fake, img_prev, img_ref):
        r"""vid2vid generator forward.

        Args:
           data (dict) : Dictionary of input data.
        Returns:
           output (dict) : Dictionary of output data.
        """

        bs, _, h, w = label.size()


        # Get SPADE conditional maps by embedding current label input.
        cond_maps_now = self.get_cond_maps(torch.cat([img_fake, img_prev], dim=1), self.ref_embedding)
        #cond_maps_now = self.get_cond_maps(torch.cat([img_ref, img_fake, img_prev], dim=1), self.ref_embedding)
        # Get label embedding for the previous frame.
        #cond_maps_prev = self.get_cond_maps(torch.cat([label_prev, img_prev], dim=1), self.ref_embedding)
        #cond_maps_ref= self.get_cond_maps(torch.cat([label_ref, img_ref], dim=1), self.ref_embedding)
        cond_maps_prev = cond_maps_now
        
        # Not the first frame, will encode the previous frame and feed to
        # the generator.
        # x_img = self.down_first(img_prev)
        # x_img_prev = self.down_first(torch.cat([img_fake, img_prev], dim=1))
        # x_img_ref = self.down_first(torch.cat([img_fake, img_ref], dim=1))
        x_img_prev = self.down_first(label)
        x_img_ref = self.down_first(label)
        # Downsampling layers.
        for i in range(self.num_downsamples_img + 1):
            j = min(self.num_downsamples_embed, i)
            x_img_prev = getattr(self, 'down_' + str(i))(x_img_prev, *cond_maps_prev[j])

            if i != self.num_downsamples_img:
                x_img_prev = self.downsample(x_img_prev)

        # Resnet blocks.
        j = min(self.num_downsamples_embed, self.num_downsamples_img + 1)
        for i in range(self.num_res_blocks):
            cond_maps = cond_maps_prev[j] if i < self.num_res_blocks // 2 \
                else cond_maps_now[j]
            x_img_prev = getattr(self, 'res_' + str(i))(x_img_prev, *cond_maps)

        x_img = x_img_prev

        # Main image generation branch.
        for i in range(self.num_downsamples_img, -1, -1):
            # Get SPADE conditional inputs.
            j = min(i, self.num_downsamples_embed)
            cond_maps = cond_maps_now[j]
            x_img = self.one_up_conv_layer(x_img, cond_maps, i)


        # Final conv layer.
        img_final = torch.tanh(self.conv_img(x_img))
        #mask = torch.sigmoid(self.conv_mask(x_img_mask))

        #Estimate alpha mask.
        mask = self.flow_network_temp(label, torch.cat([img_prev, img_fake, img_final], dim=1))

        return img_final, mask

    def one_up_conv_layer(self, x, encoded_label, i):
        r"""One residual block layer in the main branch.

        Args:
           x (4D tensor) : Current feature map.
           encoded_label (list of tensors) : Encoded input label maps.
           i (int) : Layer index.
        Returns:
           x (4D tensor) : Output feature map.
        """
        layer = getattr(self, 'up_' + str(i))
        x = layer(x, *encoded_label)
        if i != 0:
            x = self.upsample(x)
        return x


    def one_up_mask_layer(self, x, encoded_label, i):
        r"""One residual block layer in the main branch.

        Args:
           x (4D tensor) : Current feature map.
           encoded_label (list of tensors) : Encoded input label maps.
           i (int) : Layer index.
        Returns:
           x (4D tensor) : Output feature map.
        """
        layer = getattr(self, 'mask_' + str(i))
        x = layer(x)
        if i != 0:
            x = self.upsample(x)
        return x

    def get_cond_dims(self, num_downs=0):
        r"""Get the dimensions of conditional inputs.

        Args:
           num_downs (int) : How many downsamples at current layer.
        Returns:
           ch (list) : List of dimensions.
        """
        if not self.use_embed:
            ch = [self.num_input_channels]
        else:
            num_filters = getattr(self.emb_cfg, 'num_filters', 32)
            num_downs = min(num_downs, self.num_downsamples_embed)
            ch = [min(self.max_num_filters, num_filters * (2 ** num_downs))]
            #if (num_downs < self.num_multi_spade_layers):
            #    ch = ch * 2
        return ch

    def get_cond_maps(self, label, embedder):
        r"""Get the conditional inputs.

        Args:
           label (4D tensor) : Input label tensor.
           embedder (obj) : Embedding network.
        Returns:
           cond_maps (list) : List of conditional inputs.
        """
        if not self.use_embed:
            return [label] * (self.num_layers + 1)
        embedded_label = embedder(label)
        cond_maps = [embedded_label]
        cond_maps = [[m[i] for m in cond_maps] for i in
                     range(len(cond_maps[0]))]
        return cond_maps

###########################################################################

class LabelEmbedder(nn.Module):
    r"""Embed the input label map to get embedded features.

    Args:
        emb_cfg (obj): Embed network configuration.
        num_input_channels (int): Number of input channels.
        num_hyper_layers (int): Number of hyper layers.
    """

    def __init__(self, emb_cfg, num_input_channels, num_hyper_layers=0):
        super().__init__()
        num_filters = getattr(emb_cfg, 'num_filters', 32)
        max_num_filters = getattr(emb_cfg, 'max_num_filters', 1024)
        self.arch = getattr(emb_cfg, 'arch', 'encoderdecoder')
        self.num_downsamples = num_downsamples = \
            getattr(emb_cfg, 'num_downsamples', 5)
        kernel_size = getattr(emb_cfg, 'kernel_size', 3)
        weight_norm_type = getattr(emb_cfg, 'weight_norm_type', 'spectral')
        activation_norm_type = getattr(emb_cfg, 'activation_norm_type', 'none')

        self.unet = 'unet' in self.arch
        self.has_decoder = 'decoder' in self.arch or self.unet
        self.num_hyper_layers = num_hyper_layers \
            if num_hyper_layers != -1 else num_downsamples

        base_conv_block = partial(HyperConv2dBlock, kernel_size=kernel_size,
                                  padding=(kernel_size // 2),
                                  weight_norm_type=weight_norm_type,
                                  activation_norm_type=activation_norm_type,
                                  nonlinearity='leakyrelu')

        ch = [min(max_num_filters, num_filters * (2 ** i))
              for i in range(num_downsamples + 1)]

        self.conv_first = base_conv_block(num_input_channels, num_filters,
                                          activation_norm_type='none')

        # Downsample.
        for i in range(num_downsamples):
            is_hyper_conv = (i < num_hyper_layers) and not self.has_decoder
            setattr(self, 'down_%d' % i,
                    base_conv_block(ch[i], ch[i + 1], stride=2,
                                    is_hyper_conv=is_hyper_conv))

        # Upsample.
        if self.has_decoder:
            self.upsample = nn.Upsample(scale_factor=2)
            for i in reversed(range(num_downsamples)):
                ch_i = ch[i + 1] * (
                    2 if self.unet and i != num_downsamples - 1 else 1)
                setattr(self, 'up_%d' % i,
                        base_conv_block(ch_i, ch[i],
                                        is_hyper_conv=(i < num_hyper_layers)))

    def forward(self, input, weights=None):
        r"""Embedding network forward.

        Args:
            input (NxCxHxW tensor): Network input.
            weights (list of tensors): Conv weights if using hyper network.
        Returns:
            output (list of tensors): Network outputs at different layers.
        """
        if input is None:
            return None
        output = [self.conv_first(input)]

        for i in range(self.num_downsamples):
            layer = getattr(self, 'down_%d' % i)
            # For hyper networks, the hyper layers are at the last few layers
            # of decoder (if the network has a decoder). Otherwise, the hyper
            # layers will be at the first few layers of the network.
            if i >= self.num_hyper_layers or self.has_decoder:
                conv = layer(output[-1])
            else:
                conv = layer(output[-1], conv_weights=weights[i])
            # We will use outputs from different layers as input to different
            # SPADE layers in the main branch.
            output.append(conv)

        if not self.has_decoder:
            return output

        # If the network has a decoder, will use outputs from the decoder
        # layers instead of the encoding layers.
        if not self.unet:
            output = [output[-1]]

        for i in reversed(range(self.num_downsamples)):
            input_i = output[-1]
            if self.unet and i != self.num_downsamples - 1:
                input_i = torch.cat([input_i, output[i + 1]], dim=1)

            input_i = self.upsample(input_i)
            layer = getattr(self, 'up_%d' % i)
            # The last few layers will be hyper layers if necessary.
            if i >= self.num_hyper_layers:
                conv = layer(input_i)
            else:
                conv = layer(input_i, conv_weights=weights[i])
            output.append(conv)

        if self.unet:
            output = output[self.num_downsamples:]
        return output[::-1]


###########################################################################

class MaskGenerator(BaseNetwork):
    r"""Flow generator constructor.

    Args:
       flow_cfg (obj): Flow definition part of the yaml config file.
       data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg):
        super().__init__()

        num_input_channels = gen_cfg.input_label_nc
        num_prev_img_channels = gen_cfg.input_image_nc
        num_frames = gen_cfg.num_frames_G  # Num. of input frames.
        mask_cfg = gen_cfg.mask

        self.num_filters = num_filters = getattr(mask_cfg, 'num_filters', 32)
        self.max_num_filters = getattr(mask_cfg, 'max_num_filters', 1024)
        num_downsamples = getattr(mask_cfg, 'num_downsamples', 5)
        kernel_size = getattr(mask_cfg, 'kernel_size', 3)
        padding = kernel_size // 2
        self.num_res_blocks = getattr(mask_cfg, 'num_res_blocks', 6)

        activation_norm_type = getattr(mask_cfg, 'activation_norm_type',
                                       'sync_batch')
        weight_norm_type = getattr(mask_cfg, 'weight_norm_type', 'spectral')

        base_conv_block = partial(Conv2dBlock, kernel_size=kernel_size,
                                  padding=padding,
                                  weight_norm_type=weight_norm_type,
                                  activation_norm_type=activation_norm_type,
                                  nonlinearity='leakyrelu')

        # Will downsample the labels and prev frames separately, then combine.
        down_lbl = [base_conv_block(num_input_channels,
                                    num_filters)]
        down_img = [base_conv_block(num_prev_img_channels*3,
                                    num_filters)]
        for i in range(num_downsamples):
            down_lbl += [base_conv_block(self.get_num_filters(i),
                                         self.get_num_filters(i + 1),
                                         stride=2)]
            down_img += [base_conv_block(self.get_num_filters(i),
                                         self.get_num_filters(i + 1),
                                         stride=2)]

        # Resnet blocks.
        res_mask= []
        ch = self.get_num_filters(num_downsamples)
        for i in range(self.num_res_blocks):
            if i == 0:
                res_mask += [
                    Res2dBlock(ch*2, ch, kernel_size, padding=padding,
                           weight_norm_type=weight_norm_type,
                           activation_norm_type=activation_norm_type,
                           order='CNACN')]
            else:
                res_mask += [
                    Res2dBlock(ch, ch, kernel_size, padding=padding,
                           weight_norm_type=weight_norm_type,
                           activation_norm_type=activation_norm_type,
                           order='CNACN')]
        # Upsample.
        up_mask = []
        for i in reversed(range(num_downsamples)):
            up_mask += [nn.Upsample(scale_factor=2),
                        base_conv_block(self.get_num_filters(i + 1),
                                        self.get_num_filters(i))]

        conv_mask = [Conv2dBlock(num_filters, 1, kernel_size, padding=padding,
                                 nonlinearity='sigmoid')]

        self.down_lbl = nn.Sequential(*down_lbl)
        self.down_img = nn.Sequential(*down_img)
        self.res_flow = nn.Sequential(*res_mask)
        self.up_flow = nn.Sequential(*up_mask)
        self.conv_mask = nn.Sequential(*conv_mask)

    def forward(self, pose, img_warp):
        r"""Flow generator forward.

        Args:
           label (4D tensor) : Input label tensor.
           img_prev (4D tensor) : Previously generated image tensors.
        Returns:
            (tuple):
              - flow (4D tensor) : Generated flow map.
              - mask (4D tensor) : Generated occlusion mask.
        """
        #downsample = self.down_lbl(pose) + self.down_img(img_warp)
        downsample = torch.cat([self.down_lbl(pose),self.down_img(img_warp)], dim=1)
        res = self.res_flow(downsample)
        flow_feat = self.up_flow(res)
        mask = self.conv_mask(flow_feat)

        return mask




