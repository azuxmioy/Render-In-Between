# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import importlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Conv2dBlock
import functools

class Discriminator(nn.Module):
    r"""Image and video discriminator constructor.

    Args:
        dis_cfg (obj): Discriminator part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file
    """

    def __init__(self, dis_cfg):
        super().__init__()
        num_input_channels = dis_cfg.input_label_nc
        num_img_channels = dis_cfg.input_image_nc
        self.num_frames_D = dis_cfg.num_frames_D

        num_netD_input_channels = (num_input_channels + num_img_channels)
        self.use_few_shot = dis_cfg.few_shot

        if self.use_few_shot:
            num_netD_input_channels *= 2
        self.net_D = MultiPatchDiscriminator(dis_cfg.image,
                                             num_netD_input_channels)

        self.add_dis_cfg = getattr(dis_cfg, 'additional_discriminators', None)
        if self.add_dis_cfg is not None:
            for name in self.add_dis_cfg:
                add_dis_cfg = self.add_dis_cfg[name]
                num_ch = num_img_channels * (2 if self.use_few_shot else 1)
                setattr(self, 'net_D_' + name,
                        MultiPatchDiscriminator(add_dis_cfg, num_ch))

    def forward(self, label, real_image, fake_image, ref_label, ref_image, raw_image=None, fg_mask=None):
        r"""Discriminator forward.

        Args:
            data (dict): Input data.
            net_G_output (dict): Generator output.
            past_frames (list of tensors): Past real frames / generator outputs.
        Returns:
            (tuple):
              - output (dict): Discriminator output.
              - past_frames (list of tensors): New past frames by adding
                current outputs.
        """
        # Only operate on the latest output frame.
        if label.dim() == 5:
            label = label[:, -1]
        if self.use_few_shot:
            # Concat references with label map as discriminator input.
            label = torch.cat([label, ref_label, ref_image], dim=1)

        output = dict()

        # Individual frame loss.
        pred_real, pred_fake = self.discrminate_image(self.net_D, label,
                                                      real_image, fake_image)
        output['indv'] = dict()
        output['indv']['pred_real'] = pred_real
        output['indv']['pred_fake'] = pred_fake

        if raw_image is not None:
            pred_real, pred_fake = self.discrminate_image(
                self.net_D, label,
                real_image * fg_mask,
                raw_image * fg_mask)
            output['raw'] = dict()
            output['raw']['pred_real'] = pred_real
            output['raw']['pred_fake'] = pred_fake

        # Additional GAN loss on specific regions.
        if self.add_dis_cfg is not None:
            for name in self.add_dis_cfg:
                # Crop corresponding regions in the image according to the
                # crop function.
                add_dis_cfg = self.add_dis_cfg[name]
                file, crop_func = add_dis_cfg.crop_func.split('::')
                file = importlib.import_module(file)
                crop_func = getattr(file, crop_func)

                real_crop = crop_func(real_image, label)
                fake_crop = crop_func(raw_image, label)
                if self.use_few_shot:
                    ref_crop = crop_func( ref_image, label)
                    if ref_crop is not None:
                        real_crop = torch.cat([real_crop, ref_crop], dim=1)
                        fake_crop = torch.cat([fake_crop, ref_crop], dim=1)

                # Feed the crops to specific discriminator.
                if fake_crop is not None:
                    net_D = getattr(self, 'net_D_' + name)
                    pred_real, pred_fake = \
                        self.discrminate_image(net_D, None,
                                               real_crop, fake_crop)
                else:
                    pred_real = pred_fake = None
                output[name] = dict()
                output[name]['pred_real'] = pred_real
                output[name]['pred_fake'] = pred_fake

        return output

    def discrminate_image(self, net_D, real_A, real_B, fake_B):
        r"""Discriminate individual images.

        Args:
            net_D (obj): Discriminator network.
            real_A (NxC1xHxW tensor): Input label map.
            real_B (NxC2xHxW tensor): Real image.
            fake_B (NxC2xHxW tensor): Fake image.
        Returns:
            (tuple):
              - pred_real (NxC3xH2xW2 tensor): Output of net_D for real images.
              - pred_fake (NxC3xH2xW2 tensor): Output of net_D for fake images.
        """
        if real_A is not None:
            real_AB = torch.cat([real_A, real_B], dim=1)
            fake_AB = torch.cat([real_A, fake_B], dim=1)
        else:
            real_AB, fake_AB = real_B, fake_B

        pred_real = net_D.forward(real_AB)
        pred_fake = net_D.forward(fake_AB)
        return pred_real, pred_fake


##########################################################################################

class NLayerPatchDiscriminator(nn.Module):
    r"""Patch Discriminator constructor.

    Args:
        kernel_size (int): Convolution kernel size.
        num_input_channels (int): Num. of channels in the real/fake image.
        num_filters (int): Num. of base filters in a layer.
        num_layers (int): Num. of layers for the patch discriminator.
        max_num_filters (int): Maximum num. of filters in a layer.
        activation_norm_type (str): batch_norm/instance_norm/none/....
        weight_norm_type (str): none/spectral_norm/weight_norm
    """

    def __init__(self,
                 kernel_size,
                 num_input_channels,
                 num_filters,
                 num_layers,
                 max_num_filters,
                 activation_norm_type,
                 weight_norm_type):
        super(NLayerPatchDiscriminator, self).__init__()
        self.num_layers = num_layers
        padding = int(np.floor((kernel_size - 1.0) / 2))
        nonlinearity = 'leakyrelu'
        base_conv2d_block = \
            functools.partial(Conv2dBlock,
                              kernel_size=kernel_size,
                              padding=padding,
                              weight_norm_type=weight_norm_type,
                              activation_norm_type=activation_norm_type,
                              nonlinearity=nonlinearity,
                              # inplace_nonlinearity=True,
                              order='CNA')
        layers = [[base_conv2d_block(
            num_input_channels, num_filters, stride=2)]]
        for n in range(num_layers):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            stride = 2 if n < (num_layers - 1) else 1
            layers += [[base_conv2d_block(num_filters_prev, num_filters,
                                          stride=stride)]]
        layers += [[Conv2dBlock(num_filters, 1,
                                3, 1,
                                padding,
                                weight_norm_type=weight_norm_type)]]
        for n in range(len(layers)):
            setattr(self, 'layer' + str(n), nn.Sequential(*layers[n]))

    def forward(self, input_x):
        r"""Patch Discriminator forward.

        Args:
            input_x (N x C x H1 x W2 tensor): Concatenation of images and
                semantic representations.
        Returns:
            (tuple):
              - output (N x 1 x H2 x W2 tensor): Discriminator output value.
                Before the sigmoid when using NSGAN.
              - features (list): lists of tensors of the intermediate
                activations.
        """
        res = [input_x]
        for n in range(self.num_layers + 2):
            layer = getattr(self, 'layer' + str(n))
            x = res[-1]
            res.append(layer(x))
        output = res[-1]
        features = res[1:-1]
        return output, features


##########################################################################################

class MultiPatchDiscriminator(nn.Module):
    r"""Multi-resolution patch discriminator.

    Args:
        dis_cfg (obj): Discriminator part of the yaml config file.
        num_input_channels (int): Number of input channels.
    """

    def __init__(self, dis_cfg, num_input_channels):
        super(MultiPatchDiscriminator, self).__init__()
        kernel_size = getattr(dis_cfg, 'kernel_size', 4)
        num_filters = getattr(dis_cfg, 'num_filters', 64)
        max_num_filters = getattr(dis_cfg, 'max_num_filters', 512)
        num_discriminators = getattr(dis_cfg, 'num_discriminators', 3)
        num_layers = getattr(dis_cfg, 'num_layers', 3)
        activation_norm_type = getattr(dis_cfg, 'activation_norm_type', 'none')
        weight_norm_type = getattr(dis_cfg, 'weight_norm_type',
                                   'spectral_norm')
        self.nets_discriminator = []
        for i in range(num_discriminators):
            net_discriminator = NLayerPatchDiscriminator(
                kernel_size,
                num_input_channels,
                num_filters,
                num_layers,
                max_num_filters,
                activation_norm_type,
                weight_norm_type)
            self.add_module('discriminator_%d' % i, net_discriminator)
            self.nets_discriminator.append(net_discriminator)

    def forward(self, input_x):
        r"""Multi-resolution patch discriminator forward.

        Args:
            input_x (N x C x H x W tensor) : Concatenation of images and
                semantic representations.
        Returns:
            (dict):
              - output (list): list of output tensors produced by individual
                patch discriminators.
              - features (list): list of lists of features produced by
                individual patch discriminators.
        """
        output_list = []
        features_list = []
        input_downsampled = input_x
        for name, net_discriminator in self.named_children():
            if not name.startswith('discriminator_'):
                continue
            output, features = net_discriminator(input_downsampled)
            output_list.append(output)
            features_list.append(features)
            input_downsampled = F.interpolate(
                input_downsampled, scale_factor=0.5, mode='bilinear',
                align_corners=True)
        output_x = dict()
        output_x['output'] = output_list
        output_x['features'] = features_list
        return output_x


