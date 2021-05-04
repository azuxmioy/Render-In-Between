from __future__ import absolute_import

import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

#################################################################################
####               GAN Loss implementation in imaginaire                    #####
#################################################################################


@torch.jit.script
def fuse_math_min_mean_pos(x):
    r"""Fuse operation min mean for hinge loss computation of positive
    samples"""
    minval = torch.min(x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


@torch.jit.script
def fuse_math_min_mean_neg(x):
    r"""Fuse operation min mean for hinge loss computation of negative
    samples"""
    minval = torch.min(-x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


class GANLoss(nn.Module):
    r"""GAN loss constructor.

    Args:
        gan_mode (str): Type of GAN loss. ``'hinge'``, ``'least_square'``,
            ``'non_saturated'``, ``'wasserstein'``.
        target_real_label (float): The desired output label for real images.
        target_fake_label (float): The desired output label for fake images.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.gan_mode = gan_mode
        print('GAN mode: %s' % gan_mode)

    def forward(self, dis_output, t_real, dis_update=True):
        r"""GAN loss computation.

        Args:
            dis_output (tensor or list of tensors): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise
                uses the fake label as target.
            dis_update (bool): If ``True``, the loss will be used to update the
                discriminator, otherwise the generator.
        Returns:
            loss (tensor): Loss value.
        """
        if isinstance(dis_output, list):
            # For multi-scale discriminators.
            # In this implementation, the loss is first averaged for each scale
            # (batch size and number of locations) then averaged across scales,
            # so that the gradient is not dominated by the discriminator that
            # has the most output values (highest resolution).
            loss = 0
            for dis_output_i in dis_output:
                assert isinstance(dis_output_i, torch.Tensor)
                loss += self.loss(dis_output_i, t_real, dis_update)
            return loss / len(dis_output)
        else:
            return self.loss(dis_output, t_real, dis_update)

    def loss(self, dis_output, t_real, dis_update=True):
        r"""GAN loss computation.

        Args:
            dis_output (tensor): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise
                uses the fake label as target.
            dis_update (bool): Updating the discriminator or the generator.
        Returns:
            loss (tensor): Loss value.
        """
        if not dis_update:
            assert t_real, \
                "The target should be real when updating the generator."

        if self.gan_mode == 'non_saturated':
            target_tensor = self.get_target_tensor(dis_output, t_real)
            loss = F.binary_cross_entropy_with_logits(dis_output,
                                                      target_tensor)
        elif self.gan_mode == 'least_square':
            target_tensor = self.get_target_tensor(dis_output, t_real)
            loss = 0.5 * F.mse_loss(dis_output, target_tensor)
        elif self.gan_mode == 'hinge':
            if dis_update:
                if t_real:
                    loss = fuse_math_min_mean_pos(dis_output)
                else:
                    loss = fuse_math_min_mean_neg(dis_output)
            else:
                loss = -torch.mean(dis_output)
        elif self.gan_mode == 'wasserstein':
            if t_real:
                loss = -torch.mean(dis_output)
            else:
                loss = torch.mean(dis_output)
        else:
            raise ValueError('Unexpected gan_mode {}'.format(self.gan_mode))
        return loss

    def get_target_tensor(self, dis_output, t_real):
        r"""Return the target vector for the binary cross entropy loss
        computation.

        Args:
            dis_output (tensor): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise
                uses the fake label as target.
        Returns:
            target (tensor): Target tensor vector.
        """
        if t_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = dis_output.new_tensor(self.real_label)
            return self.real_label_tensor.expand_as(dis_output)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = dis_output.new_tensor(self.fake_label)
            return self.fake_label_tensor.expand_as(dis_output)



#################################################################################
#######            Perceptual Loss implementation in imaginaire         #########
#################################################################################

def apply_imagenet_normalization(input):
    r"""Normalize using ImageNet mean and std.

    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [-1, 1].

    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    # normalize the input back to [0, 1]
    normalized_input = (input + 1) / 2
    # normalize the input using the ImageNet mean and std
    mean = normalized_input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = normalized_input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (normalized_input - mean) / std
    return output

class PerceptualLoss(nn.Module):
    r"""Perceptual loss initialization.

    Args:
        cfg (Config): Configuration file.
        network (str) : The name of the loss network: 'vgg16' | 'vgg19'.
        layers (str or list of str) : The layers used to compute the loss.
        weights (float or list of float : The loss weights of each layer.
        criterion (str): The type of distance function: 'l1' | 'l2'.
        resize (bool) : If ``True``, resize the input images to 224x224.
        resize_mode (str): Algorithm used for resizing.
        instance_normalized (bool): If ``True``, applies instance normalization
            to the feature maps before computing the distance.
        num_scales (int): The loss will be evaluated at original size and
            this many times downsampled sizes.
    """

    def __init__(self, network='vgg19', layers='relu_4_1', weights=None,
                 criterion='l1', resize=False, resize_mode='bilinear',
                 instance_normalized=False, num_scales=1):
        super().__init__()
        if isinstance(layers, str):
            layers = [layers]
        if weights is None:
            weights = [1.] * len(layers)
        elif isinstance(layers, float) or isinstance(layers, int):
            weights = [weights]

        assert len(layers) == len(weights), \
            'The number of layers (%s) must be equal to ' \
            'the number of weights (%s).' % (len(layers), len(weights))
        if network == 'vgg19':
            self.model = _vgg19(layers)
        elif network == 'vgg16':
            self.model = _vgg16(layers)
        elif network == 'alexnet':
            self.model = _alexnet(layers)
        elif network == 'inception_v3':
            self.model = _inception_v3(layers)
        elif network == 'resnet50':
            self.model = _resnet50(layers)
        elif network == 'robust_resnet50':
            self.model = _robust_resnet50(layers)
        elif network == 'vgg_face_dag':
            self.model = _vgg_face_dag(layers)
        else:
            raise ValueError('Network %s is not recognized' % network)

        self.num_scales = num_scales
        self.layers = layers
        self.weights = weights
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)
        self.resize = resize
        self.resize_mode = resize_mode
        self.instance_normalized = instance_normalized
        print('Perceptual loss:')
        print('\tMode: {}'.format(network))


    def forward(self, inp, target):
        r"""Perceptual loss forward.

        Args:
           inp (4D tensor) : Input tensor.
           target (4D tensor) : Ground truth tensor, same shape as the input.

        Returns:
           (scalar tensor) : The perceptual loss.
        """
        # Perceptual loss should operate in eval mode by default.
        self.model.eval()
        inp, target = \
            apply_imagenet_normalization(inp), \
            apply_imagenet_normalization(target)
        if self.resize:
            inp = F.interpolate(
                inp, mode=self.resize_mode, size=(224, 224),
                align_corners=False)
            target = F.interpolate(
                target, mode=self.resize_mode, size=(224, 224),
                align_corners=False)

        # Evaluate perceptual loss at each scale.
        loss = 0
        for scale in range(self.num_scales):

            input_features, target_features = \
                    self.model(inp), self.model(target)

            for layer, weight in zip(self.layers, self.weights):
                # Example per-layer VGG19 loss values after applying
                # [0.03125, 0.0625, 0.125, 0.25, 1.0] weighting.
                # relu_1_1, 0.014698
                # relu_2_1, 0.085817
                # relu_3_1, 0.349977
                # relu_4_1, 0.544188
                # relu_5_1, 0.906261
                input_feature = input_features[layer]
                target_feature = target_features[layer].detach()
                if self.instance_normalized:
                    input_feature = F.instance_norm(input_feature)
                    target_feature = F.instance_norm(target_feature)

                loss += weight * self.criterion(input_feature,
                                                target_feature)
            # Downsample the input and target.
            if scale != self.num_scales - 1:
                inp = F.interpolate(
                    inp, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)
                target = F.interpolate(
                    target, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)

        return loss.float()


class _PerceptualNetwork(nn.Module):
    r"""The network that extracts features to compute the perceptual loss.

    Args:
        network (nn.Sequential) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        assert isinstance(network, nn.Sequential), \
            'The network needs to be of type "nn.Sequential".'
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        r"""Extract perceptual features."""
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                # If the current layer is used by the perceptual loss.
                output[layer_name] = x
        return output


def _vgg19(layers):
    r"""Get vgg19 layers"""
    network = torchvision.models.vgg19(pretrained=True).features
    layer_name_mapping = {1: 'relu_1_1',
                          3: 'relu_1_2',
                          6: 'relu_2_1',
                          8: 'relu_2_2',
                          11: 'relu_3_1',
                          13: 'relu_3_2',
                          15: 'relu_3_3',
                          17: 'relu_3_4',
                          20: 'relu_4_1',
                          22: 'relu_4_2',
                          24: 'relu_4_3',
                          26: 'relu_4_4',
                          29: 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg16(layers):
    r"""Get vgg16 layers"""
    network = torchvision.models.vgg16(pretrained=True).features
    layer_name_mapping = {1: 'relu_1_1',
                          3: 'relu_1_2',
                          6: 'relu_2_1',
                          8: 'relu_2_2',
                          11: 'relu_3_1',
                          13: 'relu_3_2',
                          15: 'relu_3_3',
                          18: 'relu_4_1',
                          20: 'relu_4_2',
                          22: 'relu_4_3',
                          25: 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _alexnet(layers):
    r"""Get alexnet layers"""
    network = torchvision.models.alexnet(pretrained=True).features
    layer_name_mapping = {0: 'conv_1',
                          1: 'relu_1',
                          3: 'conv_2',
                          4: 'relu_2',
                          6: 'conv_3',
                          7: 'relu_3',
                          8: 'conv_4',
                          9: 'relu_4',
                          10: 'conv_5',
                          11: 'relu_5'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _inception_v3(layers):
    r"""Get inception v3 layers"""
    inception = torchvision.models.inception_v3(pretrained=True)
    network = nn.Sequential(inception.Conv2d_1a_3x3,
                            inception.Conv2d_2a_3x3,
                            inception.Conv2d_2b_3x3,
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            inception.Conv2d_3b_1x1,
                            inception.Conv2d_4a_3x3,
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            inception.Mixed_5b,
                            inception.Mixed_5c,
                            inception.Mixed_5d,
                            inception.Mixed_6a,
                            inception.Mixed_6b,
                            inception.Mixed_6c,
                            inception.Mixed_6d,
                            inception.Mixed_6e,
                            inception.Mixed_7a,
                            inception.Mixed_7b,
                            inception.Mixed_7c,
                            nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    layer_name_mapping = {3: 'pool_1',
                          6: 'pool_2',
                          14: 'mixed_6e',
                          18: 'pool_3'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _resnet50(layers):
    r"""Get resnet50 layers"""
    resnet50 = torchvision.models.resnet50(pretrained=True)
    network = nn.Sequential(resnet50.conv1,
                            resnet50.bn1,
                            resnet50.relu,
                            resnet50.maxpool,
                            resnet50.layer1,
                            resnet50.layer2,
                            resnet50.layer3,
                            resnet50.layer4,
                            resnet50.avgpool)
    layer_name_mapping = {4: 'layer_1',
                          5: 'layer_2',
                          6: 'layer_3',
                          7: 'layer_4'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _robust_resnet50(layers):
    r"""Get robust resnet50 layers"""
    resnet50 = torchvision.models.resnet50(pretrained=False)
    state_dict = torch.utils.model_zoo.load_url(
        'http://andrewilyas.com/ImageNet.pt')
    new_state_dict = {}
    for k, v in state_dict['model'].items():
        if k.startswith('module.model.'):
            new_state_dict[k[13:]] = v
    resnet50.load_state_dict(new_state_dict)
    network = nn.Sequential(resnet50.conv1,
                            resnet50.bn1,
                            resnet50.relu,
                            resnet50.maxpool,
                            resnet50.layer1,
                            resnet50.layer2,
                            resnet50.layer3,
                            resnet50.layer4,
                            resnet50.avgpool)
    layer_name_mapping = {4: 'layer_1',
                          5: 'layer_2',
                          6: 'layer_3',
                          7: 'layer_4'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg_face_dag(layers):
    r"""Get vgg face layers"""
    network = torchvision.models.vgg16(num_classes=2622)
    state_dict = torch.utils.model_zoo.load_url(
        'http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/'
        'vgg_face_dag.pth')
    feature_layer_name_mapping = {
        0: 'conv1_1',
        2: 'conv1_2',
        5: 'conv2_1',
        7: 'conv2_2',
        10: 'conv3_1',
        12: 'conv3_2',
        14: 'conv3_3',
        17: 'conv4_1',
        19: 'conv4_2',
        21: 'conv4_3',
        24: 'conv5_1',
        26: 'conv5_2',
        28: 'conv5_3'}
    new_state_dict = {}
    for k, v in feature_layer_name_mapping.items():
        new_state_dict['features.' + str(k) + '.weight'] =\
            state_dict[v + '.weight']
        new_state_dict['features.' + str(k) + '.bias'] = \
            state_dict[v + '.bias']

    classifier_layer_name_mapping = {
        0: 'fc6',
        3: 'fc7',
        6: 'fc8'}
    for k, v in classifier_layer_name_mapping.items():
        new_state_dict['classifier.' + str(k) + '.weight'] = \
            state_dict[v + '.weight']
        new_state_dict['classifier.' + str(k) + '.bias'] = \
            state_dict[v + '.bias']

    network.load_state_dict(new_state_dict)

    class Flatten(nn.Module):
        r"""Flatten the tensor"""

        def forward(self, x):
            r"""Flatten it"""
            return x.view(x.shape[0], -1)

    layer_name_mapping = {
        1: 'avgpool',
        3: 'fc6',
        4: 'relu_6',
        6: 'fc7',
        7: 'relu_7',
        9: 'fc8'}
    seq_layers = [network.features, network.avgpool, Flatten()]
    for i in range(7):
        seq_layers += [network.classifier[i]]
    network = nn.Sequential(*seq_layers)
    return _PerceptualNetwork(network, layer_name_mapping, layers)

#################################################################################
#######        Feature Matching Loss implementation in imaginaire       #########
#################################################################################

class FeatureMatchingLoss(nn.Module):
    r"""Compute feature matching loss"""
    def __init__(self, criterion='l1'):
        super(FeatureMatchingLoss, self).__init__()
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)

    def forward(self, fake_features, real_features):
        r"""Return the target vector for the binary cross entropy loss
        computation.

        Args:
           fake_features (list of lists): Discriminator features of fake images.
           real_features (list of lists): Discriminator features of real images.

        Returns:
           (tensor): Loss value.
        """
        num_d = len(fake_features)
        dis_weight = 1.0 / num_d
        loss = fake_features[0][0].new_tensor(0)
        for i in range(num_d):
            for j in range(len(fake_features[i])):
                tmp_loss = self.criterion(fake_features[i][j],
                                          real_features[i][j].detach())
                loss += dis_weight * tmp_loss
        return loss

#################################################################################
#######        Feature Matching Loss implementation in imaginaire       #########
#################################################################################

class MaskedL1loss(nn.Module):

    def __init__(self, use_mask = True, alpha = 9.0):
        super(MaskedL1loss, self).__init__()
        self.alpha = alpha
        self.use_mask = use_mask

    def forward(self, inputs, mask, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            mask : human mask for higher reconstruction loss
            targets: ground truth labels with shape (num_classes)
        """
        global_loss = F.l1_loss(inputs, targets)
        if not self.use_mask:
            return global_loss

        #masks = torch.unsqueeze(mask, 1)
        #masks = masks.repeat(1,3,1,1)

        N = mask.sum(dtype=torch.float32)

        mask_loss =  0 if N < 1 else F.l1_loss(inputs*mask, targets*mask, reduction='sum') / N

        total_loss =( mask_loss * self.alpha + global_loss) / (1 + self.alpha)

        return total_loss


class MaskRegulationLoss(nn.Module):

    def __init__(self):
        super(MaskRegulationLoss, self).__init__()

    def image_gradient(self, x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top 
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    def forward(self, gen_mask, prior):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            mask : human mask for higher reconstruction loss
            targets: ground truth labels with shape (num_classes)
        """
        H = gen_mask.shape[-2]
        W = gen_mask.shape[-1]
        mask = prior.unsqueeze(1)
        not_mask = (1 - mask)

        mx, my = self.image_gradient(gen_mask)

        total_loss = torch.norm( mx, p=1) + \
                     torch.norm( my, p=1) + \
                     torch.norm( gen_mask, p=1)

                     #torch.norm( torch.mul(gen_mask, not_mask), p=1) + \
                     #torch.norm( torch.mul(mx, not_mask), p=1) + \
                     #torch.norm( torch.mul(my, not_mask), p=1) + \
                     #torch.norm( torch.mul(gen_mask, mask), p=1)

        return total_loss / (H * W * 4)



