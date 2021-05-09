import torch
import os
import math
import cv2
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time
import h5py
import patoolib
from easydict import EasyDict as edict
import csv
import json
from PIL import Image
from collections import OrderedDict
from scipy.ndimage import gaussian_filter1d

from torch.optim import lr_scheduler
import torch.nn.functional as F

def to_gpu(data):
    for key, item in data.items():
        if torch.is_tensor(item):
            data[key] = item.cuda()
    return data

# Parse the config yaml file.
def get_config(config):
    with open(config, 'r') as stream:
        return edict(yaml.load(stream, Loader=yaml.FullLoader))

def write_config(save_file, cfg):
    with open(save_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

def dataset_to_h5(dataset_path, outfile_path):

    outfile = h5py.File(outfile_path, 'w')

    subgroups = ['train', 'test']

    for group_name in subgroups:
        sub_group = outfile.create_group(group_name)
        model_group = {}
        model_lists = sorted([o for o in os.listdir(os.path.join(dataset_path, group_name)) if os.path.isdir(os.path.join(dataset_path, group_name, o))])

        for model in model_lists:

            if "/{}/{}".format(str(group_name), str(model)) not in outfile:
                model_group[str(model)] = sub_group.create_group(str(model))

            motion_path = os.path.join(dataset_path, group_name, model)
            motion_lists = sorted([o for o in os.listdir(motion_path) if os.path.isdir(os.path.join(motion_path, o))])

            for motion in motion_lists:

                model_group[str(model)].create_dataset(str(motion), 
                        data=np.load(os.path.join(motion_path, motion, '{}.npy'.format(str(motion)))),
                        dtype=np.float64)

    outfile.close()

def weights_init(init_type='normal', gain=0.02, bias=None):
    r"""Initialize weights in the network.

    Args:
        init_type (str): The name of the initialization scheme.
        gain (float): The parameter that is required for the initialization
            scheme.
        bias (object): If not ``None``, specifies the initialization parameter
            for bias.

    Returns:
        (obj): init function to be applied.
    """

    def init_func(m):
        r"""Init function

        Args:
            m: module to be weight initialized.
        """
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and (
                class_name.find('Conv') != -1 or
                class_name.find('Linear') != -1 or
                class_name.find('Embedding') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':
                m.reset_parameters()
            else:
                raise NotImplementedError(
                    'initialization method [%s] is '
                    'not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                if bias is not None:
                    bias_type = getattr(bias, 'type', 'normal')
                    if bias_type == 'normal':
                        bias_gain = getattr(bias, 'gain', 0.5)
                        init.normal_(m.bias.data, 0.0, bias_gain)
                    else:
                        raise NotImplementedError(
                            'initialization method [%s] is '
                            'not implemented' % bias_type)
                else:
                    init.constant_(m.bias.data, 0.0)
    return init_func


def create_zip_code_files(output_file='code_copy.zip'):
    archive_files = []
    archive_files += [os.path.join('datasets',f) for f in os.listdir('datasets') if f.endswith('py')]
    archive_files += [os.path.join('models',f) for f in os.listdir('models') if f.endswith('py')]
    archive_files += [os.path.join('utils',f) for f in os.listdir('utils') if f.endswith('py')]
    archive_files += [f for f in os.listdir('.') if f.endswith('py')]
    patoolib.create_archive(output_file, archive_files)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def remove_module_key(state_dict):
    for key in list(state_dict.keys()):
        if 'module' in key:
            state_dict[key.replace('module.','')] = state_dict.pop(key)
    return state_dict

def load_state_dict(net, path):

    if os.path.isfile(path):
        checkpoint = torch.load(path)
        print("=> Loaded checkpoint '{}'".format(path))
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(path))

    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    
    state_dict = remove_module_key(checkpoint)
    net.load_state_dict(state_dict)


def tensor2images(image_tensor, num_images=1, imtype=np.uint8):
    batch_size = image_tensor.shape[0]
    num_images = min(int(batch_size), num_images)

    # Iterate through num of images
    image_list = []
    for i in range(num_images):
        image_numpy = image_tensor[i].cpu().float().numpy()
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        if image_numpy.shape[0] == 1:
            # posemap
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        else:
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * std + mean
            image_numpy = (np.clip(image_numpy, 0, 1)) * 255.0
        
        image_list.append(image_numpy[None, ...])
    
    output_images = np.concatenate(image_list, axis=0).astype(imtype)

    if num_images == 1:
        return output_images[0, ...]
    else:
        return output_images


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory

def visualize_pose(image_tensor, num_images=1):
    # Convert to numpy if its torch tensor
    # import ipdb; ipdb.set_trace()
    b, c, h, w = image_tensor.size()
    num_images = min(b, num_images)

    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.cpu().float().numpy()
    
    preds2, maxvals2 = get_max_preds(image_tensor)


    # The skeleton map
    skeleton = np.array([
        [0, 1], [1, 8],
        [1, 2], [2, 3], [3, 4],
        [1, 5], [5, 6], [6, 7],
        [8, 9], [9, 10], [10, 11],
        [8 ,12], [12, 13], [13,14]
    ])
    
    # Iterate through all the results
    image_list = []
    for i in range(num_images):
        image = np.zeros((h, w, 3),dtype=np.uint8)
    
        for mat, val in zip(preds2[i], maxvals2[i]):
            x, y = int(mat[0]), int(mat[1])
            if x < 0 or y < 0 or val <= 0:
                continue
            else:
                cv2.circle(image, (x, y), 3, (0, 255, 0), 2)
    
        # Plot the skeleton
        for sk in skeleton:
            x1, y1 = int(preds2[i, sk[0], 0]), int(preds2[i, sk[0], 1])
            x2, y2 = int(preds2[i, sk[1], 0]), int(preds2[i, sk[1], 1])
            val1 = maxvals2[i, sk[0]]
            val2 = maxvals2[i, sk[0]]
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 :
                continue
            else:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        image_list.append(image[None, ...])
    
    # Compatible with old version api when num_images == 1
    if num_images == 1:
        return np.concatenate(image_list, axis=0)[0, ...]
    else:
        return np.concatenate(image_list, axis=0)

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def crop_face_from_output(image, input_label, crop_smaller=0):
    r"""Crop out the face region of the image (and resize if necessary to feed
    into generator/discriminator).

    Args:
        data_cfg (obj): Data configuration.
        image (NxC1xHxW tensor or list of tensors): Image to crop.
        input_label (NxC2xHxW tensor): Input label map.
        crop_smaller (int): Number of pixels to crop slightly smaller region.
    Returns:
        output (NxC1xHxW tensor or list of tensors): Cropped image.
    """
    if type(image) == list:
        return [crop_face_from_output(im, input_label, crop_smaller)
                for im in image]

    output = None
    face_size = image.shape[-2] // 32 * 8
    for i in range(input_label.size(0)):
        ys, ye, xs, xe = get_face_bbox_for_output(input_label[i:i + 1],
                                                  crop_smaller=crop_smaller)
        output_i = F.interpolate(image[i:i + 1, -3:, ys:ye, xs:xe],
                                 size=(face_size, face_size), mode='bilinear',
                                 align_corners=True)
        # output_i = image[i:i + 1, -3:, ys:ye, xs:xe]
        output = torch.cat([output, output_i]) if i != 0 else output_i
    return output


def get_face_bbox_for_output(pose, crop_smaller=0):
    r"""Get pixel coordinates of the face bounding box.

    Args:
        data_cfg (obj): Data configuration.
        pose (NxCxHxW tensor): Pose label map.
        crop_smaller (int): Number of pixels to crop slightly smaller region.
    Returns:
        output (list of int): Face bbox.
    """
    if pose.dim() == 3:
        pose = pose.unsqueeze(0)
    elif pose.dim() == 5:
        pose = pose[-1, -1:]
    _, _, h, w = pose.size()

    face_idx = 3

    face = (pose[:, face_idx] > 0).nonzero(as_tuple=False)

    ylen = xlen = h // 32 * 8
    if face.size(0):
        y, x = face[:, 1], face[:, 2]
        ys, ye = y.min().item(), y.max().item()
        xs, xe = x.min().item(), x.max().item()

        xc, yc = (xs + xe) // 2, (ys * 3 + ye * 2) // 5
        ylen = int((xe - xs) * 2.5)

        ylen = xlen = min(w, max(32, ylen))
        yc = max(ylen // 2, min(h - 1 - ylen // 2, yc))
        xc = max(xlen // 2, min(w - 1 - xlen // 2, xc))
    else:
        yc = h // 4
        xc = w // 2

    ys, ye = yc - ylen // 2, yc + ylen // 2
    xs, xe = xc - xlen // 2, xc + xlen // 2
    if crop_smaller != 0:  # Crop slightly smaller region inside face.
        ys += crop_smaller
        xs += crop_smaller
        ye -= crop_smaller
        xe -= crop_smaller
    return [ys, ye, xs, xe]


def crop_hand_from_output(image, input_label):
    r"""Crop out the hand region of the image.

    Args:
        data_cfg (obj): Data configuration.
        image (NxC1xHxW tensor or list of tensors): Image to crop.
        input_label (NxC2xHxW tensor): Input label map.
    Returns:
        output (NxC1xHxW tensor or list of tensors): Cropped image.
    """
    if type(image) == list:
        return [crop_hand_from_output(im, input_label)
                for im in image]

    output = None
    for i in range(input_label.size(0)):
        coords = get_hand_bbox_for_output(input_label[i:i + 1])
        if coords:
            for coord in coords:
                ys, ye, xs, xe = coord
                output_i = image[i:i + 1, -3:, ys:ye, xs:xe]
                output = torch.cat([output, output_i]) \
                    if output is not None else output_i
    return output


def get_hand_bbox_for_output(pose):
    r"""Get coordinates of the hand bounding box.

    Args:
        data_cfg (obj): Data configuration.
        pose (NxCxHxW tensor): Pose label map.
    Returns:
        output (list of int): Hand bbox.
    """
    if pose.dim() == 3:
        pose = pose.unsqueeze(0)
    elif pose.dim() == 5:
        pose = pose[-1, -1:]
    _, _, h, w = pose.size()
    ylen = xlen = h // 64 * 8

    coords = []
    colors = [[0.95, 0.5, 0.95], [0.95, 0.95, 0.5]]
    for i, color in enumerate(colors):
        idx = -2 if i == 0 else -1
        hand = (pose[:, idx] > 0).nonzero(as_tuple=False)

        if hand.size(0):
            y, x = hand[:, 1], hand[:, 2]
            ys, ye, xs, xe = y.min().item(), y.max().item(), \
                x.min().item(), x.max().item()
            xc, yc = (xs + xe) // 2, (ys + ye) // 2
            yc = max(ylen // 2, min(h - 1 - ylen // 2, yc))
            xc = max(xlen // 2, min(w - 1 - xlen // 2, xc))
            ys, ye, xs, xe = yc - ylen // 2, yc + ylen // 2, \
                xc - xlen // 2, xc + xlen // 2
            coords.append([ys, ye, xs, xe])
    return coords



def get_scheduler(optimizer, opt, iterations=-1):
    if 'lr_policy' not in opt or opt.lr_policy == 'constant':
        scheduler = None # constant scheduler

    elif opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1- opt.step_size) / float(opt.max_epoch-opt.step_size)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size,
                                 gamma=opt.gamma, last_epoch=iterations)
    elif opt.lr_policy == 'multistep':
        step = opt.step_size
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[step, step+step//2, step+step//2+step//4],
                                        gamma=opt.gamma, last_epoch=iterations)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler