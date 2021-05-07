  
import numpy as np
import os
import cv2
import math
import patoolib
from tqdm import tqdm
from PIL import Image
import torch

from easydict import EasyDict as edict
import yaml


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


def create_zip_code_files(output_file='code_copy.zip'):
    archive_files = []
    archive_files += [os.path.join('datasets',f) for f in os.listdir('datasets') if f.endswith('py')]
    archive_files += [os.path.join('models',f) for f in os.listdir('models') if f.endswith('py')]
    archive_files += [os.path.join('utils',f) for f in os.listdir('utils') if f.endswith('py')]
    archive_files += [os.path.join('visualize',f) for f in os.listdir('visualize') if f.endswith('py')]
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
