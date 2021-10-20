"""
This file contrains functions for io and coordinate transformation
"""  
import numpy as np
import os
import patoolib
import torch
import json
from easydict import EasyDict as edict
import yaml

def worker_init_fn(worker_id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([worker_id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))

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

def extract_valid_keypoints(pts, thres=0.0):

    output = np.zeros((1, 3))
    
    valid = (pts[:, 2] > thres)
    if valid.sum() > 5:
        output =  np.mean( pts[valid, :], axis=0, keepdims=True)
    return output

def select_largest_bb(jointdicts, thres = 0.01):

    target_idx = -1
    target_height = -1

    for i, joint_dict in enumerate(jointdicts):
        np_joints = np.array(joint_dict['pose_keypoints_2d']).copy()
        np_joints = np_joints.reshape((-1, 3))[:15, :]
        x_cor = np_joints [:, 0]
        y_cor = np_joints [:, 1]
        confidence = np_joints [:, 2]
        valid = (confidence > thres)
        if valid.sum() < 8:
            continue
        width = np.amax(x_cor[np.where(valid)]) - np.amin(x_cor[np.where(valid)])
        height = np.amax(y_cor[np.where(valid)]) - np.amin(y_cor[np.where(valid)])

        area = width * height
        if area > target_height:
            target_height = area
            target_idx = i

    return target_idx

def openpose2motion(json_dir, scale=None, offset=None, max_frame=None, thres=0.000):
    '''
    Helper function to convert openpose folder to np motion data, 
    change the coordinate by scale and offset
    '''
    json_files = sorted(os.listdir(json_dir)) 
    #length = max_frame if max_frame is not None else len(json_files) // 8 * 8
    length = max_frame if max_frame is not None else len(json_files)

    json_files = json_files[:length]
    json_files = [os.path.join(json_dir, x) for x in json_files if x.endswith('.json')]

    motion = []
    for path in json_files:
        with open(path) as f:

            jointDict = json.load(f)

            # Find the largest bbox in the image (to avoid detection errors)
            if len(jointDict['people']) > 0:
                idx = select_largest_bb(jointDict['people'])
            else:
                idx = -1

            if idx != -1:
                joint_indice = list(range(0,15) ) + [19, 22]
                pts = np.array(jointDict['people'][idx]['pose_keypoints_2d']).reshape(-1, 3) [joint_indice]
                l_pts = extract_valid_keypoints(np.array(jointDict['people'][idx]['hand_left_keypoints_2d']).reshape(-1, 3)) 
                r_pts = extract_valid_keypoints(np.array(jointDict['people'][idx]['hand_right_keypoints_2d']).reshape(-1, 3)) 
                joints = np.concatenate((pts, l_pts, r_pts), axis=0)

                confidence = joints [:, 2].copy()
                valid = (confidence > thres)
                np_joints = np.zeros_like (joints)
                np_joints[valid, :] = joints [valid, :]
                np_joints[:, 2] = confidence

            else: # if no people inside
                if len(motion) > 1:
                    np_joints = motion[-1]
                else: 
                    np_joints = np.zeros((19, 3))
                
            motion.append(np_joints)

    motion = np.stack(motion, axis=0)
    conf = motion[:, :, -1]
    valid = (conf > thres)

    motion =  motion[:, :, :2] # use x,y coordinate only

    if scale is None:
        scale = 512
    if offset is None:
        offset = 256

    motion = (motion - offset) / scale
    motion [~valid, :] = 0.0

    return motion.transpose(1,2,0), conf[:,:,np.newaxis].transpose(1,2,0), (scale, offset)


def motion2openpose(motion, conf, save_json_dir, scale=512.0, offset=256.0, sample_rate=8):
    '''
    write np motion data in to a folder of openpose file, use 
    '''
    if not os.path.exists(save_json_dir):
        print("Creating directory: {}".format(save_json_dir))
        os.makedirs(save_json_dir)

    seq_len = motion.shape[-1]

    openpose_dict = {}
    openpose_dict['version'] = 1.3
    openpose_dict['people'] = []

    person_dict = {}
    person_dict['person_id'] = [-1]
    person_dict['pose_keypoints_2d'] = []
    person_dict['face_keypoints_2d'] = []
    person_dict['hand_left_keypoints_2d'] = []
    person_dict['hand_right_keypoints_2d'] = []
    person_dict['pose_keypoints_3d'] = []
    person_dict['face_keypoints_3d'] = []
    person_dict['hand_left_keypoints_3d'] = []
    person_dict['hand_right_keypoints_3d'] = []

    openpose_dict['people'].append(person_dict)

    for i in range(seq_len):

        joints = motion[:, :, i].copy() * scale + offset
        confidence = conf [:, :, i].copy()

        output = np.concatenate([joints[:15], confidence[:15]],axis=1)
        output = np.pad(output, ((0,10),(0,0)), 'constant', constant_values=0.0)

        output[19, :] = np.concatenate([ joints[15], confidence[15]], axis=None) 
        output[22, :] = np.concatenate([ joints[16], confidence[16]], axis=None) 

        openpose_dict['people'][0]['pose_keypoints_2d'] = output.reshape(-1).tolist()


        openpose_dict['people'][0]['hand_left_keypoints_2d'] = \
                  np.concatenate([ joints[17], confidence[17]], axis=None) [np.newaxis,:].repeat(21,axis=0).reshape(-1).tolist()

        openpose_dict['people'][0]['hand_right_keypoints_2d'] = \
                  np.concatenate([ joints[18], confidence[18]], axis=None) [np.newaxis,:].repeat(21,axis=0).reshape(-1).tolist()

        save_path = os.path.join (save_json_dir, '{:06d}_keypoints.json'.format(i))
        
        with open(save_path, 'w') as fp:
            json.dump(openpose_dict, fp)

