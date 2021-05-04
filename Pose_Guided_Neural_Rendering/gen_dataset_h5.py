import sys, os
import torch
import h5py
import numpy as np
import json

import argparse


def extract_valid_keypoints(pts, thres=0.00):

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
        if valid.sum() < 3:
            continue
        width = np.amax(x_cor[np.where(valid)]) - np.amin(x_cor[np.where(valid)])
        height = np.amax(y_cor[np.where(valid)]) - np.amin(y_cor[np.where(valid)])

        area = width * height
        if area > target_height:
            target_height = area
            target_idx = i

    return target_idx

def main(args):

    dataset_path = args.image_path
    
    gt_image_dir = os.path.join(args.image_path, 'test', 'gt')
    gt_pose_dir = os.path.join(args.image_path, 'test', 'poses') 
    gt_cain_dir = os.path.join(args.image_path, 'test', 'DAIN') 
    
    train_image_dir = os.path.join(args.image_path, 'train', 'frames')
    train_pose_dir = os.path.join(args.image_path, 'train', 'poses')
    train_fake_dir = os.path.join(args.image_path, 'train', 'DAIN')
    
    outfile = h5py.File('Dataset.h5', 'w')
    dt = h5py.special_dtype(vlen=np.uint8)

    # process test image folders
    subfolderlist = [f for f in sorted(os.listdir(gt_image_dir)) if os.path.isdir(os.path.join(gt_image_dir, f)) ]

    for subfolder in subfolderlist:
        print(subfolder)

        sub_group = outfile.create_group(subfolder)

        ############################
        
        gt_image_list =  [os.path.join(gt_image_dir, subfolder, f) for f in sorted(os.listdir(os.path.join(gt_image_dir, subfolder))) if f.endswith(('png','jpg')) ] 
        dset1 = sub_group.create_dataset( 'gt_images', 
                            (len(gt_image_list),), maxshape=(len(gt_image_list),), chunks=True, dtype=dt)
        
        for i, path in enumerate(gt_image_list):
            image = open(path, 'rb')
            binary_data = image.read()
            dset1[i] = np.frombuffer(binary_data, dtype=np.uint8)

        ############################
        gt_cain_list =  [os.path.join(gt_cain_dir, subfolder, f) for f in sorted(os.listdir(os.path.join(gt_cain_dir, subfolder))) if f.endswith('png') ] 
        dset2 = sub_group.create_dataset( 'gt_dain', 
                            (len(gt_cain_list),), maxshape=(len(gt_cain_list),), chunks=True, dtype=dt)

        for i, path in enumerate(gt_cain_list):
            image = open(path, 'rb')
            binary_data = image.read()
            dset2[i] = np.frombuffer(binary_data, dtype=np.uint8)
        
        ############################
        gt_pose_list =  [os.path.join(gt_pose_dir, subfolder, f) for f in sorted(os.listdir(os.path.join(gt_pose_dir, subfolder))) if f.endswith('json') ] 

        gt_motion = []

        for i, path in enumerate(gt_pose_list):
            with open(path) as f:
                jointDict = json.load(f)

                if len(jointDict['people']) > 0:
                    idx = select_largest_bb(jointDict['people'])
                else:
                    idx = -1

                if idx != -1:
                    joint_indice = list(range(0,15) ) + [19, 22]
                    pts = np.array(jointDict['people'][idx]['pose_keypoints_2d']).reshape(-1, 3) [joint_indice, :]
                    l_pts = extract_valid_keypoints(np.array(jointDict['people'][idx]['hand_left_keypoints_2d']).reshape(-1, 3))
                    r_pts = extract_valid_keypoints(np.array(jointDict['people'][idx]['hand_right_keypoints_2d']).reshape(-1, 3))
                    gt_joints = np.concatenate((pts, l_pts, r_pts), axis=0)
                else:
                    gt_joints = np.zeros( (19, 3) )
                
                gt_motion.append(gt_joints)
        
        gt_motion = np.stack(gt_motion, axis=0)

        dset3 = sub_group.create_dataset( 'gt_poses', data=gt_motion, dtype=np.float64)

        ############################


    # process training image folders

    subfolderlist = [f for f in sorted(os.listdir(train_image_dir)) if os.path.isdir(os.path.join(train_image_dir, f)) ]

    for subfolder in subfolderlist:
        print(subfolder)
        if "/{}".format(str(subfolder)) not in outfile:
            sub_group = outfile.create_group(subfolder)
        else:
            sub_group = outfile[subfolder]
        train_image_list =  [os.path.join(train_image_dir, subfolder, f) for f in sorted(os.listdir(os.path.join(train_image_dir, subfolder))) if f.endswith(('png','jpg')) ] 
        dset4 = sub_group.create_dataset( 'train_images', 
                            (len(train_image_list),), maxshape=(len(train_image_list),), chunks=True, dtype=dt)
        
        for i, path in enumerate(train_image_list):
            image = open(path, 'rb')
            binary_data = image.read()
            dset4[i] = np.frombuffer(binary_data, dtype=np.uint8)

        ############################
        train_fake_list =  [os.path.join(train_fake_dir, subfolder, f) for f in sorted(os.listdir(os.path.join(train_fake_dir, subfolder))) if f.endswith('png') ] 
        dset5 = sub_group.create_dataset( 'train_dain', 
                            (len(train_fake_list),), maxshape=(len(train_fake_list),), chunks=True, dtype=dt)

        for i, path in enumerate(train_fake_list):
            image = open(path, 'rb')
            binary_data = image.read()
            dset5[i] = np.frombuffer(binary_data, dtype=np.uint8)
        
        ############################
        train_pose_list =  [os.path.join(train_pose_dir, subfolder, f) for f in sorted(os.listdir(os.path.join(train_pose_dir, subfolder))) if f.endswith('json') ] 

        train_motion = []

        for i, path in enumerate(train_pose_list):
            with open(path) as f:
                jointDict = json.load(f)

                if len(jointDict['people']) > 0:
                    idx = select_largest_bb(jointDict['people'])
                else:
                    idx = -1

                if idx != -1:
                    joint_indice = list(range(0,15) ) + [19, 22]
                    pts = np.array(jointDict['people'][idx]['pose_keypoints_2d']).reshape(-1, 3) [joint_indice, :]
                    l_pts = extract_valid_keypoints(np.array(jointDict['people'][idx]['hand_left_keypoints_2d']).reshape(-1, 3))
                    r_pts = extract_valid_keypoints(np.array(jointDict['people'][idx]['hand_right_keypoints_2d']).reshape(-1, 3))
                    gt_joints = np.concatenate((pts, l_pts, r_pts), axis=0)
                else:
                    gt_joints = np.zeros( (19, 3) )
                
                train_motion.append(gt_joints)
        
        train_motion = np.stack(train_motion, axis=0)

        dset6 = sub_group.create_dataset( 'train_poses', data=train_motion, dtype=np.float64)

    outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process image folder to H5 file')

    parser.add_argument("-i", "--image_path", default='./example', type=str, help="Path of the input image folder")

    main(parser.parse_args())



