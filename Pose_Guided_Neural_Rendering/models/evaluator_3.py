import torch
import json
import copy
import numpy as np
import h5py
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import imageio
import time

from tqdm import tqdm
from datasets import find_dataset_using_name
from torch.utils.data import DataLoader
from itertools import combinations
from piq import ssim, psnr

from visualize.util import motion2gif
from utils.utils import tensor2images
from PIL import Image
import albumentations as A
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def get_alb_transform(modelsize, method=cv2.INTER_CUBIC):
    
    transform_list = []
 
    transform_list.append( A.Resize(height=modelsize[0], width=modelsize[1], interpolation=method, always_apply=True) )

    return A.Compose(transform_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

class Evaluator(object):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.cfg = cfg
        
        dataset_class = find_dataset_using_name('HSM_auto')

        self.dataset = dataset_class(cfg, cfg.h5_file, phase='test')

        self.test_vid_list = cfg.test_video_list

        self.gen_mode = cfg.test_gen_mode #  [keyframe | previous]


        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)

        self.scale_unit = 128
        self.offset = 256
        self.model = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.transform_i = get_alb_transform(modelsize = (self.dataset.height, self.dataset.width))


    def set_model(self, model):
        self.model = copy.deepcopy(model)


    def evaluate_from_dataset(self, model, epoch, max_keyframes=None, use_gpu=False, gen_vid=True):
        
        print('evaluate in epoch:{:>3}'.format(epoch))
        model.eval()
        store_device = torch.device('cuda') if use_gpu else torch.device('cpu')

        for test_video in self.test_vid_list:

            with h5py.File(self.dataset.root, "r") as f:

                total_frames = len(list(f[test_video]['gt_images']))
                num_keyframes = (total_frames+1) // 2
                sample_rate = 2 

                seq_len = total_frames

                seq_len = seq_len if max_keyframes is None else min (max_keyframes * sample_rate + 1, seq_len)

                indice = np.arange(seq_len, dtype=int)
                chunk =( (indice) // sample_rate +1 ) *  sample_rate 

                
            results = {}
            results['gt'] = []
            results['cain'] = []
            results['gen'] = []
            results['fuse'] = []
            results['skeleton'] = []
            results['pose'] = []
            results['mask'] = []

            cnt = 0
            cain_psnr = 0.0
            cain_ssim = 0.0
            gen_psnr = 0.0
            gen_ssim = 0.0

            print('Generating video frames: {}'.format(test_video))


            with torch.no_grad():

                for i in tqdm(range(seq_len)):
                    
                    tar_data = self.dataset.get_testdata_with_key(test_video, i)

                    results['gt'].append(tar_data['img'].data.cpu())
                    results['cain'].append(tar_data['cain'].data.cpu())
                    results['skeleton'].append(tar_data['skel'].data.cpu())
                    results['pose'].append(tar_data['pose'].data.cpu())

                    if (i % sample_rate) == 0 :
                        results['gen'].append(tar_data['img'].data.cpu())
                        results['mask'].append(torch.zeros(1, 1, tar_data['img'].shape[-2], tar_data['img'].shape[-1]).cpu())
                        results['fuse'].append(tar_data['img'].data.cpu())

                    else:

                        ref_index = int(chunk[i])
                        ref_data = self.dataset.get_testdata_with_key(test_video, ref_index)


                        prev_label = torch.cat([results['skeleton'][-1],results['pose'][-1]], dim=1).to(self.device)
                        tar_label = torch.cat([tar_data['skel'],tar_data['pose']], dim=1).to(self.device)
                        ref_label = torch.cat([ref_data['skel'],ref_data['pose']], dim=1).to(self.device)

                        #prev_label = results['skeleton'][-1].to(self.device)
                        #tar_label = tar_data['skel'].to(self.device)
                        
                        prev_img = results['fuse'][-1].to(self.device)
                        cain_img = tar_data['cain'].to(self.device)
                        ref_img = ref_data['img'].to(self.device)
                        pred_img, pred_mask = model (tar_label, prev_label, ref_label, cain_img, prev_img, ref_img)
                        #pred_img, pred_mask = model (tar_data['skel'], results['skeleton'][-1], cain_img, prev_img)

                        mask = pred_mask.repeat(1,3,1,1)
                        not_mask = (1 - mask)
                        fuse_img = pred_img * mask + cain_img * not_mask

                        results['gen'].append(pred_img.data.cpu())
                        results['mask'].append(pred_mask.data.cpu())
                        results['fuse'].append(fuse_img.data.cpu())

                        if i % (sample_rate / 2) == 0 and i % (sample_rate ) != 0:
                            gt = tar_data['img'].clone().to(self.device)
                            ps, ss = self.compute_metrics(fuse_img, gt, mask=tar_data['mask'])
                            gen_psnr += ps
                            gen_ssim += ss

                            ps, ss = self.compute_metrics(cain_img, gt, mask=tar_data['mask'])
                            cain_psnr += ps
                            cain_ssim += ss
                            cnt += 1

            print('CAIN')
            print('PSNR: {:6f}'.format( (cain_psnr / cnt ) ))
            print('SSIM: {:6f}'.format( (cain_ssim / cnt ) ))

            print('Generate')
            print('PSNR: {:6f}'.format( (gen_psnr / cnt ) ))
            print('SSIM: {:6f}'.format( (gen_ssim / cnt ) ))

            if gen_vid:
                self.make_video(results, os.path.join(self.cfg.eval_dir, '{}_{:03d}.mp4'.format(test_video, epoch) ), fps=30)
        
        
        model.train()


    def compute_metrics(self, pred, target, mask=None):
        pred = pred
        target = target
        pred_norm =  torch.clamp(pred * self.std.to(pred) + self.mean.to(pred), 0, 1)
        target_norm =  torch.clamp(target * self.std.to(target) + self.mean.to(target), 0, 1)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1,3,1,1).to(pred)
            pred_norm = pred_norm * mask
            target_norm = target_norm * mask
        
        PSNR = psnr(pred_norm, target_norm, data_range=1., reduction='mean')
        SSIM = ssim(pred_norm, target_norm, data_range=1.)

        return PSNR, SSIM

    def make_video(self, results, save_path, fps=30, save_frame=False):

        pred, fuse, target, cain, skeleton, mask = \
            results['gen'], results['fuse'], results['gt'], results['cain'], results['skeleton'], results['mask']

        videowriter = imageio.get_writer(save_path, fps=fps)
        if save_frame:
            frames_dir = os.path.join(os.path.dirname(save_path), 'output_frames')
            if not os.path.exists(frames_dir):
                print("Creating directory: {}".format(frames_dir))
                os.makedirs(frames_dir)

        frame_idx = 0
        for gen, fu, gt, fake, sk, msk in tqdm(zip(pred, fuse, target, cain, skeleton, mask), total = len(pred)):

            fig = plt.figure(figsize=(48, 26), dpi=40, facecolor='white')

            ax1 = plt.subplot(2,3,1)
            ax1.set_title('Predict', fontsize=60, color='b')
            ax2 = plt.subplot(2,3,2)
            ax2.set_title('Mask', fontsize=60, color='b')
            ax3 = plt.subplot(2,3,3)
            ax3.set_title('Fuse', fontsize=60, color='b')
            ax4 = plt.subplot(2,3,4)
            ax4.set_title('CAIN', fontsize=60, color='b')
            ax5 = plt.subplot(2,3,5)
            ax5.set_title('Ground Truth', fontsize=60, color='b')
            ax6 = plt.subplot(2,3,6)
            ax6.set_title('Skeleton', fontsize=60, color='b')

            pil_gen = Image.fromarray(tensor2images(gen))
            pil_fake = Image.fromarray(tensor2images(fake))
            pil_gt = Image.fromarray(tensor2images(gt))
            pil_sk = Image.fromarray(tensor2images(sk))
            pil_msk = Image.fromarray(tensor2images(msk))
            pil_fu = Image.fromarray(tensor2images(fu))

            ax1.imshow(pil_gen)
            ax2.imshow(pil_msk)
            ax3.imshow(pil_fu)
            ax4.imshow(pil_fake)
            ax5.imshow(pil_gt)
            ax6.imshow(pil_sk)

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            if save_frame:
                Image.fromarray(img).save(os.path.join(frames_dir, "%04d.png" % frame_idx))
        
            videowriter.append_data(img)
            frame_idx +=1
            plt.close()

        videowriter.close()


    def read_json_keypoint(self, json_dir):

        def extract_valid_keypoints(pts, thres=0.00):

            output = np.zeros((1, 3))
    
            valid = (pts[:, 2] > thres)
            if valid.sum() > 5:
                output =  np.mean( pts[valid, :], axis=0, keepdims=True)
            return output

        def select_largest_bb(jointdicts, thres = 0.1):

            target_idx = -1
            target_height = -1

            for i, joint_dict in enumerate(jointdicts):
                np_joints = np.array(joint_dict['pose_keypoints_2d']).copy()
                np_joints = np_joints.reshape((-1, 3))[:15, :]
                x_cor = np_joints [:, 0]
                y_cor = np_joints [:, 1]
                confidence = np_joints [:, 2]
                valid = (confidence > thres)
                if valid.sum() < 4:
                    continue
                width = np.amax(x_cor[np.where(valid)]) - np.amin(x_cor[np.where(valid)])
                height = np.amax(y_cor[np.where(valid)]) - np.amin(y_cor[np.where(valid)])

                area = width * height
                if area > target_height:
                    target_height = area
                    target_idx = i

            return target_idx

        with open(json_dir) as f:
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
                
        return gt_joints

    def evaluate_from_folder (self, model, train_dir, cain_dir, pose_dir, save_dir, gt_dir=None, use_gpu=True, gen_vid=True):

        store_device = torch.device('cuda') if use_gpu else torch.device('cpu')
        subfolderlist = [f for f in sorted(os.listdir(pose_dir)) if os.path.isdir(os.path.join(pose_dir, f)) ]
        model.eval()
        for subfolder in subfolderlist:
            print('Evaluating {} .....'.format(subfolder))

            frames_dir = os.path.join(save_dir, subfolder)
            if not os.path.exists(frames_dir):
                print("Creating directory: {}".format(frames_dir))
                os.makedirs(frames_dir)

            image_list = [os.path.join(train_dir, subfolder, f) for f in sorted(os.listdir(os.path.join(train_dir, subfolder))) if f.endswith(('jpg', 'png')) ] 
            cain_list = [os.path.join(cain_dir, subfolder, f) for f in sorted(os.listdir(os.path.join(cain_dir, subfolder))) if f.endswith(('jpg', 'png')) ] 
            pose_list = [os.path.join(pose_dir, subfolder, f) for f in sorted(os.listdir(os.path.join(pose_dir, subfolder))) if f.endswith(('json')) ] 

            if gt_dir is not None:
                gtlist = [os.path.join(gt_dir, subfolder, f) for f in sorted(os.listdir(os.path.join(gt_dir, subfolder))) if f.endswith(('jpg', 'png')) ] 


            num_keyframes = len(image_list)
            num_frame = len(pose_list)

            sample_rate = 2 ** int( np.log2 ( (num_frame - 1) / (num_keyframes - 1)) )
            seq_len = (num_keyframes-1) * sample_rate + 1
            
            results = {}
            results['gt'] = []
            results['cain'] = []
            results['skeleton'] = []
            results['pose'] = []
            results['gen'] = []
            results['fuse'] = []
            results['mask'] = []

            with torch.no_grad():

                ####### Pre load CAIN and skeleton data #########
                for i in tqdm(range(seq_len)):
                    key_index = i // sample_rate
                    cain_pil = np.asarray(Image.open(cain_list[i]))

                    if gt_dir is not None:
                        img_pil = np.asarray(Image.open(gtlist[i]))
                    else:
                        img_pil = np.asarray(Image.open(image_list[key_index]))

                    pose = self.read_json_keypoint(pose_list[i])
                    joint_lists = [(pose[i,0], pose[i,1]) for i in range(pose.shape[0])]
                    joint_conf = [ pose[i,2] for i in range(pose.shape[0])]

                    transformed_image = self.transform_i(image=img_pil, keypoints=joint_lists) ['image']
                    transformed_landmark = self.transform_i(image=img_pil, keypoints=joint_lists) ['keypoints']
                    transformed_cain = self.transform_i(image=cain_pil, keypoints=joint_lists) ['image']

                    skeleton = self.dataset._generate_skeleton(transformed_landmark, joint_conf, self.dataset.height, self.dataset.width)
                    posemap = self.dataset._generate_pose_map(transformed_landmark, joint_conf, self.dataset.height, self.dataset.width)

                    torch_gt = self.dataset.to_tensor_norm(transformed_image).unsqueeze(0)
                    torch_cain = self.dataset.to_tensor_norm(transformed_cain).unsqueeze(0)

                    torch_sk = self.dataset.to_tensor_norm(skeleton).unsqueeze(0)
                    torch_pose = torch.from_numpy(posemap).float().unsqueeze(0)
                    

                    results['gt'].append(torch_gt)
                    results['cain'].append(torch_cain.to(store_device))
                    results['skeleton'].append(torch_sk.to(store_device))
                    results['pose'].append(torch_pose.to(store_device))

                ####### Inference time #########
                for i in tqdm(range(seq_len)):

                    if (i % sample_rate)  == 0 :
                    #if i ==0 or i == seq_len-1:
                        results['gen'].append(results['gt'][i])
                        results['mask'].append(torch.zeros(1, 1, torch_gt.shape[-2], torch_gt.shape[-1]))
                        results['fuse'].append(results['gt'][i])

                    else:
                        ref_index = (i // sample_rate + 1 ) * sample_rate
                        prev_index = i - 1

                        prev_label = torch.cat([results['skeleton'][prev_index],results['pose'][prev_index]], dim=1).to(self.device)
                        tar_label = torch.cat([results['skeleton'][i],results['pose'][i]], dim=1).to(self.device)
                        ref_label = torch.cat([results['skeleton'][ref_index], results['pose'][ref_index]], dim=1).to(self.device)

                        prev_img = results['fuse'][-1].to(self.device)
                        cain_img = results['cain'][i].to(self.device)
                        ref_img = results['gt'][ref_index].clone().to(self.device)

                        pred_img, pred_mask = model (tar_label, prev_label, ref_label, cain_img, prev_img, ref_img)
                        mask = pred_mask.repeat(1,3,1,1)
                        not_mask = (1 - mask)
                        fuse_img = pred_img * mask + cain_img * not_mask

                        results['gen'].append(pred_img.data.cpu())
                        results['mask'].append(pred_mask.data.cpu())
                        results['fuse'].append(fuse_img.data.cpu())


                    save_name = os.path.join(frames_dir, os.path.basename(cain_list[i]))[:-4] + '.png'
                    Image.fromarray(tensor2images(results['fuse'][i])).save(save_name)

            if gen_vid:
                self.make_video(results, os.path.join(save_dir, '{}.mp4'.format(subfolder) ), fps=30)


        model.eval()

            #Geting low fps frames
