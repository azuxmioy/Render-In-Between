import torch
import numpy as np
import h5py
import os
import cv2
from tqdm import tqdm
from piq import ssim, psnr

from datasets import find_dataset_using_name
from utils.utils import tensor2images, read_json_keypoint
from utils.visualize import make_video

from PIL import Image
import albumentations as A
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def get_alb_transform(modelsize, method=cv2.INTER_CUBIC):
    '''
    For test we simply resize the imput the our model size
    '''
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

        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.transform_i = get_alb_transform(modelsize = (self.dataset.height, self.dataset.width))


    def evaluate_from_dataset(self, model, epoch, max_keyframes=None, use_gpu=False, gen_vid=True):
        '''
        use the h5 test split for evaluation
        '''
        
        print('evaluate in epoch:{:>3}'.format(epoch))
        model.eval()

        for test_video in self.test_vid_list:

            with h5py.File(self.dataset.root, "r") as f:

                total_frames = len(list(f[test_video]['gt_images']))
                sample_rate = 2 
                seq_len = total_frames if max_keyframes is None else min (max_keyframes * sample_rate + 1, total_frames)

                
            results = {}
            results['gt'] = []
            results['dain'] = []
            results['gen'] = []
            results['fuse'] = []
            results['skeleton'] = []
            results['pose'] = []
            results['mask'] = []

            cnt = 0
            dain_psnr = 0.0
            dain_ssim = 0.0
            gen_psnr = 0.0
            gen_ssim = 0.0

            print('Generating video frames: {}'.format(test_video))


            with torch.no_grad():

                for i in tqdm(range(seq_len)):
                    
                    tar_data = self.dataset.get_testdata_with_key(test_video, i)

                    results['gt'].append(tar_data['img'].data.cpu())
                    results['dain'].append(tar_data['dain'].data.cpu())
                    results['skeleton'].append(tar_data['skel'].data.cpu())
                    results['pose'].append(tar_data['pose'].data.cpu())

                    if (i % sample_rate) == 0 :
                        results['gen'].append(tar_data['img'].data.cpu())
                        results['mask'].append(torch.zeros(1, 1, tar_data['img'].shape[-2], tar_data['img'].shape[-1]).cpu())
                        results['fuse'].append(tar_data['img'].data.cpu())

                    else:
                        prev_label = torch.cat([results['skeleton'][-1],results['pose'][-1]], dim=1).to(self.device)
                        tar_label = torch.cat([tar_data['skel'],tar_data['pose']], dim=1).to(self.device)
                        
                        prev_img = results['fuse'][-1].to(self.device)
                        dain_img = tar_data['dain'].to(self.device)

                        pred_img, pred_mask = model (tar_label, prev_label, dain_img, prev_img)

                        mask = pred_mask.repeat(1,3,1,1)
                        not_mask = (1 - mask)
                        fuse_img = pred_img * mask + dain_img * not_mask

                        results['gen'].append(pred_img.data.cpu())
                        results['mask'].append(pred_mask.data.cpu())
                        results['fuse'].append(fuse_img.data.cpu())

                        if i % (sample_rate / 2) == 0 and i % (sample_rate ) != 0:
                            gt = tar_data['img'].clone().to(self.device)
                            ps, ss = self.compute_metrics(fuse_img.cpu(), gt.cpu(), mask=tar_data['mask'])
                            gen_psnr += ps
                            gen_ssim += ss

                            ps, ss = self.compute_metrics(dain_img.cpu(), gt.cpu(), mask=tar_data['mask'])
                            dain_psnr += ps
                            dain_ssim += ss
                            cnt += 1
            '''
            print('DAIN')
            print('PSNR: {:6f}'.format( (dain_psnr / cnt ) ))
            print('SSIM: {:6f}'.format( (dain_ssim / cnt ) ))

            print('Generate')
            print('PSNR: {:6f}'.format( (gen_psnr / cnt ) ))
            print('SSIM: {:6f}'.format( (gen_ssim / cnt ) ))
            '''
            metric = {}

            metric['DAIN_PSNR'] = dain_psnr / cnt
            metric['DAIN_SSIM'] = dain_ssim / cnt
            metric['OURS_PSNR'] = gen_psnr / cnt
            metric['OURS_SSIM'] = gen_ssim / cnt

            if gen_vid:
                make_video(results, os.path.join(self.cfg.eval_dir, '{}_{:03d}.mp4'.format(test_video, epoch) ), fps=30)
        
        
        model.train()
        return metric

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

    def evaluate_from_folder (self, model, train_dir, dain_dir, pose_dir, save_dir, gt_dir=None, gen_vid=False):
        '''
        save generated image results to the save_dir folder 
        '''
        subfolderlist = [f for f in sorted(os.listdir(pose_dir)) if os.path.isdir(os.path.join(pose_dir, f)) ]
        model.eval()
        for subfolder in subfolderlist:
            print('Evaluating {} .....'.format(subfolder))

            frames_dir = os.path.join(save_dir, subfolder)
            if not os.path.exists(frames_dir):
                print("Creating directory: {}".format(frames_dir))
                os.makedirs(frames_dir)

            image_list = [os.path.join(train_dir, subfolder, f) for f in sorted(os.listdir(os.path.join(train_dir, subfolder))) if f.endswith(('jpg', 'png')) ] 
            dain_list = [os.path.join(dain_dir, subfolder, f) for f in sorted(os.listdir(os.path.join(dain_dir, subfolder))) if f.endswith(('jpg', 'png')) ] 
            pose_list = [os.path.join(pose_dir, subfolder, f) for f in sorted(os.listdir(os.path.join(pose_dir, subfolder))) if f.endswith(('json')) ] 

            if gt_dir is not None:
                gtlist = [os.path.join(gt_dir, subfolder, f) for f in sorted(os.listdir(os.path.join(gt_dir, subfolder))) if f.endswith(('jpg', 'png')) ] 


            num_keyframes = len(image_list)
            num_frame = len(pose_list)

            sample_rate = 2 ** int( np.log2 ( (num_frame - 1) / (num_keyframes - 1)) )
            seq_len = (num_keyframes-1) * sample_rate + 1
            
            results = {}
            results['gt'] = []
            results['dain'] = []
            results['skeleton'] = []
            results['pose'] = []
            results['gen'] = []
            results['fuse'] = []
            results['mask'] = []

            with torch.no_grad():

                ####### Pre load DAIN and skeleton data #########
                for i in tqdm(range(seq_len)):
                    key_index = i // sample_rate
                    dain_pil = np.asarray(Image.open(dain_list[i]))

                    if gt_dir is not None:
                        img_pil = np.asarray(Image.open(gtlist[i]))
                    else:
                        img_pil = np.asarray(Image.open(image_list[key_index]))

                    pose = read_json_keypoint(pose_list[i])
                    joint_lists = [(pose[i,0], pose[i,1]) for i in range(pose.shape[0])]
                    joint_conf = [ pose[i,2] for i in range(pose.shape[0])]

                    transformed_image = self.transform_i(image=img_pil, keypoints=joint_lists) ['image']
                    transformed_landmark = self.transform_i(image=img_pil, keypoints=joint_lists) ['keypoints']
                    transformed_dain = self.transform_i(image=dain_pil, keypoints=joint_lists) ['image']

                    skeleton = self.dataset._generate_skeleton(transformed_landmark, joint_conf, self.dataset.height, self.dataset.width)
                    posemap = self.dataset._generate_pose_map(transformed_landmark, joint_conf, self.dataset.height, self.dataset.width)

                    torch_gt = self.dataset.to_tensor_norm(transformed_image).unsqueeze(0)
                    torch_dain = self.dataset.to_tensor_norm(transformed_dain).unsqueeze(0)

                    torch_sk = self.dataset.to_tensor_norm(skeleton).unsqueeze(0)
                    torch_pose = torch.from_numpy(posemap).float().unsqueeze(0)
                    

                    results['gt'].append(torch_gt)
                    results['dain'].append(torch_dain.to(self.device))
                    results['skeleton'].append(torch_sk.to(self.device))
                    results['pose'].append(torch_pose.to(self.device))

                ####### Inference time #########
                for i in tqdm(range(seq_len)):

                    if (i % sample_rate)  == 0 :
                    #if i ==0 or i == seq_len-1:
                        results['gen'].append(results['gt'][i])
                        results['mask'].append(torch.zeros(1, 1, torch_gt.shape[-2], torch_gt.shape[-1]))
                        results['fuse'].append(results['gt'][i])

                    else:
                        prev_index = i - 1

                        prev_label = torch.cat([results['skeleton'][prev_index],results['pose'][prev_index]], dim=1).to(self.device)
                        tar_label = torch.cat([results['skeleton'][i],results['pose'][i]], dim=1).to(self.device)

                        prev_img = results['fuse'][-1].to(self.device)
                        dain_img = results['dain'][i].to(self.device)

                        pred_img, pred_mask = model (tar_label, prev_label, dain_img, prev_img)
                        mask = pred_mask.repeat(1,3,1,1)
                        not_mask = (1 - mask)
                        fuse_img = pred_img * mask + dain_img * not_mask

                        results['gen'].append(pred_img.data.cpu())
                        results['mask'].append(pred_mask.data.cpu())
                        results['fuse'].append(fuse_img.data.cpu())


                    save_name = os.path.join(frames_dir, os.path.basename(dain_list[i]))[:-4] + '.png'
                    Image.fromarray(tensor2images(results['fuse'][i])).save(save_name)

            if gen_vid:
                make_video(results, os.path.join(save_dir, '{}.mp4'.format(subfolder) ), fps=30)
