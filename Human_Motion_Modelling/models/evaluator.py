import torch
import json
import copy
import numpy as np
import h5py
import os
import shutil
from tqdm import tqdm
from datasets import find_dataset_using_name
from torch.utils.data import DataLoader
from itertools import combinations
from visualize.util import motion2gif
from utils.utils import motion2openpose

class Evaluator(object):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.cfg = cfg
        
        dataset_class = find_dataset_using_name('AMASS')
        self.dataset = dataset_class(self.cfg, self.cfg.h5_file, return_type=cfg.return_type, phase='test')
        self.rotation_axes = np.array(cfg.rotation_axes)
        self.return_type = cfg.return_type
        try:
            # make sure that the key starts with '/'
            self.rand_view = np.load(self.cfg.evaluate_view).copy()
            print('## Evaluation view loaded! ##')
        except:
            print("Can't read viewpoint file for evaluation, now a new file")
            #self.rand_view = np.random.uniform(-1, 1, 1000) * np.pi
            self.rand_view = np.random.uniform(-self.rotation_axes, self.rotation_axes, (1000,3))
            np.save(self.cfg.evaluate_view, self.rand_view)
            print('############################')
            print('   Evaluation random view   ')
            print(self.rand_view)
            print('############################')



        self.scale_unit = 128
        self.offset = 256
        self.model = None

    def set_model(self, model):
        self.model = copy.deepcopy(model)

    def infer_h5_file(self, model, save_file_name):
        
        model.eval()

        print('Evaluating.....')

        outfile = h5py.File(save_file_name, 'w')

        output_group = outfile.create_group('pred')
        gt_group = outfile.create_group('gt')
        interp_group = outfile.create_group('interp')

        for i, (dataset_key, motion_key) in enumerate ( tqdm(self.dataset.samples)):

            view = self.rand_view[ i % 1000]
            start, gt, interp, input, encoder_mask, decoder_mask = \
                    self.dataset.get_2d_motion_with_key (dataset_key, motion_key, view, self.cfg.test_sample_rate)

            output = model.inference( input, interp, encoder_mask, decoder_mask, self.cfg.test_sample_rate)

            output_group.create_dataset("{}_{}".format(dataset_key, motion_key), 
                data=self._post_process(output, start),
                dtype=np.float64)

            gt_group.create_dataset("{}_{}".format(dataset_key, motion_key), 
                data=gt,
                dtype=np.float64)

            interp_group.create_dataset("{}_{}".format(dataset_key, motion_key), 
                data=self._post_process(interp.unsqueeze(0), start),
                dtype=np.float64)

        outfile.close()
        model.train()

    def evaluate_from_h5(self, h5_path):
        mse_global = 0.0
        mae_global = 0.0
        mma_global = 0.0
        mse_interp = 0.0
        mae_interp = 0.0
        mma_interp = 0.0

        cnt = 0
        cnt2 = 0

        with h5py.File(h5_path, "r") as f:

            for name in list(f['gt']):

                try:
                    gt = np.array(f['gt'][name]).copy()
                    infer = np.array(f['pred'][name]).copy()
                    interp = np.array(f['interp'][name]).copy()

                except:
                    raise ValueError("[Error] Can't read key %s from h5 dataset" % name)

                gt_global = self._relocate(gt, localize=False)
                infer_global = self._relocate(infer, localize=False)
                interp_global = self._relocate(interp, localize=False)

                J, D, T = gt_global.shape
                cnt += J * D * T
                cnt2 +=1
                #root_idx = 8
                # use interp body center
                #infer_global = infer_global + ( interp_global[root_idx : root_idx+1, :, :] - infer_global[root_idx : root_idx+1, :, :] )

                mse_global += np.sum((gt_global - infer_global) ** 2)
                mae_global += np.sum(np.abs(gt_global - infer_global))
                mma_global += np.amax(np.abs(gt_global - infer_global))

                mse_interp += np.sum((gt_global - interp_global) ** 2)
                mae_interp += np.sum(np.abs(gt_global - interp_global))
                mma_interp += np.amax(np.abs(gt_global - interp_global))

                '''
                gt_local = self._relocate(gt, localize=True)
                infer_local = self._relocate(infer, localize=True)

                mse_local += np.sum((gt_local - infer_local) ** 2)
                mae_local += np.sum(np.abs(gt_local - infer_local))
                '''

        result = {}
        
        result['mse_global'] = mse_global / cnt
        result['mae_global'] = mae_global / cnt
        result['max_global'] = mma_global / cnt2
        result['mse_interp'] = mse_interp / cnt
        result['mae_interp'] = mae_interp / cnt
        result['max_interp'] = mma_interp / cnt2

        return result 
        
    def visualize_skeleton(self, h5_path, save_img_path, samples=5):
        if not os.path.exists(save_img_path):
            print("Creating directory: {}".format(save_img_path))
            os.makedirs(save_img_path)

        with h5py.File(h5_path, "r") as f:
            print('Visualizing as resuts as GIF.....')
            gif_idx = np.random.randint(len(list(f['gt'])), size=samples)

            for idx in tqdm(gif_idx):
                name = list(f['gt'])[idx]
                try:
                    gt = np.array(f['gt'][name]).copy()
                    infer = np.array(f['pred'][name]).copy()
                    interp = np.array(f['interp'][name]).copy()

                except:
                    raise ValueError("[Error] Can't read key %s from h5 dataset" % name)

                print(np.amax(infer))
                print(np.amin(infer))

                gt_render = self._to_render(self._relocate(gt, localize=False))
                infer_render= self._to_render(self._relocate(infer, localize=False))
                interp_global = self._to_render(self._relocate(interp, localize=False))

                path = os.path.join(save_img_path, name + '_global.gif')
                motion2gif (infer_render, 512, 512, path, relocate=False)
                path = os.path.join(save_img_path, name + '_interp.gif')
                motion2gif (interp_global, 512, 512, path, relocate=False)
                path = os.path.join(save_img_path, name + '_gt.gif')
                motion2gif (gt_render, 512, 512, path, relocate=False)
        print('Visualization done!!!')
      

    def interpolate_openpose(self, json_dir, sample_rate, save_dir):


        # Source motion
        (scale, offset, conf), input_motion, interp_motion, \
            encoder_mask, decoder_mask = self.dataset.get_openpose_data(json_dir, sample_rate)

        # Retarget motion

        output = self.model.inference( input_motion, interp_motion, encoder_mask, decoder_mask, sample_rate)
        
        out = self._post_process(output, 0)

        interp = self._post_process(interp_motion.unsqueeze(0), 0)

        if os.path.exists(save_dir['pred_dir']):
            shutil.rmtree(save_dir['pred_dir'])
            print('detete {} ... '.format(save_dir['pred_dir']))
        if os.path.exists(save_dir['linear_dir']):
            shutil.rmtree(save_dir['linear_dir'])
            print('detete {} ... '.format(save_dir['linear_dir']))

        motion2openpose (out, conf, save_dir['pred_dir'], scale=scale, offset=offset, sample_rate=sample_rate)
        motion2openpose (interp, conf, save_dir['linear_dir'], scale=scale, offset=offset, sample_rate=sample_rate)


        # Target motion

        print('visualize done!')

    
    def _denormalize(self, data):
        return data * self.dataset.std_pose[:, :, np.newaxis] + self.dataset.mean_pose[:, :, np.newaxis]

    def _globalize (self, data, start):

        velocity = data[-1].copy()

        if self.return_type == '3D':
            D = 3
            motion_inv = np.r_[np.zeros((1, 3, data.shape[-1])), data[0:-1]]
        else:
            D = 2
            motion_inv = np.r_[data[:8], np.zeros((1, 2, data.shape[-1])), data[8:-1]]

        # restore centre position
        '''
        centers = np.zeros_like(velocity)
        sum = 0
        for i in range(data.shape[-1]):
            sum += velocity[:, i]
            centers[:, i] = sum
        centers += start.reshape([D, 1])
        '''
        centers = velocity
        
        return motion_inv + centers.reshape((1, D, -1))

    def _post_process (self, data, start):

        if self.return_type == '3D':
            D = 3
        else:
            D = 2
        
        data = data.detach().cpu().numpy()[0].reshape(-1, D, data.shape[-1])
        data = self._denormalize(data)
        data = self._globalize(data, start)

        return data

    def _relocate(self, motion, localize=False):

        if self.return_type == '3D':
            root_idx = 0
        else:
            root_idx = 8

        if localize:
            # fix hip joint in all frames
            motion = motion - motion[root_idx : root_idx+1, :, :]
        else:
            # align hip joint in the first frame
            center = motion[root_idx, :, 0]
            motion = motion - center[np.newaxis, :, np.newaxis]

        return motion

    def _to_render(self, motion, center=None):
        return motion * self.scale_unit +  self.offset



    