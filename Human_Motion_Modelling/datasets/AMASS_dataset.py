from datasets.base_dataset import BaseDataset
import numpy as np
import torch
import h5py
import random
import os
from tqdm import tqdm
from utils.utils import openpose2motion

class AMASSDataset(BaseDataset):

    def __init__(self, cfg, root, return_type='3D', phase='train'):


        BaseDataset.__init__(self, cfg, root)

        print('############################')
        print('     Preparing Dataset      ')
        print('############################')

        self.phase = phase             # 'train' | 'test'
        self.return_type = return_type # 'render' for debug usage | 'network' return 2D joints | '3D' return 3D joints


        # Training settings
        self.train_sample_size = cfg.train_sample_size
        self.rate = cfg.train_sample_rate
        self.max_seq_length = cfg.max_seq_length

        self.train_noise = cfg.train_noise
        self.noise_weight = cfg.noise_weight
        self.noise_rate   = cfg.noise_rate
        self.joint_drop_rate = cfg.joint_drop_rate
        self.flip_rate = cfg.flip_rate

        # 2D projection settings
        self.rotation_aug = cfg.rotation_aug
        self.rotation_axes = np.array(cfg.rotation_axes)

        self.project = cfg.camera_project
        self.projection_noise = cfg.projection_noise and phase == 'train'
        self.focal_len = cfg.focal
        self.cam_depth = cfg.depth
        self.frame_boarder = cfg.frame_boarder

        # Rendering mode setting (for debug)
        self.scale_unit = 200

        # test mode setting
        self.evaluate_noise = cfg.evaluate_noise
        self.eval_rate = cfg.test_sample_rate

        self.openpose_scale= cfg.openpose_scale              
        self.openpose_offset= cfg.openpose_offset 

        if phase == 'train':
            self.sub_datasets = cfg.train_split
        elif phase == 'test':
            self.sub_datasets = cfg.test_split
        else: 
            raise NotImplementedError("Unknown split type!")

        # initialize data sample path
        self.samples = []

        try:
            for dataset_name in self.sub_datasets:
                with h5py.File(self.root, "r") as f:
                    motion_lists = list(f[dataset_name])

                tuple_list = [(dataset_name, motion_lists[i]) for i in range(len(motion_lists))] 
                self.samples.extend(tuple_list)
        except:
            print("Can't find AMASS_3D_joints.h5 dataset file, inference pose only")

        # precompute mean and variance for normalization
        mean_path = os.path.join(cfg.data_root, 'mean_pose_'+ str(self.return_type)
                            + '_' + str(self.project) + '_{:.0f}_{:.0f}'.format(self.focal_len, self.cam_depth) + '.npy' )
        std_path = os.path.join(cfg.data_root, 'std_pose_'+ str(self.return_type)
                            + '_' + str(self.project) + '_{:.0f}_{:.0f}'.format(self.focal_len, self.cam_depth) + '.npy' )       
        try:
            self.mean_pose = np.load(mean_path).copy()
            self.std_pose = np.load(std_path).copy()

        except:
            print("Can't read mean and std pose from h5 dataset, now compute from dataset...")
            self._compute_mean_pose()

            np.save(mean_path, self.mean_pose)
            np.save(std_path, self.std_pose)

        print(self.mean_pose)
        print(self.std_pose)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        dataset_key, motion_key = self.samples[index]
        
        # Get motion clips from h5 dataset file
        motion = self._get_h5_motion (dataset_key, motion_key)

        # Crop the full clips into fixed length traning motion
        crop_motion, mask, start = self._random_temproal_crop(motion)

        # generate viewpoint augmentation
        view = np.random.uniform(-self.rotation_axes, self.rotation_axes) * np.pi if self.rotation_aug else None

        # Rotate 3D joints based on viewpoint
        data_3d = self._rotate_motion_3d(self._centralize_motion(crop_motion), view)

        # Project 3D points to 2D using virtual cameras
        data_2d = self._project_2D(data_3d)

        if self.return_type == 'render':
            data = self._scale_to_rendering(data_2d)
            return{
                'data': data,
                'openpose': self._joints_to_openpose(data),
                'mask': mask,
                'dataset_key': dataset_key,
                'motion_key': motion_key,
                'start': start
            }

        elif self.return_type == 'network':

            encoder_mask, decoder_mask = self.generate_training_mask(mask, sample_rate = self.rate)

            data = self._normalize_to_network(data_2d.copy(), random_drop=False)

            input =  self._normalize_to_network(data_2d, random_drop=self.train_noise)

            # linear interpolation prior
            interp = self._get_interpolate_motion(input.copy(), self.rate)

            # ensure we remove intermediate frames as inputs
            input = input * ~encoder_mask.reshape(1,1,-1).astype(bool)

            return{
                'data': torch.from_numpy(data).float().reshape(-1,data.shape[-1]), 
                'input': torch.from_numpy(input).float().reshape(-1,input.shape[-1]),
                'interp': torch.from_numpy(interp).float().reshape(-1,interp.shape[-1]),
                'src_mask':  torch.from_numpy(encoder_mask).bool(),
                'tar_mask': torch.from_numpy(decoder_mask).bool(),
                'mask': torch.from_numpy(mask).bool(),
            }
        elif self.return_type == '3D':
            
            encoder_mask, decoder_mask = self.generate_training_mask(mask, sample_rate = self.rate)

            data = self._normalize_to_network(data_3d.copy(), random_drop=False)
            input = self._normalize_to_network(data_3d, random_drop=self.train_noise)
            interp = self._get_interpolate_motion(input.copy(), self.rate)

            input = input * ~encoder_mask.reshape(1,1,-1).astype(bool)

            return{
                'data': torch.from_numpy(data).float().reshape(-1,data.shape[-1]),
                'input': torch.from_numpy(input).float().reshape(-1,input.shape[-1]),
                'interp': torch.from_numpy(interp).float().reshape(-1,interp.shape[-1]),
                'src_mask':  torch.from_numpy(encoder_mask).bool(),
                'tar_mask': torch.from_numpy(decoder_mask).bool(),
                'mask': torch.from_numpy(mask).bool(),
            }
        else:
            raise NotImplementedError("Unknown data return type")

    def get_2d_motion_with_key(self, dataset_key, motion_key, view, rate=30):
        '''
        Helper function for getting test samples during inference
        (Full length, do not crop along time dimension)
        '''
        
        # Get the h5 reader
        with h5py.File(self.root, "r") as f:
            try:
                data = np.array(f[dataset_key][motion_key]['joints']).copy()
            except:
                raise ValueError("[Error] Can't read key (%s, %s) from h5 dataset" % (dataset_key, motion_key))
        data = np.transpose(data, (1, 2, 0))
        
        _, _, T = data.shape

        if  T > self.max_seq_length:
            start = (T-self.max_seq_length) // 2
            data = data[:, :, start: start+self.max_seq_length]
            T = self.max_seq_length
        else:
            T = ((T-1) // 16) * 16 + 1
            data = data[:, :, :T]
        
        decoder_mask = np.array([0]*data.shape[-1])

        data_3d = self._rotate_motion_3d(self._centralize_motion(data), view)
        data_2d = self._project_2D(data_3d)

        if self.return_type == '3D':
            gt = data_3d.copy()
            start = gt[0,:,0]
            data = data_3d.copy()
        else:
            gt = self._joints_to_openpose(data_2d)
            start = gt[8,:,0]
            data = data_2d.copy()

        encoder_mask, _ = self.generate_training_mask(decoder_mask, rate)

        input = self._normalize_to_network(data, self.evaluate_noise)

        interpolate = self._get_interpolate_motion(input.copy(), rate)

        input = input * ~encoder_mask.reshape(1,1,-1).astype(bool)

        return start, gt, \
               torch.from_numpy(interpolate).float().reshape(-1, interpolate.shape[-1]), \
               torch.from_numpy(input).float().reshape(-1, input.shape[-1]), \
               torch.from_numpy(encoder_mask).bool(), torch.from_numpy(decoder_mask).bool()

    def generate_training_mask(self, mask, sample_rate=30):
        '''
        Helper function for generating the encoder decoder trainig mask
        '''
        seq_len = mask.shape[-1] 
        assert (seq_len - 1) % sample_rate == 0

        sample_mask = np.ones( seq_len, dtype=np.int32)
        sample_mask [::sample_rate] = 0
        encoder_mask = np.bitwise_or( sample_mask, mask)

        sample_size = self.train_sample_size
        indices = np.random.randint ( 0, seq_len, sample_size)
        
        decoder_mask = encoder_mask.copy()
        decoder_mask[indices] = 0
        #decoder_mask = mask.copy()
        return encoder_mask, decoder_mask

    def get_openpose_data (self, json_dir, sample_rate=8):
        '''
        Helper function for transfering the openpose json dir to the network inputs
        '''
        motion, conf, (scale, offset) = openpose2motion (json_dir, scale=self.openpose_scale, offset=self.openpose_offset)

        decoder_mask = np.array([0]*motion.shape[-1])

        run = int(np.log2(sample_rate))

        interp_motion, interp_mask, inerp_conf= self._interpolate_frames(motion, decoder_mask, conf=conf, times=run)

        encoder_mask, _ = self.generate_training_mask(interp_mask, sample_rate)

        interp_motion = self._localize_motion(interp_motion)
        interp_motion = self._normalize_motion(interp_motion)
        interp_motion = interp_motion.reshape([-1, interp_motion.shape[-1]])

        input_motion = interp_motion.copy() * ~encoder_mask.reshape(1,-1).astype(bool)

        return (scale, offset, inerp_conf), \
               torch.from_numpy(input_motion).float(), \
               torch.from_numpy(interp_motion).float(), \
               torch.from_numpy(encoder_mask).bool(), \
               torch.from_numpy(interp_mask).bool()


    def _compute_mean_pose(self):

        mean = 0.0
        std = 0.0
        nsamples = 0
        all_joints = []

        for sample in tqdm(self.samples):
            dataset_key, motion_key = sample 

            motion = self._get_h5_motion (dataset_key, motion_key)

            view = None
            point_3d = self._rotate_motion_3d(self._centralize_motion(motion), view)

            point_2d = self._project_2D(point_3d)

            if self.return_type == '3D':
                data = point_3d
                data = self._localize_motion(data)

            else:
                data = point_2d
                data = self._joints_to_openpose(data)
                data = self._localize_motion(data)

            mean += np.mean(data, axis=2, dtype=np.float64)
            std += np.std(data, axis=2, dtype=np.float64)

        self.mean_pose = mean / len(self.samples)
        self.std_pose = std / len(self.samples)

        self.std_pose[np.where(self.std_pose == 0)] = 1e-9


    def _get_h5_motion(self, dataset_key, motion_key):

        # Get the h5 reader
        with h5py.File(self.root, "r") as f:
            try:
                data = np.array(f[dataset_key][motion_key]['joints']).copy()
            except:
                raise ValueError("[Error] Can't read key (%s, %s) from h5 dataset" % (dataset_key, motion_key))
        data = np.transpose(data, (1, 2, 0)) # Joints * Dimension * Time Length
        return data

    def _random_temproal_crop(self, data):

        seq_len = data.shape[2]

        if seq_len < self.max_seq_length:

            seq_len = ((seq_len-1) // self.rate) * self.rate + 1

            diff = self.max_seq_length - seq_len

            data_pad = np.pad(data[:,:,:seq_len], [(0,0), (0,0), (0, diff)], mode='constant' )
            mask = np.array([0]*seq_len + [1]*diff, dtype=np.int32)

            return data_pad, mask, 0
        else:
            start_idx = random.randint(0, seq_len - self.max_seq_length)
            mask = np.array([0]*self.max_seq_length, dtype=np.int32)

            data_crop = data[:,:, start_idx:start_idx+self.max_seq_length]

            return data_crop, mask, start_idx

    def get_change_of_basis(self, motion3d, angles=None):
        """
        Get the unit vectors for local rectangular coordinates for given 3D motion
        :param motion3d: numpy array. 3D motion from 3D joints positions, shape (nr_joints, 3, nr_frames).
        :param angles: tuple of length 3. Rotation angles around each axis.
        :return: numpy array. unit vectors for local rectangular coordinates's , shape (3, 3).
        """
        def rotate_basis(local3d, angles):
            """
            Rotate local rectangular coordinates from given view_angles.
            :param local3d: numpy array. Unit vectors for local rectangular coordinates's , shape (3, 3).
            :param angles: tuple of length 3. Rotation angles around each axis.
            :return:
            """
            #angles = np.array([0,0,angles])

            cx, cy, cz = np.cos(angles)
            sx, sy, sz = np.sin(angles)

            x = local3d[0]
            x_cpm = np.array([
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0]
            ], dtype='float')
            x = x.reshape(-1, 1)
            mat33_x = cx * np.eye(3) + sx * x_cpm + (1.0 - cx) * np.matmul(x, x.T)

            mat33_z = np.array([
                [cz, sz, 0],
                [-sz, cz, 0],
            [0, 0, 1]
            ], dtype='float')

            local3d = local3d @ mat33_x.T @ mat33_z
            return local3d
        
        # 2 RightArm 5 LeftArm 9 RightUpLeg 12 LeftUpLeg
        horizontal = (motion3d[17] - motion3d[16] + motion3d[2] - motion3d[1]) / 2
        horizontal = np.mean(horizontal, axis=1)
        horizontal = horizontal / np.linalg.norm(horizontal)
        local_z = np.array([0, 0, 1])
        local_y = np.cross(horizontal, local_z)  # bugs!!!, horizontal and local_Z may not be perpendicular
        local_y = local_y / np.linalg.norm(local_y)
        local_x = np.cross(local_y, local_z)
        local = np.stack([local_x, local_y, local_z], axis=0)

        if angles is not None:
            local = rotate_basis(local, angles)

        return local

    def _joints_to_openpose(self, data):
        body = np.zeros((19, data.shape[1], data.shape[2]))

        # Rearrange the SMPL body indices into openpose indices
        indices = np.array([15, 12, 17, 19, 21, 16, 18, 20, 0, 
                            2,  5,  8 , 1,  4,  7,  10, 11])
        body[:17] = data[indices]
        # averaged hand position
        body[17] = np.mean(data[22:37], axis=0)
        body[18] = np.mean(data[37:52], axis=0) 

        return body


    def _project_2D(self, point3d, info=None):

        focal = self.focal_len
        depth = self.cam_depth
        d_min = self.cam_depth * 0.1

        if self.projection_noise:
            focal += np.random.uniform(-d_min, d_min)
            depth += np.random.uniform(-d_min, d_min)

        if self.project == 'orthogonal':
            point2d = point3d[:, [0, 2], :]
            point2d[:, 1, :] = -point2d [:, 1, :]
        else:
            point2d = np.divide(focal * point3d[:, [0, 2], :], np.maximum((point3d[:, [1, 1], :] + depth), d_min))
            point2d[:, 1, :] = -point2d [:, 1, :]
        
        ''' 
        # Sanity check, for points that are too close to camera, we get very large value
        if np.min(point2d) > self.frame_boarder or np.min(point2d) < -self.frame_boarder:
            print(np.min(point2d))
            print(np.max(point2d))
            if info is not None:
                print (info)
        '''

        point2d  = np.clip(point2d, -self.frame_boarder, self.frame_boarder)

        return point2d

    def _interpolate_frames(self, data, mask, conf=None, times=1):
        """
        Perform linear interpolation between each frame, resulting 2*L-1 frames
        Use for the openpose input case (for inference)
        Input J*D*L -> J*D* 2L-1
        """
        def interpolate(data, mask, conf=None):
            length = data.shape[-1]

            tmp = np.zeros((data.shape[0], data.shape[1], length*2 -1))
            mid = (data[:,:,1:] + data[:,:,:-1] ) / 2
            tmp[:,:,::2] = data
            tmp[:,:,1::2] = mid

            new_conf = None
            if conf is not None:
                new_conf = np.zeros((conf.shape[0], conf.shape[1], length*2 -1))
                mid = (conf[:,:,1:] + conf[:,:,:-1] ) / 2
                new_conf[:,:,::2] = conf
                new_conf[:,:,1::2] = mid

            new_mask = np.zeros( length*2 -1, dtype=np.int32)
            new_mask[::2] = mask
            new_mask[1::2] = mask[1:]

            return tmp, new_mask, new_conf

        new_mask = mask.copy()
        new_data = data.copy()
        new_conf = conf.copy()

        for i in range(times):
            new_data, new_mask, new_conf = interpolate(new_data, new_mask, new_conf)

        return new_data, new_mask, new_conf

    def _get_interpolate_motion(self, motion, rate, mode='linear'):
        '''
        input J*D*L motion -> J*D*L interpolated motion (interpolated by key frame)
        '''
        if mode == 'linear':
            motion = motion.copy()
            seq_len = motion.shape[-1]
            indice = np.arange(seq_len, dtype=int)
            chunk = indice // rate
            remain = indice % rate

            prev = motion[:,:,chunk * rate]

            next = np.concatenate( [ motion[:,:,(chunk[:-1]+1) * rate], motion[:,:,-1,np.newaxis]], axis=-1)

            interpolate = ( prev / rate * (rate-remain) ) + (next / rate * remain)
        
        else: # Quadratic
            motion = motion.copy()
            seq_len = motion.shape[-1]
            indice = np.arange(seq_len, dtype=int)
            chunk = indice // rate
            remain = indice % rate

            prev = np.concatenate( [ -1 * motion[:,:,(chunk[:rate+1]+1) * rate], motion[:,:,(chunk[(rate+1):]-1) * rate]], axis=-1)

            this = motion[:,:,chunk * rate]

            next = np.concatenate( [ motion[:,:,(chunk[:-1]+1) * rate], motion[:,:,-1,np.newaxis]], axis=-1)

            t = remain / rate
            interpolate = this + ( (next-this) + (prev-this) ) / 2 * (t**2) + ( (next-this) - (prev-this) ) / 2 * t

        return interpolate

    def _rotate_motion_3d(self, motion3d, angles=None):

        basis = self.get_change_of_basis(motion3d, angles)

        motion3d = basis @ motion3d

        return motion3d
    
    def _centralize_motion(self, data):
        """
        centralize motion to the center of mass among time
        """
        length = data.shape[-1]
        centers = np.mean(data[0, :, :], axis=1, keepdims=True) # N_dim x 1
        data = data - np.repeat(centers, length, axis=1)
        return data

    def _localize_motion(self, motion):
        """
        Motion fed into our network is the local motion, i.e. coordinates relative to the hip joint.
        This function removes global motion of the hip joint, and instead represents global motion with velocity
        Use 3D SMPL joints indices
        """

        D = motion.shape[1]

        if self.return_type == '3D':
            root_idx = 0
        else:
            root_idx = 8
        # subtract centers to local coordinates
        centers = motion[root_idx, :, :] # N_dim x T
        motion = motion - centers

        # adding velocity
        '''
        translation = centers[:, 1:] - centers[:, :-1]
        velocity = np.c_[np.zeros((D, 1)), translation]
        velocity = velocity.reshape(1, D, -1)
        '''
        velocity = centers [np.newaxis, :, :]
        if self.return_type == '3D':
            motion = np.r_[motion[1:], velocity]
        else:
            motion = np.r_[motion[:8], motion[9:], velocity]

        # motion_proj = np.r_[motion_proj[:8], motion_proj[9:]]

        return motion

    def _normalize_motion(self, motion):

        return (motion - self.mean_pose[:, :, np.newaxis]) / self.std_pose[:, :, np.newaxis]

    def _scale_to_rendering(self, data):

        data = self._centralize_motion (data)

        data = data * self.scale_unit + 300

        return data

    def _normalize_to_network(self, data, random_drop=False):

        if not self.return_type == '3D':
            data = self._joints_to_openpose(data)

        if random_drop:
            data = self._random_drop(data)
    
        data = self._localize_motion(data)
        data_norm = self._normalize_motion(data)

        return data_norm

    def _random_drop(self, data):

        J, D, L = data.shape


        if self.phase == 'train':
            idx_t = np.arange(0, L, self.rate)
            noise_frame = np.random.choice(idx_t, self.noise_rate, replace=False)
            drop_frame = np.random.choice(idx_t, self.joint_drop_rate, replace=False)
            flip_frame = np.random.choice(idx_t, self.flip_rate, replace=False)
        else:
            idx_t = np.arange(0, L, self.eval_rate)
            noise_frame = np.random.choice(idx_t, 4, replace=False)
            drop_frame = np.random.choice(idx_t, 4, replace=False)
            flip_frame = np.random.choice(idx_t, 4, replace=False)       


        ## Apply noise ##
        noise = np.random.rand(J,D,L) * self.noise_weight
        noise_idx =  np.array([3, 4, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18])
        noise_idx =  np.random.choice(noise_idx, 5, replace=False).reshape(-1,1,1)
        data [noise_idx, :, noise_frame] = data [noise_idx, :, noise_frame] + noise [noise_idx, :, noise_frame]

        ## Apply random drop ##
        drop_idx =  np.array([0, 3, 4, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18])
        drop_idx =  np.random.choice(drop_idx, 3, replace=False).reshape(-1,1,1)
        data [drop_idx, :, drop_frame] = 0.0

        ## Apply random flip ##

        right_leg = np.array([9, 10, 11, 16]).reshape(-1,1,1)
        left_leg = np.array([12, 13, 14, 15]).reshape(-1,1,1)


        tmp_right = data [right_leg, :, flip_frame]
        data [right_leg, :, flip_frame] = data [left_leg, :, flip_frame]
        data [left_leg, :, flip_frame] = tmp_right

        return data

