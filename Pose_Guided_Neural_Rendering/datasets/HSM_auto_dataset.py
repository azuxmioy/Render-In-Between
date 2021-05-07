from datasets.base_dataset import BaseDataset
from utils.keypoint2img import *
import numpy as np
import torch
import h5py
import random
import os
import io
import cv2
from PIL import Image, ImageFilter
from tqdm import tqdm
from scipy import ndimage
from torchvision import transforms
import albumentations as A
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def __crop(img, size, pos):
    im_h = img.height
    im_w = img.width

    th, tw = size
    x1, y1 = pos    

    return img.crop((x1, y1, x1 + tw , y1 + th))

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def get_transform(loadsize, modelsize, crop_pos, flip, method=Image.BICUBIC, random_flip=True, normalize=True, toTensor=True):

    transform_list = []
    ### resize input image
    transform_list.append( transforms.Resize(loadsize, interpolation=method) )

    transform_list.append( transforms.Lambda(lambda img: __crop(img, modelsize, crop_pos ) ))

    ### random flip
    if random_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, flip)))
    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_alb_transform(loadsize, modelsize, crop_pos=None, method=cv2.INTER_CUBIC, param=None):
    
    transform_list = []
    ### resize input image
    if crop_pos is not None:
        transform_list.append( A.Resize(height=loadsize[0], width=loadsize[1], interpolation=method, always_apply=True) )
        if param is not None:
            transform_list.append( A.ShiftScaleRotate (shift_limit=[param['shift'], param['shift']],
                                                       scale_limit=[param['scale'], param['scale']],
                                                       rotate_limit=[param['angle'], param['angle']],
                                                       interpolation=method,
                                                       border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True))
        #transform_list.append( A.Crop (x_min=crop_pos[0], y_min=crop_pos[1],
        #                           x_max=crop_pos[0]+modelsize[1],
        #                           y_max=crop_pos[1]+modelsize[0], always_apply=True) )
    else:
        transform_list.append( A.Resize(height=modelsize[0], width=modelsize[1], interpolation=method, always_apply=True) )

    return A.Compose(transform_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

class HSMAutoDataset(BaseDataset):

    def __init__(self, cfg, root, phase='train'):


        BaseDataset.__init__(self, cfg, root)

        print('############################')
        print('     Preparing Dataset      ')
        print('############################')

        self.train_vid_list = cfg.train_video_list
        self.test_vid_list = cfg.test_video_list

        self.load_width = cfg.load_width
        self.load_height = cfg.load_height

        self.width = cfg.model_width       # this is network size
        self.height = cfg.model_height     # this is network size

        self.sampling_mode = cfg.sampling_mode
        self.random_blur_rate = cfg.random_blur_rate
        self.random_drop_prob = cfg.random_drop_prob
        self.skeleton_thres = cfg.skeleton_thres
        self.foot_thres = cfg.foot_thres
        self.gauss_sigma = cfg.gauss_sigma
        self.phase = phase


        # # of frames per training batch
        self.max_frames = cfg.max_frames



        self.to_tensor_norm = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))])

        self.to_tensor = transforms.Compose([transforms.ToTensor()])
      
        self.videl_list = self.train_vid_list if self.phase == 'train' else self.test_vid_list


        self.n_frames = {}
        self.samples = []

        for vid_name in self.videl_list:
            with h5py.File(self.root, "r") as f:
                if self.phase == 'train':
                    sample_list = list(f[vid_name]['train_fake'])
                else:
                    sample_list = list(f[vid_name]['gt_images'])
                self.n_frames[vid_name] = len(sample_list)
                tuple_list = [(vid_name, list(range(idx, idx+self.max_frames)), 'f') for idx in range(len(sample_list)+2-self.max_frames)]
                #tuple_list += [(vid_name, list(range(idx, idx-self.max_frames, -1)), 'b') for idx in range(len(sample_list)+1, self.max_frames-1, -1)]

                self.samples.extend(tuple_list)

    def get_max_frames(self):
        return self.max_frames

    def update_max_frame(self, new_max_frame):
        self.max_frames = new_max_frame

        self.n_frames = {}
        self.samples = []

        for vid_name in self.videl_list:
            with h5py.File(self.root, "r") as f:
                if self.phase == 'train':
                    sample_list = list(f[vid_name]['train_fake'])
                else:
                    sample_list = list(f[vid_name]['gt_images'])
                self.n_frames[vid_name] = len(sample_list)
                tuple_list = [(vid_name, list(range(idx, idx+self.max_frames)), 'f') for idx in range(len(sample_list)+2-self.max_frames)]
                #tuple_list += [(vid_name, list(range(idx, idx-self.max_frames, -1)), 'b') for idx in range(len(sample_list)+1, self.max_frames-1, -1)]

                self.samples.extend(tuple_list)


    def get_testdata_with_key(self, video_key, frame_index):

        with h5py.File(self.root, "r") as f:
            try:
                img = np.array(f[video_key]['gt_images'][frame_index]).copy()
                cain = np.array(f[video_key]['gt_cain'][frame_index]).copy()
                pose = np.array(f[video_key]['gt_poses'][frame_index]).copy()
            except:
                raise ValueError("[Error] Can't read key (%s, %s) from h5 dataset" % (video_key, frame_index))


        transform_i = get_alb_transform(loadsize = (self.load_height, self.load_width), modelsize = (self.height, self.width))

        img = np.asarray(Image.open(io.BytesIO(img)))
        cain= np.asarray(Image.open(io.BytesIO(cain)))

        joint_lists = [(pose[i,0], pose[i,1]) for i in range(pose.shape[0])]
        joint_conf = [ pose[i,2] for i in range(pose.shape[0])]

        transformed_image = transform_i(image=img, keypoints=joint_lists) ['image']
        transformed_landmark = transform_i(image=img, keypoints=joint_lists) ['keypoints']
        transformed_cain = transform_i(image=cain, keypoints=joint_lists) ['image']


        posemap = self._generate_pose_map(transformed_landmark, joint_conf, self.height, self.width  )
        skeleton = self._generate_skeleton(transformed_landmark, joint_conf, self.height, self.width  )
        mask, _ = self._generate_human_mask(transformed_landmark, joint_conf, self.height, self.width  )


        return{
            'img' : self.to_tensor_norm (transformed_image).unsqueeze(0),
            'cain' : self.to_tensor_norm (transformed_cain).unsqueeze(0),
            'pose' : torch.from_numpy(posemap).float().unsqueeze(0),
            'skel' : self.to_tensor_norm (skeleton).unsqueeze(0),
            'mask' : self.to_tensor(mask).squeeze().unsqueeze(0)
        }



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        # decide transform params
        x_pos = np.random.randint(20, 120)
        y_pos = np.random.randint(10, 30)
        #x_pos = np.random.randint(50, 300)
        #y_pos = np.random.randint(0, 50)
        param = dict()
        param['shift'] = np.random.random_sample() * 0.125 - 0.0625
        param['angle'] = np.random.random_sample() * 20 - 10
        param['scale'] = np.random.random_sample() * 0.2 - 0.1

        transform_i = get_alb_transform(  loadsize = (self.load_height, self.load_width),
                                          modelsize = (self.height, self.width),
                                          crop_pos = (x_pos, y_pos),
                                          param=param)
        # start reading data
        video_key, frame_list, side = self.samples[index]

        img_list = []
        pose_list = []
        skel_list = []
        mask_list = []
        fake_list = []

        for i, frame_idx in enumerate(frame_list):
            data = self._get_h5_data(video_key, frame_idx)
            img = np.asarray(Image.open(io.BytesIO(data['img'])))
            landmark = data['pose']

            joint_lists = [(landmark[i,0], landmark[i,1]) for i in range(landmark.shape[0])]
            joint_conf = [ landmark[i,2] for i in range(landmark.shape[0])]

            transformed = transform_i(image=img, keypoints=joint_lists)
            transformed_image = transformed['image']
            transformed_landmark = transformed['keypoints']


            posemap = self._generate_pose_map(transformed_landmark, joint_conf, self.height, self.width )
            skeleton = self._generate_skeleton(transformed_landmark, joint_conf, self.height, self.width )
            mask, part_mask = self._generate_human_mask(transformed_landmark, joint_conf, self.height, self.width )

            img_list.append(self.to_tensor_norm(transformed_image))
            pose_list.append(torch.from_numpy(posemap).float())
            skel_list.append(self.to_tensor_norm(skeleton))
            mask_list.append(self.to_tensor(mask).float())

            if i == 0:
                fake_list.append(torch.zeros(img_list[0].shape))
            else:
                fake_data = self._get_fake_data(video_key, frame_idx)
                fake = Image.open(io.BytesIO(fake_data['fake']))
                fake_np = np.asarray(fake)
                blur = np.asarray(fake.filter(ImageFilter.GaussianBlur(radius = 10)))

                transformed_fake = self.to_tensor_norm(transform_i(image=fake_np, keypoints=joint_lists) ['image'])
                transformed_blur = self.to_tensor_norm(transform_i(image=blur, keypoints=joint_lists) ['image'])

                if self.phase == 'train':
                    blur_mask = self.to_tensor(part_mask).repeat(3,1,1).float()
                    transformed_fake = transformed_blur * blur_mask + transformed_fake * (1 - blur_mask)

                fake_list.append(transformed_fake)


        return{
                # L * C * H * W
                'img' : torch.stack(img_list, dim=0),
                'pose' : torch.stack(pose_list, dim=0),
                'skel' : torch.stack(skel_list, dim=0),
                'mask' : torch.stack(mask_list, dim=0).squeeze(),
                'fake' : torch.stack(fake_list, dim=0)
            }


    def _get_h5_data(self, video_key, frame_index):

        # Get the h5 reader
        data = {}

        with h5py.File(self.root, "r") as f:
            try:
                data['img'] = np.array(f[video_key]['train_images'][frame_index]).copy()
                data['pose'] = np.array(f[video_key]['train_poses'][frame_index]).copy()

            except:
                raise ValueError("[Error] Can't read key (%s, %s) from h5 dataset" % (video_key, frame_index))
        return data

    def _get_fake_data(self, video_key, frame_index):

        # Get the h5 reader
        data = {}

        with h5py.File(self.root, "r") as f:
            try:
                data['fake'] = np.array(f[video_key]['train_fake'][frame_index-1]).copy()

            except:
                raise ValueError("[Error] Can't read key (%s, %s) from h5 dataset" % (video_key, frame_index))

        return data

    def _select_src_sample(self, vid_name, idx):

        if self.sampling_mode == 'random':
            a = np.arange(self.n_frames[vid_name]+2)
            a = np.delete(a, idx+1)
            src_idx = np.random.choice(a)
            assert(src_idx < self.n_frames[vid_name]+2 )
            return int(src_idx)

        elif self.sampling_mode == 'neighbor':
            src_idx = np.random.choice([idx, idx+2])
            assert(src_idx < self.n_frames[vid_name]+2 )
            return int(src_idx)
        else: 
            raise NotImplementedError("Unknown sampling type!")


    def _generate_pose_map(self, landmark, conf, height, width):
        
        maps = []
        n_landmark = len(landmark)

        gauss_sigma = np.random.randint(self.gauss_sigma-1, self.gauss_sigma+1, size=n_landmark)

        for i in range(n_landmark):

            map = np.zeros([height, width], dtype=np.float)

            x = landmark[i][0]
            y = landmark[i][1]
            c = conf[i]
            if self.phase == 'train':
                if x>=0 and y>=0 and c>self.skeleton_thres and x<width and y<height and (np.random.rand() > self.random_drop_prob):
                    map[int(y), int(x)]=1
                    map = ndimage.filters.gaussian_filter(map, sigma = gauss_sigma[i])
                    map = map/map.max()
            else: 
                if x>=0 and y>=0 and c>self.skeleton_thres and x<width and y<height:
                    map[int(y), int(x)]=1
                    map = ndimage.filters.gaussian_filter(map, sigma = self.gauss_sigma)
                    map = map/map.max()

            maps.append(map)

        maps = np.stack(maps, axis=0)
        return maps

    def _generate_skeleton(self, landmark, conf, height, width):

        pose_img = np.zeros((height, width, 3), np.uint8)
        size = (width, height) 

        edge_lists = define_edge_lists(p=len(landmark))
        pts = extract_valid_keypoints(landmark, conf, size, thres1=self.skeleton_thres, thres2=self.foot_thres)
        drop_prob = self.random_drop_prob if self.phase == 'train' else 0.0 
        pose_img = connect_keypoints(pts, edge_lists, size, drop_prob, pose_img)

        return pose_img


    def _generate_human_mask(self, landmark, conf, height, width):

        dict_pose_line = {}
        n_landmark = len(landmark)

        DICT_POSE = {}
        DICT_POSE['head'] = [
            (0, 1)]
        DICT_POSE['hand'] = [
            (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)]
        DICT_POSE['legs'] = [
            (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]  
        DICT_POSE['body'] = [
            (1, 8), (2, 9), (5, 12)] 

        if n_landmark == 19:
            DICT_POSE['hand'] += [(4, 18), (7, 17)]
            DICT_POSE['legs'] += [(11, 16), (14, 15)]


        mask = np.zeros((height, width, 3), dtype=np.uint8)
        part_mask = np.zeros((height, width, 3), dtype=np.uint8)

        binary = np.zeros((height, width))
        part_binary = np.zeros((height, width))

        for i in range(n_landmark):
            x = landmark[i][0]
            y = landmark[i][1]
            c = conf[i]

            if x>=0 and y>=0 and c>self.skeleton_thres and x<width and y<height:
                dict_pose_line[i] = (int(x), int(y))
                radius = 15
                if (i==0): radius = 30
                cv2.circle(mask, (int(x), int(y)), radius=radius, color=(255,255,255), thickness=-1)
        
        # Head mask:
        for start_p, end_p in DICT_POSE['head']:
            if start_p in dict_pose_line and end_p in dict_pose_line:
                start_p = dict_pose_line[start_p]
                end_p = dict_pose_line[end_p]
                cv2.line(mask, start_p, end_p, color=(255,255,255), thickness=30)
                if np.random.rand() < self.random_blur_rate:
                    cv2.line(part_mask, start_p, end_p, color=(255,255,255), thickness=30)
        # Hand mask:
        for start_p, end_p in DICT_POSE['hand']:
            if start_p in dict_pose_line and end_p in dict_pose_line:
                start_p = dict_pose_line[start_p]
                end_p = dict_pose_line[end_p]
                cv2.line(mask, start_p, end_p, color=(255,255,255), thickness=30)
                if np.random.rand() < self.random_blur_rate:
                    cv2.line(part_mask, start_p, end_p, color=(255,255,255), thickness=30)

        # Leg mask:
        for start_p, end_p in DICT_POSE['legs']:
            if start_p in dict_pose_line and end_p in dict_pose_line:
                start_p = dict_pose_line[start_p]
                end_p = dict_pose_line[end_p]
                cv2.line(mask, start_p, end_p, color=(255,255,255), thickness=30)
                if np.random.rand() < self.random_blur_rate:
                    cv2.line(part_mask, start_p, end_p, color=(255,255,255), thickness=30)
        # Body mask:
        for start_p, end_p in DICT_POSE['body']:
            if start_p in dict_pose_line and end_p in dict_pose_line:
                start_p = dict_pose_line[start_p]
                end_p = dict_pose_line[end_p]
                cv2.line(mask, start_p, end_p, color=(255,255,255), thickness=40)
                if np.random.rand() < self.random_blur_rate:
                    cv2.line(part_mask, start_p, end_p, color=(255,255,255), thickness=40)


        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        binary[mask>128] = 1
        part_mask = cv2.cvtColor(part_mask, cv2.COLOR_BGR2GRAY)
        part_binary[part_mask>128] = 1
        return binary.astype(np.bool), part_binary.astype(np.bool)


