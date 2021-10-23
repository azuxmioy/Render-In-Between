from datasets.base_dataset import BaseDataset
from utils.keypoint2img import *
import numpy as np
import torch
import h5py
import io
import cv2
from PIL import Image, ImageFilter
from tqdm import tqdm
from scipy import ndimage
from torchvision import transforms
import albumentations as A
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

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
        if modelsize[0] < loadsize[0] and modelsize[1] < loadsize[1]:

            transform_list.append( A.Crop (x_min=crop_pos[0], y_min=crop_pos[1],
                                  x_max=crop_pos[0]+modelsize[1],
                                  y_max=crop_pos[1]+modelsize[0], always_apply=True) )
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

        self.load_width = cfg.load_width     # this is data preprocessing size
        self.load_height = cfg.load_height

        self.width = cfg.model_width         # this is network input size
        self.height = cfg.model_height     

        # Skeletal image preprocessing setting
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
      
        self.video_list = self.train_vid_list if self.phase == 'train' else self.test_vid_list


        self.n_frames = {}
        self.samples = []

        try: 
            for vid_name in self.video_list:
                with h5py.File(self.root, "r") as f:
                    if self.phase == 'train':
                        sample_list = list(f[vid_name]['train_dain'])
                    else:
                        sample_list = list(f[vid_name]['gt_images'])
                    self.n_frames[vid_name] = len(sample_list)
                    tuple_list = [(vid_name, list(range(idx, idx+self.max_frames)), 'f') for idx in range(len(sample_list)+2-self.max_frames)]

                    self.samples.extend(tuple_list)

        except: # do not read h5 file when inference only
            pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        # decide transform params
        x_pos = np.random.randint(0, max(self.load_width-self.width,1))
        y_pos = np.random.randint(0, max(self.load_height-self.height,1))
        param = dict()
        param['shift'] = np.random.random_sample() * 0.125 - 0.0625
        param['angle'] = np.random.random_sample() * 20 - 10
        param['scale'] = np.random.random_sample() * 0.2 - 0.1

        transform_i = get_alb_transform(  loadsize = (self.load_height, self.load_width),
                                          modelsize = (self.height, self.width),
                                          crop_pos = (x_pos, y_pos),
                                          param=param)
        # start reading data
        video_key, frame_list, _ = self.samples[index]

        img_list = []
        pose_list = []
        skel_list = []
        mask_list = []
        back_list = []

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
                back_list.append(torch.zeros(img_list[0].shape))
            else:
                back_data = self._get_back_data(video_key, frame_idx)
                back = Image.open(io.BytesIO(back_data['dain']))
                back_np = np.asarray(back)
                blur = np.asarray(back.filter(ImageFilter.GaussianBlur(radius = 10)))

                transformed_back = self.to_tensor_norm(transform_i(image=back_np, keypoints=joint_lists) ['image'])
                transformed_blur = self.to_tensor_norm(transform_i(image=blur, keypoints=joint_lists) ['image'])

                if self.phase == 'train':
                    blur_mask = self.to_tensor(part_mask).repeat(3,1,1).float()
                    transformed_back = transformed_blur * blur_mask + transformed_back * (1 - blur_mask)

                back_list.append(transformed_back)


        return{
                # L * C * H * W
                'img' : torch.stack(img_list, dim=0),
                'pose' : torch.stack(pose_list, dim=0),
                'skel' : torch.stack(skel_list, dim=0),
                'mask' : torch.stack(mask_list, dim=0).squeeze(),
                'back' : torch.stack(back_list, dim=0)
            }

    def _get_h5_data(self, video_key, frame_index):
        '''
        # return image and its corresponding pose
        '''
        data = {}

        with h5py.File(self.root, "r") as f:
            try:
                data['img'] = np.array(f[video_key]['train_images'][frame_index]).copy()
                data['pose'] = np.array(f[video_key]['train_poses'][frame_index]).copy()

            except:
                raise ValueError("[Error] Can't read key (%s, %s) from h5 dataset" % (video_key, frame_index))
        return data

    def _get_back_data(self, video_key, frame_index):
        '''
        # return image's background image, the index is differed by 1
        '''
        data = {}

        with h5py.File(self.root, "r") as f:
            try:
                data['dain'] = np.array(f[video_key]['train_dain'][frame_index-1]).copy()

            except:
                raise ValueError("[Error] Can't read key (%s, %s) from h5 dataset" % (video_key, frame_index))

        return data

    def _generate_pose_map(self, landmark, conf, height, width):
        '''
        generate a 19 channel pose map where each joints is represented by random gaussion peak
        return nparray of 19 * H * W
        '''
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
        '''
        generate a 3 channel skeletal map where each limbs is represented by an unique color
        return nparray of H * W * 3
        '''
        pose_img = np.zeros((height, width, 3), np.uint8)
        size = (width, height) 

        edge_lists = define_edge_lists(p=len(landmark))
        pts = extract_valid_keypoints(landmark, conf, size, thres1=self.skeleton_thres, thres2=self.foot_thres)
        drop_prob = self.random_drop_prob if self.phase == 'train' else 0.0 
        pose_img = connect_keypoints(pts, edge_lists, size, drop_prob, pose_img)

        return pose_img


    def _generate_human_mask(self, landmark, conf, height, width):
        '''
        generate two binary human-centric dilation mask that connect each limbs with thick lines
        one is use for mask regularization and one use for generate random blur
        return nparray of H * W 
        '''
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

    def get_max_frames(self):
        return self.max_frames

    def update_max_frame(self, new_max_frame):
        '''
        Replace the original sample lists by a new one where
        each sample contains new_max_frame of frames
        '''
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

                self.samples.extend(tuple_list)


    def get_testdata_with_key(self, video_key, frame_index):
        '''
        use in the inference mode, return as batch size of 1
        return tensor of 1 * L * D * H * W
        '''
        with h5py.File(self.root, "r") as f:
            try:
                img = np.array(f[video_key]['gt_images'][frame_index]).copy()
                dain = np.array(f[video_key]['gt_dain'][frame_index]).copy()
                pose = np.array(f[video_key]['gt_poses'][frame_index]).copy()
            except:
                raise ValueError("[Error] Can't read key (%s, %s) from h5 dataset" % (video_key, frame_index))


        transform_i = get_alb_transform(loadsize = (self.load_height, self.load_width), modelsize = (self.height, self.width))

        img = np.asarray(Image.open(io.BytesIO(img)))
        dain= np.asarray(Image.open(io.BytesIO(dain)))

        joint_lists = [(pose[i,0], pose[i,1]) for i in range(pose.shape[0])]
        joint_conf = [ pose[i,2] for i in range(pose.shape[0])]

        transformed_image = transform_i(image=img, keypoints=joint_lists) ['image']
        transformed_landmark = transform_i(image=img, keypoints=joint_lists) ['keypoints']
        transformed_dain = transform_i(image=dain, keypoints=joint_lists) ['image']


        posemap = self._generate_pose_map(transformed_landmark, joint_conf, self.height, self.width  )
        skeleton = self._generate_skeleton(transformed_landmark, joint_conf, self.height, self.width  )
        mask, _ = self._generate_human_mask(transformed_landmark, joint_conf, self.height, self.width  )


        return{
            'img' : self.to_tensor_norm (transformed_image).unsqueeze(0),
            'dain' : self.to_tensor_norm (transformed_dain).unsqueeze(0),
            'pose' : torch.from_numpy(posemap).float().unsqueeze(0),
            'skel' : self.to_tensor_norm (skeleton).unsqueeze(0),
            'mask' : self.to_tensor(mask).squeeze().unsqueeze(0)
        }

