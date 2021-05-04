import sys, os
import torch
import h5py
import numpy as np

from human_body_prior.tools.omni_tools import copy2cpu as c2c

from human_body_prior.body_model.body_model import BodyModel

comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bm_path = 'body_models/smplh/male/model.npz'
dmpl_path = 'body_models/dmpls/male/model.npz'

num_betas = 10 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

gdr2num = {'male':-1, 'neutral':0, 'female':1}
gdr2num_rev = {v:k for k,v in gdr2num.items()}

expr_code = 'VXX_SVXX_TXX' #VERSION_SUBVERSION_TRY

msg = ''' Initial use of standard AMASS dataset preparation pipeline '''

amass_dir = './data'

amass_splits = {
    'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'DFaust_67',
              'BMLhandball', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'BioMotionLab_NTroje', 'ACCAD']
}
amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['vald'])))


outfile = h5py.File('AMASS_3D_joints.h5', 'w')


for split_name, datasets in amass_splits.items():

    for dataset_name in datasets:

        model_lists = sorted([o for o in os.listdir(os.path.join(amass_dir, dataset_name)) if os.path.isdir(os.path.join(amass_dir, dataset_name, o))])

        print(dataset_name)
        sub_group = outfile.create_group(dataset_name)
        motion_group = {}

        for model in model_lists:

            motion_path = os.path.join(amass_dir, dataset_name, model)
            motion_lists = sorted([o for o in os.listdir(motion_path) if o.endswith('.npz')])

            for motion in motion_lists:
                npz_fname = os.path.join(motion_path, motion)
                try:
                    cdata = np.load(npz_fname)
                except:
                    logger('Could not read %s! skipping..'%npz_fname)
                    continue
                if 'poses' not in list(cdata):
                    continue

                N = len(cdata['poses'])
                
                # Adjust the lenght here for joints you want to keep
                cdata_ids = range(int(0.1*N), int(0.9*N),1)
                n_frames = len(cdata_ids) 
                if n_frames<1: continue

                motion_name = '{}_{}'.format(str(model), str(motion))

                if "/{}/{}".format(str(dataset_name), motion_name) not in outfile:
                    motion_group[motion_name] = sub_group.create_group(motion_name)

                bm = BodyModel(bm_path=bm_path, num_betas=num_betas, num_dmpls=num_dmpls, path_dmpl=dmpl_path, batch_size=n_frames).to(comp_device)

                root_orient = torch.Tensor(cdata['poses'][cdata_ids, :3]).to(comp_device)  # controls the global root orientation (L * 3)
                pose_body =  torch.Tensor(cdata['poses'][cdata_ids, 3:66]).to(comp_device)  # controls the body (L * 21 * 3)
                pose_hand = torch.Tensor(cdata['poses'][cdata_ids, 66:]).to(comp_device)  # controls the finger articulation (L * 21 * 3)
                betas = torch.Tensor(np.repeat(cdata['betas'][:10][np.newaxis].astype(np.float64), repeats=n_frames, axis=0)).to(comp_device) # (L * 10)
                dmpls = torch.Tensor(cdata['dmpls'][cdata_ids]).to(comp_device) 
                trans = torch.Tensor(cdata['trans'][cdata_ids]).to(comp_device) 

                body = bm(pose_body=pose_body, pose_hand = pose_hand, betas=betas, dmpls=dmpls, root_orient=root_orient, trans=trans)
                joints = body.Jtr

                motion_group[motion_name].create_dataset('joints',
                        data=c2c(joints), dtype=np.float64)
                '''
                motion_group[motion_name].create_dataset('root_orient',
                        data=c2c(root_orient), dtype=np.float64)
                motion_group[motion_name].create_dataset('pose_body',
                        data=c2c(pose_body), dtype=np.float64)
                motion_group[motion_name].create_dataset('pose_hand',
                        data=c2c(pose_hand), dtype=np.float64)
                motion_group[motion_name].create_dataset('dmpls',
                        data=c2c(dmpls),dtype=np.float64)
                motion_group[motion_name].create_dataset('trans',
                        data=c2c(trans),dtype=np.float64)
                motion_group[motion_name].create_dataset('betas',
                        data=c2c(betas),dtype=np.float64)
                motion_group[motion_name].create_dataset('gender',
                        data=cdata['gender'].astype('<S10'),dtype='<S10')
                motion_group[motion_name].create_dataset('dataset_name',
                        data=np.string_(dataset_name).astype('<S'),dtype='<S30')
                '''

outfile.close()



