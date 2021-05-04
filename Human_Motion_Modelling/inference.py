import os, sys
import time
import numpy as np
import torch
import random
import argparse
import copy
import shutil

from datasets import find_dataset_using_name
from utils.utils import *
from visualize.util import motion2video, hex2rgb
from models.trainer import MotInterp_Trainer
from models.evaluator import Evaluator

from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
from visualize.record_summary import record_scalar_summaries
from visualize.util import print_evaluation, motion2gif

class DanceTrans_inference(torch.nn.Module):
    def __init__(self, enc, transformer):
        super(DanceTrans_inference, self).__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.pos_encode = enc.to(self.device)
        self.transformer = transformer.to(self.device)
        #self.seq2seq = seq2seq.to(self.device)


    def inference(self, data, interp, encoder_mask, decoder_mask, rate):

        self.pos_encode.eval()
        self.transformer.eval()

        # data: N * C * L

        input = torch.unsqueeze(data, dim=0).to(self.device)
        #input = torch.unsqueeze(interp, dim=0).to(self.device)

        #target = torch.unsqueeze(data, dim=0).to(self.device)
        target = torch.unsqueeze(interp, dim=0).to(self.device)

        #target = data.clone().to(self.device)

        #target = torch.zeros_like(input).to(self.device)
        #target = torch.randn(data.shape).to(self.device)

        src_mask = torch.unsqueeze(encoder_mask, dim=0).to(self.device)
        tgt_mask = torch.unsqueeze(decoder_mask, dim=0).to(self.device)

        pos_src = self.pos_encode(src_mask).to(self.device)
        pos_tar = self.pos_encode(tgt_mask).to(self.device)

        pred, _ = self.transformer.forward(
                input, src_mask, pos_src, target, tgt_mask, pos_tar, rate)

        pred = pred.permute(1, 2, 0)

        return pred

def main(opts):

    # import ipdb; ipdb.set_trace()
    # Set random seed.
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    #cudnn.benchmark = True


    # Load experiment setting
    config = get_config(opts.config)

    print('------------ Options -------------')
    for k, v in sorted(config.items()):
         print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')



    if not opts.resume:
        output_directory = os.path.join(opts.save_root, opts.name)
        train_writer = SummaryWriter(output_directory +'/logs/')
        checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
        if os.path.exists(os.path.join(output_directory,'code.zip')):
            os.remove(os.path.join(output_directory,'code.zip'))
        create_zip_code_files( os.path.join(output_directory,'code.zip'))
        shutil.copyfile(opts.config, os.path.join(output_directory, 'config.yaml'))
    else:
        output_directory = os.path.join(opts.save_root, opts.name)
        train_writer = SummaryWriter(output_directory +'/logs/')
        checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

    
    config.out_dir = checkpoint_directory
    config.eval_dir = image_directory

    dataset_class = find_dataset_using_name('AMASS')
    dataset = dataset_class(config, config.h5_file, return_type=config.return_type, phase='train')

    loader = DataLoader(dataset=dataset, 
                        batch_size=opts.batch_size, 
                        shuffle=True, 
                        num_workers=opts.workers,
                        pin_memory=True)

    trainer = MotInterp_Trainer(config, opts.resume)
    evaluator = Evaluator(config)
    #evaluator.visualize_skeleton(os.path.join('checkpoints/pose', 'test_220.out'), os.path.join(output_directory,'gif'))

    
    if opts.resume:
        evaluator.set_model(DanceTrans_inference(trainer.pos_encode, trainer.transformer))
        #evaluator.infer_h5_file(DanceTrans_inference(trainer.pos_encode, trainer.transformer), os.path.join(image_directory,'resume_999.out'))
        #print_evaluation(evaluator.evaluate_from_h5(os.path.join(image_directory,'resume_999.out')), 999) 
        #evaluator.visualize_skeleton(os.path.join(image_directory,'resume_999.out'), os.path.join(image_directory,'gif'))
        
        pose_dir = '/root/data/HSM_full/test2/train_poses'
        save_dir = '/root/data/HSM_full/test2/'
        subfolderlist = [f for f in sorted(os.listdir(pose_dir)) if os.path.isdir(os.path.join(pose_dir, f)) ]
        
        for i, subfolder in enumerate(subfolderlist):
            pose_path = os.path.join(pose_dir, subfolder)
            save_path = {
                'pred_dir' : os.path.join(save_dir, 'Predict_8', subfolder),
                'linear_dir' : os.path.join(save_dir, 'Linear_8', subfolder) }
            evaluator.interpolate_openpose(pose_path, sample_rate=8, save_dir=save_path)
    


    # how many iterations per epoch
    epoch_iteration = int( np.ceil( len(loader.dataset)  / opts.batch_size) )

    # Total number of steps.
    start_epoch = 0
    global_step = 0
    for epoch in range(start_epoch, config.nr_epochs):

        for i, data in enumerate(loader):

            trainer.set_input(data)
            trainer.optimize_parameters()

            if (global_step + 1) % 20 == 0 :
                losses = trainer.get_current_losses()
                record_scalar_summaries(losses, train_writer, global_step)

            global_step += 1
            #motion = data['openpose']
            #print(motion[0].shape)

            #print(data['long_src_mask'][0])
            #print(data['long_tar_mask'][0])

            #motion2gif(motion[0], 512, 512, os.path.join(image_directory, '{:03d}[{:04d}]_{}_{}.gif'.format(i, data['start'][0], data['dataset_key'][0], data['motion_key'][0]) ))
        
        if (epoch) % config.eval_step == 0:
            #output_file = 'test_{:03d}.out'.format(epoch+1)
            output_file = 'test.out'
            evaluator.infer_h5_file(DanceTrans_inference(trainer.pos_encode, trainer.transformer), os.path.join(image_directory, output_file))
            results = evaluator.evaluate_from_h5(os.path.join(image_directory, output_file))
            print_evaluation(results, epoch+1, os.path.join(output_directory, 'history.txt'), train_writer)
            #evaluator.visualize_skeleton(os.path.join(image_directory, output_file), os.path.join(image_directory,'gif_{:03d}'.format(epoch)))

        if (epoch) % config.save_step == 0:
            trainer.save_network(epoch)

        lrs = trainer.update_learning_rate()
        train_writer.add_scalar('LR', lrs[0][0], epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='play our Dance Transformer')

    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
    parser.add_argument('--save-root', type=str, default='./checkpoints/', help="outputs path")
    parser.add_argument('--name', type=str, default='pose', help="outputs path")
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--seed', type=int, default=123)

    main(parser.parse_args())
