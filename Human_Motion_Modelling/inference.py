import os, sys
import time
import numpy as np
import torch
import random
import argparse

from models.trainer import MotInterp_Trainer
from models.evaluator import Evaluator

from utils.utils import *

class Model_inference(torch.nn.Module):
    def __init__(self, enc, transformer):
        super(Model_inference, self).__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.pos_encode = enc.to(self.device)
        self.transformer = transformer.to(self.device)

    def inference(self, data, interp, encoder_mask, decoder_mask, rate):

        self.pos_encode.eval()
        self.transformer.eval()

        # data: N * C * L

        input = torch.unsqueeze(data, dim=0).to(self.device)
        target = torch.unsqueeze(interp, dim=0).to(self.device)

        src_mask = torch.unsqueeze(encoder_mask, dim=0).to(self.device)
        tgt_mask = torch.unsqueeze(decoder_mask, dim=0).to(self.device)

        pos_src = self.pos_encode(src_mask).to(self.device)
        pos_tar = self.pos_encode(tgt_mask).to(self.device)

        pred, _ = self.transformer.forward(
                input, src_mask, pos_src, target, tgt_mask, pos_tar, rate)

        pred = pred.permute(1, 2, 0)

        return pred

def main(opts):

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

    config.out_dir = opts.save_dir

    opts.resume = True
    trainer = MotInterp_Trainer(config, opts.resume)
    evaluator = Evaluator(config)

    
    evaluator.set_model(Model_inference(trainer.pos_encode, trainer.transformer))

        
    pose_dir = opts.pose_dir
    save_dir = opts.save_dir

    subfolderlist = [f for f in sorted(os.listdir(pose_dir)) if os.path.isdir(os.path.join(pose_dir, f)) ]
        
    for i, subfolder in enumerate(subfolderlist):
        pose_path = os.path.join(pose_dir, subfolder)
        save_path = {
                'pred_dir' : os.path.join(save_dir, 'Predict_motion', subfolder),
                'linear_dir' : os.path.join(save_dir, 'Linear_motion', subfolder) }
        evaluator.interpolate_openpose(pose_path, sample_rate=opts.upsample_rate, save_dir=save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='play our Dance Transformer')

    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
    parser.add_argument('--save-dir', type=str, default='./checkpoints/', help="outputs path")
    parser.add_argument('--pose-dir', type=str, help="input low FPS pose path")
    parser.add_argument('--upsample-rate', type=int, default=2, help="input low FPS pose path")

    parser.add_argument('--seed', type=int, default=123)

    main(parser.parse_args())
