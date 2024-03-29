import os, sys
import numpy as np
import torch
import random
import argparse
from models.trainer import Motion_recovery_auto
from models.evaluator import Evaluator

from utils.utils import *

def main(opts):

    # Set random seed.
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)


    # Load experiment setting
    config = get_config(opts.config)
    
    config.out_dir = opts.save_dir
    config.eval_dir = opts.save_dir

    opts.resume = True

    trainer = Motion_recovery_auto(config, opts.resume)
    evaluator = Evaluator(config)

    train_dir = os.path.join(opts.input_dir, 'inputs')
    dain_dir = os.path.join(opts.input_dir, 'DAIN')
    pose_dir =  os.path.join(opts.input_dir, 'Predict_motion')
    save_dir = os.path.join(opts.save_dir, 'Generated_frames')
    # gt_dir = os.path.join(opts.input_dir, 'gt')
    evaluator.evaluate_from_folder(trainer.net_G, train_dir, dain_dir, pose_dir, save_dir, gt_dir=None, gen_vid=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='play our Dance Transformer')

    parser.add_argument('--config', type=str, default='configs/HSM.yaml', help='Path to the config file.')
    parser.add_argument('--save-dir', type=str, default='../example', help="outputs path")
    parser.add_argument('--input-dir', type=str,  required=True ,help="input low FPS frames and pose input")

    parser.add_argument('--seed', type=int, default=123)

    main(parser.parse_args())
