import os, sys
import time
import numpy as np
import torch
import random
import argparse
import shutil

from datasets import find_dataset_using_name

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.trainer import Motion_recovery_auto
from models.evaluator import Evaluator

from utils.record_summary import record_image_summaries, record_scalar_summaries
from utils.visualize import print_losses
from utils.utils import *


def main(opts):

    # Set random seed.
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)


    # Load experiment setting
    config = get_config(opts.config)

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

    dataset_class = find_dataset_using_name('HSM_auto')
    dataset = dataset_class(config, config.h5_file)

    loader = DataLoader(dataset=dataset, 
                        batch_size=opts.batch_size, 
                        shuffle=True, 
                        num_workers=opts.workers,
                        pin_memory=True,
                        drop_last=False)


    trainer = Motion_recovery_auto(config, opts.resume)
    evaluator = Evaluator(config)

    # how many iterations per epoch
    epoch_iteration = len(loader.dataset) // opts.batch_size 

    # Total number of steps.
    start_epoch = -1
    total_steps = (trainer.start_epoch+1) * epoch_iteration

    for epoch in range(start_epoch+1, config.nr_epochs):
        epoch_start_time = time.time()
        print('Epoch: %d, total %d samples' % (epoch, len(loader.dataset)))
        for i, data in enumerate(loader):
            iter_start_time = time.time()

            trainer.set_input(data)
            trainer.optimize_parameters()

            
            # Dump training stats in log file / display in console
            #if (total_steps + 1) % config.print_freq == 0:
            #    errors = trainer.get_current_losses()
            #    t = (time.time() - iter_start_time) / opts.batch_size
            #    print_losses(epoch+1, i+1, errors, t, os.path.join(output_directory, 'history.txt'))
            
            # Record scalars to tensorboard
            if (total_steps + 1) % config.display_freq == 0:
                losses = trainer.get_current_losses()
                record_scalar_summaries(losses, train_writer, total_steps + 1)
            
            # Record images to tensorboard
            if (total_steps + 1) % (config.display_freq * 5) == 0:
                # Record image summaries
                image_results = trainer.get_current_visuals(num_images=config.num_image)
                record_image_summaries(image_results, train_writer, total_steps + 1, config.num_image)

            total_steps += 1

        trainer.update_learning_rate()

        if (epoch+1) % config.save_step == 0:
            trainer.save_network(epoch+1)

        if (epoch+1) % config.eval_step == 0:
            evaluator.evaluate_from_dataset(trainer.net_G, epoch+1, max_keyframes=config.eval_frames)

        if (epoch+1) % config.update_frame_step == 0:
            loader.dataset.update_max_frame(loader.dataset.get_max_frames() + 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='play our Dance Transformer')

    parser.add_argument('--config', type=str, default='configs/HSM.yaml', help='Path to the config file.')
    parser.add_argument('--save-root', type=str, default='./checkpoints/', help="outputs path")
    parser.add_argument('--name', type=str, default='pose', help="outputs path")
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--seed', type=int, default=123)

    main(parser.parse_args())
