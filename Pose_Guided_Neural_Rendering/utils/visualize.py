  
import numpy as np
import os
import cv2
import math
import imageio
from tqdm import tqdm
from PIL import Image

def print_losses(epoch, i, errors, t, dump_file=None):

    message = '(epoch: %d, iters: %d, time: %.3f) \n' % (epoch, i, t)
    for key, value in errors.items():
        message += '{:10} : {:12.6}' . format (key, value)
    print(message)
    
    if dump_file is not None:
        print(message, file=open(dump_file, 'a'))

def print_evaluation(results_dict, dump_file=None):
    for key, value in results_dict.items():
        print('{:10} : {:12.6%}' . format (key, value))
        if dump_file is not None:
            print('{:10} : {:12.6%}'. format (key, value), file=open(dump_file, 'a'))