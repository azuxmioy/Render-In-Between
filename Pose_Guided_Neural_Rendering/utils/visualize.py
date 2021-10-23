  
import numpy as np
import os
import imageio
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from utils.utils import tensor2images

def print_losses(epoch, i, errors, t, dump_file=None):

    message = '(epoch: %d, iters: %d, time: %.3f) \n' % (epoch, i, t)
    for key, value in errors.items():
        message +='%s: %.3f ' % (key, value.item())
    print(message)
    
    if dump_file is not None:
        print(message, file=open(dump_file, 'a'))

def print_evaluation(results_dict, epoch, dump_file=None, writer=None):
    print('evaluate in epoch:{:>3}'.format(epoch))

    for key, value in results_dict.items():
        print('{:10} : {:12.6}' . format (key, value))

    if dump_file is not None:
        print('evaluate in epoch:{:>3}'.format(epoch), file=open(dump_file, 'a'))
        for key, value in results_dict.items():
            print('{:10} : {:12.6}'. format (key, value), file=open(dump_file, 'a'))
    
    if writer is not None:
        text=''
        for key, value in results_dict.items():
            text += '{:10} : {:12.6}\n'.format (key, value)
        writer.add_text('eval_{}'.format(epoch), text)

def make_video(results, save_path, fps=30, save_frame=False):
    pred, fuse, target, dain, skeleton, mask = \
        results['gen'], results['fuse'], results['gt'], results['dain'], results['skeleton'], results['mask']
    videowriter = imageio.get_writer(save_path, fps=fps)

    if save_frame:
        frames_dir = os.path.join(os.path.dirname(save_path), 'output_frames')
        if not os.path.exists(frames_dir):
            print("Creating directory: {}".format(frames_dir))
            os.makedirs(frames_dir)

    frame_idx = 0
    for gen, fu, gt, fake, sk, msk in tqdm(zip(pred, fuse, target, dain, skeleton, mask), total = len(pred)):
        fig = plt.figure(figsize=(48, 26), dpi=40, facecolor='white')
        ax1 = plt.subplot(2,3,1)
        ax1.set_title('Predict', fontsize=60, color='b')
        ax2 = plt.subplot(2,3,2)
        ax2.set_title('Mask', fontsize=60, color='b')
        ax3 = plt.subplot(2,3,3)
        ax3.set_title('Fuse', fontsize=60, color='b')
        ax4 = plt.subplot(2,3,4)
        ax4.set_title('CAIN', fontsize=60, color='b')
        ax5 = plt.subplot(2,3,5)
        ax5.set_title('Ground Truth', fontsize=60, color='b')
        ax6 = plt.subplot(2,3,6)
        ax6.set_title('Skeleton', fontsize=60, color='b')
        pil_gen = Image.fromarray(tensor2images(gen))
        pil_fake = Image.fromarray(tensor2images(fake))
        pil_gt = Image.fromarray(tensor2images(gt))
        pil_sk = Image.fromarray(tensor2images(sk))
        pil_msk = Image.fromarray(tensor2images(msk))
        pil_fu = Image.fromarray(tensor2images(fu))
        ax1.imshow(pil_gen)
        ax2.imshow(pil_msk)
        ax3.imshow(pil_fu)
        ax4.imshow(pil_fake)
        ax5.imshow(pil_gt)
        ax6.imshow(pil_sk)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if save_frame:
            Image.fromarray(img).save(os.path.join(frames_dir, "%04d.png" % frame_idx))
    
        videowriter.append_data(img)
        frame_idx +=1
        plt.close()
    videowriter.close()