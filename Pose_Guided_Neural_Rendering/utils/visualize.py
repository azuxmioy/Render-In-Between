  
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

def motion2video(motion, h, w, save_path, colors, transparency=False, motion_tgt=None, fps=25, save_frame=False):
    nr_joints = motion.shape[0]
    videowriter = imageio.get_writer(save_path, fps=fps)
    vlen = motion.shape[-1]
    if save_frame:
        frames_dir = save_path[:-4] + '-frames'
        ensure_dir(frames_dir)
    for i in tqdm(range(vlen)):
        [img, img_cropped] = joints2image(motion[:, :, i], colors, transparency, H=h, W=w, nr_joints=nr_joints)
        if motion_tgt is not None:
            [img_tgt, img_tgt_cropped] = joints2image(motion_tgt[:, :, i], colors, transparency, H=h, W=w, nr_joints=nr_joints)
            img_ori = img.copy()
            img = cv2.addWeighted(img_tgt, 0.3, img_ori, 0.7, 0)
            img_cropped = cv2.addWeighted(img_tgt, 0.3, img_ori, 0.7, 0)
            bb = bounding_box(img_cropped)
            img_cropped = img_cropped[:, bb[2]:bb[3], :]
        if save_frame:
            save_image(img_cropped, os.path.join(frames_dir, "%04d.png" % i))
        videowriter.append_data(img)
    videowriter.close()


def motion2gif(motion, h, w, save_path, transparency=False, motion_tgt=None, relocate=False, save_frame=False):
    nr_joints = motion.shape[0]
    images = []
    vlen = motion.shape[-1]
    color1 = hex2rgb('#a50b69#b73b87#db9dc3')
    color2 = hex2rgb('#0000ff#0000aa#000055')

    if save_frame:
        frames_dir = os.path.join(os.path.dirname(save_path), 'test_img')
        if not os.path.exists(frames_dir):
            print("Creating directory: {}".format(frames_dir))
            os.makedirs(frames_dir)

    if motion_tgt is not None and relocate:
        assert motion.shape == motion_tgt.shape
        offset = motion_tgt [8:9, :, :] - motion[8:9, :, :]
        motion = motion + offset

    for i in range(vlen):
        if motion_tgt is not None:

            img, img_cropped = joints2image(motion[:, :, i], color1, transparency, H=h, W=w, nr_joints=nr_joints)

            img_tgt, img_tgt_cropped = joints2image(motion_tgt[:, :, i], color2, transparency, H=h, W=w, nr_joints=nr_joints)
            img_ori = img.copy()
            img = cv2.addWeighted(img_tgt, 0.5, img_ori, 0.5, 0)

        else:
            img, img_cropped = joints2image(motion[:, :, i], color1, transparency, H=h, W=w, nr_joints=nr_joints)

        if save_frame:
            Image.fromarray(img).save(os.path.join(frames_dir, "%04d.png" % i))

        images.append(img)

    imageio.mimsave(save_path, images)


def joints2image(joints_position, colors, transparency=False, H=512, W=512, nr_joints=49, imtype=np.uint8):
    nr_joints = joints_position.shape[0]

    if nr_joints == 49: # full joints(49): basic(15) + eyes(2) + toes(2) + hands(30)
        limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], \
                   [8, 9], [8, 13], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16],
                   ]#[0, 17], [0, 18]] #ignore eyes

        L = rgb2rgba(colors[0]) if transparency else colors[0]
        M = rgb2rgba(colors[1]) if transparency else colors[1]
        R = rgb2rgba(colors[2]) if transparency else colors[2]

        colors_joints = [M, M, R, R, R, L, L,
                  L, M, R, R, R, R, L, L, L,
                  L, L, R] + [R] * 15 + [L] * 15

        colors_limbs = [M, L, R, M, L, L, R,
                  R, L, R, L, L, L, R, R, R,
                  R, R]
        hips = joints_position[8]
        neck = joints_position[1]
        head = joints_position[0]

    elif nr_joints == 15 or nr_joints == 17: # basic joints(15) + (toe(2))
        limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], 
                   [6, 7], [8, 9], [8, 12], [9, 10], [10, 11], [12, 13], [13, 14]]
                    # [0, 15], [0, 16] two eyes are not drawn

        L = rgb2rgba(colors[0]) if transparency else colors[0]
        M = rgb2rgba(colors[1]) if transparency else colors[1]
        R = rgb2rgba(colors[2]) if transparency else colors[2]

        colors_joints = [M, M, R, R, R, L, L,
                         L, M, R, R, R, L, L, L,]

        colors_limbs = [M, R, L, M, R, R, L,
                        L, R, L, R, R, L, L]
        if nr_joints == 17:
            limbSeq  += [[14, 15], [11, 16] ]
            colors_joints += [L, R]
            colors_limbs  += [L, R]

        hips = joints_position[8]
        neck = joints_position[1]
        head = joints_position[0]

    elif nr_joints == 22 or nr_joints == 52:
        limbSeq = [[15, 12], [12, 17], [12, 16], [12, 0], [17, 19], [19, 21], [16, 18], [18, 20],
                   [0, 2], [0, 1], [2, 5], [5, 8], [8,11], [1, 4], [4, 7], [7, 10]]
                    # [0, 15], [0, 16] two eyes are not drawn

        L = rgb2rgba(colors[0]) if transparency else colors[0]
        M = rgb2rgba(colors[1]) if transparency else colors[1]
        R = rgb2rgba(colors[2]) if transparency else colors[2]

        colors_joints = [M, L, R, M, L, R, M, L, R, M, L, R,
                         M, L, R, M, L, R, L, R, L, R]

        colors_limbs = [M, R, L, M, R, R, L, L,
                        R, L, R, R, R, L, L, L]
        hips = joints_position[0]
        neck = joints_position[12]
        head = joints_position[15]

    else:
        raise ValueError("Only support number of joints be 49 or 17 or 15")

    if transparency:
        canvas = np.zeros(shape=(H, W, 4))
    else:
        canvas = np.ones(shape=(H, W, 3)) * 255

    torso_length = ((hips[1] - neck[1]) ** 2 + (hips[0] - neck[0]) ** 2) ** 0.5

    head_radius = int(torso_length/4.5)
    end_effectors_radius = int(torso_length/15)
    end_effectors_radius = 7
    joints_radius = 7

    cv2.circle(canvas, (int(head[0]),int(head[1])), 20, colors_joints[0], thickness=-1)

    for i in range(0, len(colors_joints)):
        '''
        if i in (17, 18):
            continue
        elif i > 18:
            radius = 2
        else:
        '''
        radius = joints_radius
        cv2.circle(canvas, (int(joints_position[i][0]),int(joints_position[i][1])), radius, colors_joints[i], thickness=-1)
        cv2.putText(canvas, "%d" %i, (int(joints_position[i][0]),int(joints_position[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors_joints[i])
        
    stickwidth = 2

    for i in range(len(limbSeq)):
        limb = limbSeq[i]
        cur_canvas = canvas.copy()
        point1_index = limb[0]
        point2_index = limb[1]

        #if len(all_peaks[point1_index]) > 0 and len(all_peaks[point2_index]) > 0:
        point1 = joints_position[point1_index]
        point2 = joints_position[point2_index]
        X = [point1[1], point2[1]]
        Y = [point1[0], point2[0]]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        alpha = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(alpha), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors_limbs[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        bb = bounding_box(canvas)
        canvas_cropped = canvas[:,bb[2]:bb[3], :]
    
    return canvas.astype(imtype), canvas_cropped.astype(imtype)

def rgb2rgba(color):
    return (color[0], color[1], color[2], 255)

def hex2rgb(hex, number_of_colors=3):
    h = hex
    rgb = []
    for i in range(number_of_colors):
        h = h.lstrip('#')
        hex_color = h[0:6]
        rgb_color = [int(hex_color[i:i+2], 16) for i in (0, 2 ,4)]
        rgb.append(rgb_color)
        h = h[6:]
    return rgb

def bounding_box(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox