import numpy as np
import json
from scipy.optimize import curve_fit
import warnings

def select_largest_bb(jointdicts, thres = 0.3):

    target_idx = -1
    target_height = -1

    for i, joint_dict in enumerate(jointdicts):
        np_joints = np.array(joint_dict['pose_keypoints_2d']).copy()
        np_joints = np_joints.reshape((-1, 3))[:15, :]
        x_cor = np_joints [:, 0]
        y_cor = np_joints [:, 1]
        confidence = np_joints [:, 2]
        valid = (confidence > thres)
        if valid.sum() < 4:
            continue
        width = np.amax(x_cor[np.where(valid)]) - np.amin(x_cor[np.where(valid)])
        height = np.amax(y_cor[np.where(valid)]) - np.amin(y_cor[np.where(valid)])

        area = width * height
        if area > target_height:
            target_height = area
            target_idx = i

    return target_idx

def func(x, a, b, c):    
    return a * x**2 + b * x + c

def linear(x, a, b):
    return a * x + b

def setColor(im, yy, xx, color):
    if len(im.shape) == 3:
        if (im[yy, xx] == 0).all():            
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]            
        else:            
            im[yy, xx, 0] = ((im[yy, xx, 0].astype(float) + color[0]) / 2).astype(np.uint8)
            im[yy, xx, 1] = ((im[yy, xx, 1].astype(float) + color[1]) / 2).astype(np.uint8)
            im[yy, xx, 2] = ((im[yy, xx, 2].astype(float) + color[2]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]

def drawEdge(im, x, y, bw=1, color=(255,255,255), draw_end_points=False):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                setColor(im, yy, xx, color)

        # edge endpoints
        if draw_end_points:
            for i in range(-bw*3, bw*3):
                for j in range(-bw*3, bw*3):
                    if (i**2) + (j**2) < (4 * bw**2):
                        yy = np.maximum(0, np.minimum(h-1, np.array([y[0], y[-1]])+i))
                        xx = np.maximum(0, np.minimum(w-1, np.array([x[0], x[-1]])+j))
                        setColor(im, yy, xx, color)

def interpPoints(x, y):    
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interpPoints(y, x)
        if curve_y is None:
            return None, None
    else:        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")    
            if len(x) < 3:
                popt, _ = curve_fit(linear, x, y, maxfev=10000)
            else:
                popt, _ = curve_fit(func, x, y)                
                if abs(popt[0]) > 1:
                    return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(int(x[0]), int(x[-1]), int(x[-1]-x[0]))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)

def read_keypoints(json_input, size, background=None):
    with open(json_input, encoding='utf-8') as f:
        keypoint_dicts = json.loads(f.read())["people"]

    edge_lists = define_edge_lists()
    w, h = size
    if background is None:
        pose_img = np.zeros((h, w, 3), np.uint8)
    else:
        pose_img = background

    if len(keypoint_dicts) > 0:
        idx = select_largest_bb(keypoint_dicts)
    else:
        idx = -1
    
    if idx != -1:
        keypoint_dict = keypoint_dicts[idx]
        pose_pts = np.array(keypoint_dict["pose_keypoints_2d"]).reshape(-1, 3)
        pts = extract_valid_keypoints(pose_pts, edge_lists) [:15, :]
        pose_img = connect_keypoints(pts, edge_lists, size, pose_img)

    return pose_img

def extract_valid_keypoints(pts, conf, size, thres1=0.1, thres2=0.0001):
    p = len(pts)
    w, h = size
    output = np.zeros((p, 2))

    foot_idx = [8,9,10,11,12,13,14,15,16]

    for i in range(p):
        thes = thres2 if i in foot_idx else thres1
        x = pts[i][0]
        y = pts[i][1]
        c = conf[i]
        if x>=0 and y>=0 and c>thes and x<w and y<h:
            output[i, 0] = x
            output[i, 1] = y
    
    return output

def connect_keypoints(pts, edge_lists, size, random_drop_prob,  background=None):
    pose_pts =  pts
    w, h = size
    if background is None:
        output_edges = np.zeros((h, w, 3), np.uint8)
    else:
        output_edges=background
    pose_edge_list, pose_color_list = edge_lists

    ### pose    
    for i, edge in enumerate(pose_edge_list):
        x, y = pose_pts[edge, 0], pose_pts[edge, 1]
        if (np.random.rand() > random_drop_prob) and (0 not in x):
            curve_x, curve_y = interpPoints(x, y)                                        
            drawEdge(output_edges, curve_x, curve_y, bw=4, color=pose_color_list[i], draw_end_points=True)

    return output_edges

def define_edge_lists(p=15):
    ### pose        
    pose_edge_list = []
    pose_color_list = []

    pose_edge_list += [        
        [ 0,  1], [ 1,  8],                                         # body
        [ 1,  2], [ 2,  3], [ 3,  4],                               # right arm
        [ 1,  5], [ 5,  6], [ 6,  7],                               # left arm
        [ 8,  9], [ 9, 10], [10, 11], # right leg
        [ 8, 12], [12, 13], [13, 14]  # left leg
    ]
    pose_color_list += [
        [153,  0, 51], [153,  0,  0],
        [153, 51,  0], [153,102,  0], [153,153,  0],
        [102,153,  0], [ 51,153,  0], [  0,153,  0],
        [  0,153, 51], [  0,153,102], [  0,153,153], 
        [  0,102,153], [  0, 51,153], [  0,  0,153]
    ]
    if p == 19:
        pose_edge_list += [ [4, 18], [7, 17], [11, 16], [14, 15] ]
        pose_color_list += [ [208,208,  0], [  0,208,  0], [  0,208,208], [  0,  0,208] ]

    return pose_edge_list, pose_color_list
