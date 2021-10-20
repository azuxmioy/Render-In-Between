import time
import os
from torch.autograd import Variable
import torch
import numpy as np
import numpy
import networks
from my_args import args
from imageio import imread, imsave
from AverageMeter import  *
import shutil
import datetime
import cv2
from  PIL import Image
torch.backends.cudnn.benchmark = True

model = networks.__dict__[args.netName](
                                    channel = args.channels,
                                    filter_size = args.filter_size,
                                    timestep = 0.5,
                                    training = False)

if args.use_cuda:
    model = model.cuda()

model_path = './model_weights/best.pth'
if not os.path.exists(model_path):
    print("*****************************************************************")
    print("**** We couldn't load any trained weights ***********************")
    print("*****************************************************************")
    exit(1)

if args.use_cuda:
    pretrained_dict = torch.load(model_path)
else:
    pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

model_dict = model.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
model.load_state_dict(model_dict)
# 4. release the pretrained dict for saving memory
pretrained_dict = []

model = model.eval() # deploy mode

frames_dir = args.frame_input_dir
output_dir = args.frame_output_dir

torch.set_grad_enabled(False)

loop_timer = AverageMeter()

framelist = [f for f in sorted(os.listdir(frames_dir)) if f.endswith(('jpg','png'))]

# we want to have input_frame between (start_frame-1) and (end_frame-2)
# this is because at each step we read (frame) and (frame+1)
# so the last iteration will actuall be (end_frame-1) and (end_frame)


n_runs = 1

if not os.path.exists(output_dir):
    print("Creating directory: {}".format(output_dir))
    os.makedirs(output_dir)

for i in range (len(framelist) ):
    filename_frame_1 = os.path.join(frames_dir, framelist[i])
    im = Image.open(filename_frame_1).resize((args.width, args.height))
    im.save(os.path.join(output_dir, framelist[i][:-4] + "_{:.06f}.png".format(0)))



for _ in range(n_runs):
    print("run %d rounds" % n_runs)
    framelist = [f for f in sorted(os.listdir(output_dir)) if f.endswith(('jpg','png'))]

    for i in range (len(framelist)-2 ):
        start_time = time.time()
        filename_frame_1 = os.path.join(output_dir, framelist[i])
        filename_frame_2 = os.path.join(output_dir, framelist[i+2])

        img1 = cv2.imread(filename_frame_1)[...,::-1]
        img2 = cv2.imread(filename_frame_2)[...,::-1]
        img1 = cv2.resize(img1, dsize=(args.width, args.height), interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, dsize=(args.width, args.height), interpolation=cv2.INTER_CUBIC)

        X0 = torch.from_numpy(np.transpose(img1, (2,0,1)).astype("float32") / 255.0).type(args.dtype)
        X1 = torch.from_numpy(np.transpose(img2, (2,0,1)).astype("float32") / 255.0).type(args.dtype)

        assert (X0.size(1) == X1.size(1))
        assert (X0.size(2) == X1.size(2))

        intWidth = X0.size(2)
        intHeight = X0.size(1)
        channels = X0.size(0)
        if not channels == 3:
            print(f"Skipping {filename_frame_1}-{filename_frame_2} -- expected 3 color channels but found {channels}.")
            continue

        if intWidth != ((intWidth >> 7) << 7):
            intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft = int((intWidth_pad - intWidth) / 2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intPaddingLeft = 32
            intPaddingRight= 32

        if intHeight != ((intHeight >> 7) << 7):
            intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

        X0 = Variable(torch.unsqueeze(X0,0))
        X1 = Variable(torch.unsqueeze(X1,0))
        X0 = pader(X0)
        X1 = pader(X1)

        if args.use_cuda:
            X0 = X0.cuda()
            X1 = X1.cuda()

        y_s, offset, filter = model(torch.stack((X0, X1),dim = 0))
        y_ = y_s[args.save_which]

        if args.use_cuda:
            X0 = X0.data.cpu().numpy()
            if not isinstance(y_, list):
                y_ = y_.data.cpu().numpy()
            else:
                y_ = [item.data.cpu().numpy() for item in y_]
            offset = [offset_i.data.cpu().numpy() for offset_i in offset]
            filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
            X1 = X1.data.cpu().numpy()
        else:
            X0 = X0.data.numpy()
            if not isinstance(y_, list):
                y_ = y_.data.numpy()
            else:
                y_ = [item.data.numpy() for item in y_]
            offset = [offset_i.data.numpy() for offset_i in offset]
            filter = [filter_i.data.numpy() for filter_i in filter]
            X1 = X1.data.numpy()

        X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
        y_ = [np.transpose(255.0 * item.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight,
                                intPaddingLeft:intPaddingLeft+intWidth], (1, 2, 0)) for item in y_]
        offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
        filter = [np.transpose(
            filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
            (1, 2, 0)) for filter_i in filter]  if filter is not None else None
        X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))

        frame_name = framelist[i][:-12]
        value1 = float(framelist[i][-12:-4])
        value2 = float(framelist[i+1][-12:-4])
        if value2 == 0.0:
            value2 = 1.0
        value = ( value1 + value2 ) / 2.0
        
        output_frame_file_path = os.path.join(output_dir,"fake-" + frame_name + "{:.06f}.png".format(value) )
        imsave(output_frame_file_path, np.round(y_[0]).astype(numpy.uint8))


        end_time = time.time()
        loop_timer.update(end_time - start_time)

        frames_left = len(framelist) - 1 - i
        estimated_seconds_left = frames_left * loop_timer.avg
        estimated_time_left = datetime.timedelta(seconds=estimated_seconds_left)
        print(f"****** Processed frame {i} | Time per frame (avg): {loop_timer.avg:2.2f}s | Time left: {estimated_time_left} ******************" )

print("Finished processing images.")


'''
for i in range (len(framelist)-1 ):

    start_time = time.time()

    X0 = torch.from_numpy(np.transpose(imread(filename_frame_1), (2,0,1)).astype("float32") / 255.0).type(args.dtype)
    X1 = torch.from_numpy(np.transpose(imread(filename_frame_2), (2,0,1)).astype("float32") / 255.0).type(args.dtype)

    assert (X0.size(1) == X1.size(1))
    assert (X0.size(2) == X1.size(2))

    intWidth = X0.size(2)
    intHeight = X0.size(1)
    channels = X0.size(0)
    if not channels == 3:
        print(f"Skipping {filename_frame_1}-{filename_frame_2} -- expected 3 color channels but found {channels}.")
        continue

    if intWidth != ((intWidth >> 7) << 7):
        intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
        intPaddingLeft = int((intWidth_pad - intWidth) / 2)
        intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
    else:
        intPaddingLeft = 32
        intPaddingRight= 32

    if intHeight != ((intHeight >> 7) << 7):
        intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
        intPaddingTop = int((intHeight_pad - intHeight) / 2)
        intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
    else:
        intPaddingTop = 32
        intPaddingBottom = 32

    pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

    X0 = Variable(torch.unsqueeze(X0,0))
    X1 = Variable(torch.unsqueeze(X1,0))
    X0 = pader(X0)
    X1 = pader(X1)

    if args.use_cuda:
        X0 = X0.cuda()
        X1 = X1.cuda()

    y_s, offset, filter = model(torch.stack((X0, X1),dim = 0))
    y_ = y_s[args.save_which]

    if args.use_cuda:
        X0 = X0.data.cpu().numpy()
        if not isinstance(y_, list):
            y_ = y_.data.cpu().numpy()
        else:
            y_ = [item.data.cpu().numpy() for item in y_]
        offset = [offset_i.data.cpu().numpy() for offset_i in offset]
        filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
        X1 = X1.data.cpu().numpy()
    else:
        X0 = X0.data.numpy()
        if not isinstance(y_, list):
            y_ = y_.data.numpy()
        else:
            y_ = [item.data.numpy() for item in y_]
        offset = [offset_i.data.numpy() for offset_i in offset]
        filter = [filter_i.data.numpy() for filter_i in filter]
        X1 = X1.data.numpy()

    X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
    y_ = [np.transpose(255.0 * item.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight,
                                intPaddingLeft:intPaddingLeft+intWidth], (1, 2, 0)) for item in y_]
    offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
    filter = [np.transpose(
        filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
        (1, 2, 0)) for filter_i in filter]  if filter is not None else None
    X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))

    interpolated_frame_number = 0
    shutil.copy(filename_frame_1, os.path.join(output_dir, framelist[i] + "{:0>3d}.png".format(interpolated_frame_number)))
    for item, time_offset in zip(y_, time_offsets):
        interpolated_frame_number += 1
        output_frame_file_path = os.path.join(output_dir, framelist[i] + "{:0>3d}.png".format(interpolated_frame_number))
        imsave(output_frame_file_path, np.round(item).astype(numpy.uint8))

    end_time = time.time()
    loop_timer.update(end_time - start_time)

    frames_left = len(framelist) - 1 - i
    estimated_seconds_left = frames_left * loop_timer.avg
    estimated_time_left = datetime.timedelta(seconds=estimated_seconds_left)
    print(f"****** Processed frame {i} | Time per frame (avg): {loop_timer.avg:2.2f}s | Time left: {estimated_time_left} ******************" )

# Copying last frame
last_frame_filename = os.path.join(frames_dir, framelist[-1])
shutil.copy(last_frame_filename, os.path.join(output_dir, framelist[-1] + "{:0>3d}.png".format(0) ))

print("Finished processing images.")
'''
