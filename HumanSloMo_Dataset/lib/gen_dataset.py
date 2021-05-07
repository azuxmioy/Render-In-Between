import os
import argparse
from collections import defaultdict, OrderedDict
import json
import cv2
from tqdm import tqdm
import numpy as np

EDN_list = ['00_Dance', '01_Dance']
video_list =  ['02_Boxing', '03_Boxing', '04_Basketball', '05_Body', '06_Body', '07_Kungfu', '08_Kungfu', '09_Kungfu']

def main(args):

    with open(args.json_path) as pf:
        json_data = json.load(pf)

    for clip_name, clip in tqdm(json_data.items()):

        if not os.path.exists(os.path.join(args.output_path, clip_name)):
            os.makedirs(os.path.join(args.output_path, clip_name))

        video_name = clip['video_name']
        image_list = clip['video_frames']
        clip_len = clip['num_frames']


        if clip_name in video_list:
            vidcap = cv2.VideoCapture(os.path.join(args.input_path, video_name + '.mp4'))
            success, image = vidcap.read()
            count = 0
            frame = 0
            while success:
                image_name = "frame%05d.png" % count
                if image_name in image_list:
                    image = image[:, 100:1180]
                    image = cv2.resize(image, (768,512))  # to save space and memory usage, we downsize to 768 x 512  
                    cv2.imwrite(os.path.join(args.output_path, clip_name, "frame%05d.png" % count), image )     # save frame as png file
                    frame += 1
                if frame >= clip_len:
                    break
                success, image = vidcap.read()
                count += 1
        else:
            for img_name in image_list:
                image_path = os.path.join(args.input_path, video_name, 'val/test_img/', img_name) # Copy from the EDN dataset
                image = cv2.imread(image_path)
                image = image[:, 128:896]   # to save space and memory usage, we downsize to 768 x 512  
                cv2.imwrite(os.path.join(args.output_path, clip_name, img_name), image )     # save frame as png file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generating image frames from videos')

    parser.add_argument('-i', '--input-path', type=str, default='videos/', help='folder path of downloaded videos')
    parser.add_argument('-j', '--json-path', type=str, default='metadata/train_list.json', help='dataset json file')
    parser.add_argument('-o', '--output-path', type=str, default='HumanSloMo/train/')


    main(parser.parse_args())
