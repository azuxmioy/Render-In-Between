
import argparse
import glob
import csv
import os

#/root/share/openpose/examples/openpose/openpose.bin --image_dir {} --face --hand --display 0 --render_pose 0 --write_json {}
#python3 generate.py --exp_name CAIN_fin --dataset custom --data_root {} --save_path {} --runs {} --img_fmt png --batch_size 32 --test_batch_size 16 --model cain --depth 3 --loss 1*L1 --resume --mode test

def main(args):

    subfolderlist = [f for f in sorted(os.listdir(args.input_path)) if os.path.isdir(os.path.join(args.input_path,f)) ]
    print(subfolderlist)
    runs = [3, 3, 3, 3, 3, \
            3, 3, 3, 3, 3, \
            3, 3, 3, 3, 3, \
            3, 3, 4, 2, 2, \
            4, 4, 4, 3, 3, \
            3, 3, 3, 3, 3, \
            3, 3, 3, 3, 3, \
            3, 3, 3, 4, 2, \
            2, 2, 4, 3, 3, \
            3, 4, 4, 4, 4]
    with open(args.tmp_file, "w") as f:

        for i, subfolder in enumerate(subfolderlist):
            image_path = os.path.join(args.input_path, subfolder)
            save_path = os.path.join(args.output_path, subfolder)
        
            cmd_base = "python3 -W ignore DAIN_interpolate.py --netName DAIN_slowmotion"
            cmd_base += " --frame_input_dir {} --frame_output_dir {} --width 480 --height 320 --img_ext png".format(image_path, save_path)

            f.write("%s\n" % cmd_base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='to download videos for ReID')

    parser.add_argument('-o', "--output_path", type=str, required=True, help="Path of the output video folder")
    parser.add_argument('-i', "--input_path", default='video_data.csv', help="Location of the csv file (video list)")
    parser.add_argument('-t', "--tmp_file", default='tmp.sh', help="Output script location.")
    
    main(parser.parse_args())




