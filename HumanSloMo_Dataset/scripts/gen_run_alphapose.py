
import argparse
import os


def main(args):

    subfolderlist = [f for f in sorted(os.listdir(args.input_path)) if os.path.isdir(os.path.join(args.input_path,f)) ]
    print(subfolderlist)

    with open(args.tmp_file, "w") as f:

        for subfolder in subfolderlist:
            image_path = os.path.join(args.input_path, subfolder)
            save_path = os.path.join(args.output_path, subfolder)
        
            cmd_base = "python3 scripts/demo_inference.py --cfg configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression-coco_wholebody.yml "
            cmd_base +="--checkpoint pretrained_models/final_DPG.pth --min_box_area 10000 --format open --pose_track --sp "
            cmd_base += "--indir {} --outdir {}".format(image_path, save_path)

            f.write("%s\n" % cmd_base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='to download videos for ReID')

    parser.add_argument('-o', "--output_path", type=str, required=True, help="Path of the output video folder")
    parser.add_argument('-i', "--input_path", default='video_data.csv', help="Location of the csv file (video list)")
    parser.add_argument('-t', "--tmp_file", default='tmp.sh', help="Output script location.")
    
    main(parser.parse_args())




