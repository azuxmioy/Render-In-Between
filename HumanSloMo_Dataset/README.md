# HumanSlomo Dataset

<img src="https://github.com/azuxmioy/Render-In-Between/blob/main/img/thumbnail.gif?raw=true">

We collected a set of high FPS creative commons of human videos from Youtube. The videos are manually split into several continuous clips for training and test. You can also build your video dataset using the provided scripts.

## Download our h5 dataset

⬇️ HumanSlomo.h5 [[MEGA](https://mega.nz/file/IItADBRa#hbdnPfKcVFL0QCRwBTGuNTCBIGXSm6hM6SI4fecrI94)] (3.23GB)

Our h5 dataset includes preprocessed poses and warping background images. It is mainly used for training. Of course you can build the dataset with other pose detectors or optical flow models and add customized videos yourself. Please follow the steps below.

## Download the videos
First, make sure you have [youtube-dl](https://github.com/ytdl-org/youtube-dl) and [gdown](https://github.com/wkentaro/gdown) installed in your environment. You can install via `pip` or `brew` (MacOS)
```bash
pip install gdown youtube-dl
```
In the file [video.csv](https://github.com/azuxmioy/Render-In-Between/blob/main/HumanSloMo_Dataset/metadata/video.csv), we provide video links and other information used in our HumanSlomo dataset. Besides, we also inculde two videos from [Everybody Dance Now](https://github.com/carolineec/EverybodyDanceNow). Please download the files via the following google drive links.

```bash
bash download.sh VIDEOPATH metadata/video.csv
cd VIDEOPATH
gdown https://drive.google.com/uc?id=1OlqoZumoeWyWmoGGrrf3AyHU7Rr1O55P
gdown https://drive.google.com/uc?id=1Fi0U27qA1RS2T5kCI0E4eg3nVdmTKNkj
tar xvf subject1.tar
tar xvf subject2.tar
mv subject1 00_Dance
mv subject2 01_Dance
```

Check your VIDEOPATH folder, you should get **2** image folders and **8** `mp4` videos in total.

## Build the dataset 

```bash
bash gen_dataset.sh VIDEOPATH DATASETPATH
```
The script will split the videos into several continuous clips for training and inference. Note that if you want to build your own video dataset, make sure to proive [json](https://github.com/azuxmioy/Render-In-Between/blob/main/HumanSloMo_Dataset/metadata/train_list.json) files for the script. It takes several time to extract video frames into images. (Be aware of your data storage)

## Pre-process pose estimation file and warping image

To save time and memory usage for training and inference, we pre-process required inputs data at this stage. Please prepare the following code for pose estimation and background warping. We provide python scripts based on their code to infrerece data, please check the [script folder](https://github.com/azuxmioy/Render-In-Between/tree/main/HumanSloMo_Dataset/scripts). 

### Pose estimation

You can use whatever pose detectors such as [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) or [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose). The only thing need to be careful is the output json file format. We follow the [default openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md#pose-output-format-body_25) format (2D BODY_25 keypoints+hand keypoints). In our paper and experiments, we modified the AlphaPose output format into the openpose style. You can download our modified version [here](https://mega.nz/file/0QVihJKb#vTqxBCnisHc_5pjD1LkvjFZ0xxynCnw5jMRTQUDwEoI) and please make sure you have installed some dependent packeges.

```bash
pip install cython
sudo apt-get install libyaml-dev
python setup.py build develop --user
```

### Background image warping
We use [DAIN](https://github.com/baowenbo/DAIN) to warp background images. Please clone their repo and follow their installation instruction. We strongly suggest using **only Pytorch version 1.3.1**. Because in other versions there are changes in the grid sampling function, which makes the output images shinked with some boarder paddings.

```
git clone https://github.com/baowenbo/DAIN.git
```

## Compile dataset to h5 file
Make sure you arrage the file structure as in this [example]() folder.
```
path/to/your/dataset/
|-- train/
|   |-- frames/ ..................... low FPS video frames
|        ...
|   |-- poses/  ..................... corresponding poses
|        ...
|   |-- DAIN/   ..................... background for training
|        ...
|-- test/
|   |-- gt/ ..................... high FPS video frames
|        ...
|   |-- inputs/ ..................... low FPS video frames for test
|        ...
|   |-- poses/  ..................... high FPS poses
|        ...
|   |-- input_poses/  ..................... low FPS video poses for test
|        ...
|   |-- DAIN/   ..................... high FPS backgrounds
|        ...
|-- 
```
Our data loader read h5 file during training. This is for saving storage file counts and I/O on our server. To generate the h5 file, simply run the following command and you'll get your h5 file in the `OUTPUTPATH`.

```
bash gen_h5.sh DATASETPATH OUTPUTPATH
```
