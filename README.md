# Render In-between: Motion Guided Video Synthesis for Action Interpolation

[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0327.pdf)] [[Supp](https://www.bmvc2021-virtualconference.com/assets/supp/0327_supp.zip)] [[arXiv](https://arxiv.org/abs/2111.01029)] [[4min Video](https://www.bmvc2021-virtualconference.com/programme/schedule/#poster-session-id-3)]

<img src="https://github.com/azuxmioy/Render-In-Between/blob/main/img/teaser.gif?raw=true">
<img src="https://github.com/azuxmioy/Render-In-Between/blob/main/img/overview.jpg?raw=true">

This is the official Pytorch implementation for our work. Our proposed framework is able to synthesize challenging human videos in an action interpolation setting. This repository contains three subdirectories, including code and scripts for preparing our collected [HumanSlomo](https://github.com/azuxmioy/Render-In-Between/tree/main/HumanSloMo_Dataset) dataset, the implementation of [human motion modeling network](https://github.com/azuxmioy/Render-In-Between/tree/main/Human_Motion_Modelling) trained on the large-scale AMASS dataset, as well as the [pose-guided neural rendering model](https://github.com/azuxmioy/Render-In-Between/tree/main/Pose_Guided_Neural_Rendering) to synthesize video frames from poses. Please check each subfolder for the detailed information and how to execute the code. 

## [HumanSlomo Dataset](https://github.com/azuxmioy/Render-In-Between/tree/main/HumanSloMo_Dataset) 

<img src="https://github.com/azuxmioy/Render-In-Between/blob/main/img/thumbnail.gif?raw=true">

We collected a set of high FPS creative commons of human videos from Youtube. The videos are manually split into several continuous clips for training and test. You can also build your video dataset using the provided scripts.

## [Human Motion Modeling](https://github.com/azuxmioy/Render-In-Between/tree/main/Human_Motion_Modelling)

<img src="https://github.com/azuxmioy/Render-In-Between/blob/main/img/motion.gif?raw=true">

Our human motion model is trained on a large scale motion capture dataset AMASS. We provide code to synthesize 2D human motion sequences for training from the SMPL parameters defined in AMASS. You can also simply use the pre-trained model to interpolate low-frame-rate noisy human body joints to high-frame-rate motion sequences.


## [Pose Guided Neural Rendering](https://github.com/azuxmioy/Render-In-Between/tree/main/Pose_Guided_Neural_Rendering)

<img src="https://github.com/azuxmioy/Render-In-Between/blob/main/img/generation.gif?raw=true">

The neural rendering model learned to map the pose sequences back to the original video domain. The final result is composed with the background warping from DAIN and the generated human body according to the predicted blending mask autoregressively. The model is trained in a conditional image generation setting, given only low-frame-rate videos as training data. Therefore, you can train your custom neural rendering model by constructing your own video dataset.

## Quick Start

⬇️ example.zip [[MEGA](https://mega.nz/file/8dsFRKzA#Uw1AF-lOdmb5y9zYNj5EiIm-4CYx6nHh5nM3OeqHjZU)] (25.4MB)

Download this example action clip which includes necessary input files for our pipeline.

The first step is generating high FPS motion from low FPS poses with our motion modeling network.
```
cd Human_Motion_Modelling
python inference.py --pose-dir ../example/input_poses --save-dir ../example/ --upsample-rate 2
```


⬇️ checkpoints.zip [[MEGA](https://mega.nz/file/tRsWHA4L#vZTA9Zc29EgAvajSfQb98lc2JDETy1gjPOFLWtll77Y)] (147.2MB)

Next we will map high FPS poses back to video frames with our pose-guided neural rendering. Download the checkpoint files to the corresponding folder to run the model.
```
cd Pose_Guided_Neural_Rendering
python inference.py --input-dir ../example/ --save-dir ../example/
```


## Citation
```
@inproceedings{ho2021render,
    author = {Hsuan-I Ho, Xu Chen, Jie Song, Otmar Hilliges},
    title = {Render In-between: Motion GuidedVideo Synthesis for Action Interpolation},
    booktitle = {BMVC},
    year = {2021}
}
```

## Acknowledgement
We use the pre-processing code in [AMASS](https://github.com/nghorbani/amass) to synthesize our motion dataset. [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) is used for generating 2D human body poses. [DAIN](https://github.com/baowenbo/DAIN) is used for warping background images. Our human motion modeling network is based on the transformer backbone in [DERT](https://github.com/facebookresearch/detr). Our pose-guided neural rendering model is based on [imaginaire](https://github.com/NVlabs/imaginaire). We sincerely thank these authors for their awesome work.
