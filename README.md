# Render In-between: Motion GuidedVideo Synthesis for Action Interpolation

### [Hsuan-I Ho](https://azuxmioy.github.io/), [Xu Chen](https://ait.ethz.ch/people/xu/), [Jie Song](https://ait.ethz.ch/people/song/), [Otmar Hilliges](https://ait.ethz.ch/people/hilliges/)

[Paper] [Video] [Poster]

![](https://i.imgur.com/YTUsW3S.jpg)
![](https://i.imgur.com/85NbuHk.jpg)


This is the offical Pytorch implementation for our work. Our proposed framework is able to synthesize challenging human videos in an action interpolation setting. This repository contains three subdirectories, including code and scripts for preparing our collected [HumanSlomo](https://github.com/azuxmioy/Render-In-Between/tree/main/HumanSloMo_Dataset) dataset, the implementation of [human motion modeling network](https://github.com/azuxmioy/Render-In-Between/tree/main/Human_Motion_Modelling) trained on the large-scale AMASS dataset, as well as the [pose guided neural rendering model](https://github.com/azuxmioy/Render-In-Between/tree/main/Pose_Guided_Neural_Rendering) to synthesize video frames from poses. Please check each subfolder for the detail information and how to execute the code. 

## [HumanSlomo Dataset](https://github.com/azuxmioy/Render-In-Between/tree/main/HumanSloMo_Dataset) 

![](https://i.imgur.com/Dbmkrta.gif)


We collected a set of high FPS createive commons of human videos from Youtube. The video are manually split into several continuous clips for training and test. You can also build your own video dataset using the provided scripts.

## [Human Motion Modeling](https://github.com/azuxmioy/Render-In-Between/tree/main/Human_Motion_Modelling)
![](https://i.imgur.com/6RDU54s.gif)

Our human motion model is trained on a large scale motion capture dataset AMASS. We provide code to synthesize 2D human motion sequences for training from the SMPL parameters defined in AMASS. You can also simply use the pre-trained model to interpolate low-frame-rate noisy human body joints to high-frame-rate motion sequences.


## [Pose Guided Neural Rendering](https://github.com/azuxmioy/Render-In-Between/tree/main/Pose_Guided_Neural_Rendering)
![](https://i.imgur.com/8W1ndka.gif)
The neural rendering model learned to map the pose sequences back to the original video domain. The final result is composed with the background warping from DAIN and the generated human body according to the predicted blending mask autoregressively. The model is trained in a conditional image generation setting, given only low-frame-rate videos as training data. Therefore, you can train your custom neural rendering model by constructing your own video dataset.

## Results and Videos
TBA

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

imaginaire, AMASS, DAIN, AlphaPose, DERT
