# Pose Guided Neural Rendering

<img src="https://github.com/azuxmioy/Render-In-Between/blob/main/img/generation.gif?raw=true">

The neural rendering model learned to map the pose sequences back to the original video domain. The final result is composed with the background warping from DAIN and the generated human body according to the predicted blending mask autoregressively. The model is trained in a conditional image generation setting, given only low-frame-rate videos as training data. Therefore, you can train your custom neural rendering model by constructing your own video dataset.

## Inference from images
Download the models we trained on HumanSlomo dataset. Unzip them into the `checkpoints` folder.

⬇️ checkpoints.zip [[MEGA](https://mega.nz/file/tRsWHA4L#vZTA9Zc29EgAvajSfQb98lc2JDETy1gjPOFLWtll77Y)] (147.2MB)

Prepare the following folders with images in `INPUT_PATH` .
```
|-- INPUT_PATH/
|   |-- inputs/ ..................... low FPS video frames
|        ...
|   |-- Predict_motion/ ............. generated high FPS motion
|        ...
|   |-- DAIN/  ..................... generated high FPS background images
|        ...
|-- 
```
Then run the command, you will get high FPS frames in `SAVE_PATH`.
```
python inference.py --input-dir INPUT_PATH --save-dir SAVE_PATH
```

## Train your model

The neural rendering model generates human bodies based on training data. While training with the whole dataset is possible, it is recommended to use only one subject per training for better visualization quality. First you have to put `HumanSlomo.h5` in the same folder, then start a new training session.

```
python train.py --config configs/HSM.yaml --batch-size 4 --name SESSION_NAME
```

Adjust the training clips you want to use in the configuration [yaml](https://github.com/azuxmioy/Render-In-Between/blob/main/Pose_Guided_Neural_Rendering/configs/HSM.yaml) file.