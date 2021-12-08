# Human Motion Modeling

<img src="https://github.com/azuxmioy/Render-In-Between/blob/main/img/motion.gif?raw=true">

Our human motion model is trained on a large scale motion capture dataset AMASS. We provide code to synthesize 2D human motion sequences for training from the SMPL parameters defined in AMASS. You can also simply use the pre-trained model to interpolate low-frame-rate noisy human body joints to high-frame-rate motion sequences.

## Dependency

```
pip install torch==1.4.0 torchvision==0.5.0
```

**Note**: We implemented our code and strongly suggest using Pytorch version 1.4.0. Using other version might cause NaN training loss due to the implementation changes in the attention module.

We also provide scripts to generate customized motion dataset. If you want to prepare your own motion dataset, please follow the installation steps of [AMASS](https://github.com/nghorbani/amass) and [VPoser](https://github.com/nghorbani/human_body_prior).


## Data preparation

You can download our pre-built motion dataset in h5 format for training and evaluation. 

⬇️ AMASS_3D_joints.h5 [[MEGA](https://mega.nz/file/FA9EUZRI#QtARPUv1SYwReVvXUiHGbuoIgiJkvs5dqlbCURtdF8o)] (17GB)

Note that in this `h5` file we extract sequences of 21 3D joint posistions to represent human motion (without global rotation and translation). If you need other information such as joint oreientation, global parameters, or shape information, you will need to build it from this [script](https://github.com/azuxmioy/Render-In-Between/blob/main/Human_Motion_Modelling/AMASS/gen_amass_h5.py).


If you want to build your own dataset from raw AMASS motion capture data (in `npy` format), please visit their [webpage](https://amass.is.tue.mpg.de/index.html) and register an account for the download links. There are several sub-datasets and you need to unzip them into a folder to run our script. Also you need a human body model which you can download via [https://mano.is.tue.mpg.de/](https://mano.is.tue.mpg.de/). 

Given the data in `body_models` and `data` folder, you can run our script to generate h5 file for training and test.

```
python AMASS/gen_amass_h5.py 
```

## Training

First put `AMASS_3D_joints.h5` dataset file in the `AMASS` folder. Then start a new training session
```
python train.py --config configs/config.yaml --batch-size 64 --name SESSION_NAME
```

To resume the training session, you need to use identical `SESSION_NAME` with `--resume` flag
```
python train.py --resume --config configs/config.yaml --batch-size 64 --name SESSION_NAME
```
You can visualize the training loss via the tensorboard in the `--save-root` folder.

## Inference
To run the inference code for motion upsampling, adjust the flag `--upsample-rate` for the target motion. For example, the following command will upsample motion at `INPUT_POSE` folder and output its 8x slow motion to `SAVE_DIR`.

```
python inference.py --resume --config configs/config.yaml --pose-dir INPUT_POSE --save-dir SAVE_DIR --upsample-rate 8
```

If you want to evaluate motion from the AMASS dataset, please refer to the evaluation code snipplet at the [training script](https://github.com/azuxmioy/Render-In-Between/blob/main/Human_Motion_Modelling/train.py#L117).

```python=
output_file = 'validation.out'
evaluator.infer_h5_file(Model_inference(trainer.pos_encode, trainer.transformer), os.path.join(image_directory, output_file))
results = evaluator.evaluate_from_h5(os.path.join(image_directory, output_file))
print_evaluation(results, epoch+1, os.path.join(output_directory, 'history.txt'), train_writer)
# Visualize generated human skeleton, uncomment to generate gif file (can be very slow)
evaluator.visualize_skeleton(os.path.join(image_directory, output_file), os.path.join(image_directory,'gif_{:03d}'.format(epoch)))
```
