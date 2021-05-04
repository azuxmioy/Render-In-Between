"""
This file contrains functions to record scalar/image tensorboard summaries
"""
import numpy as np


# Record the scalar summaries
def record_scalar_summaries(input_dict, writer, global_step):
    # Convert to scalars
    # import ipdb; ipdb.set_trace()
    scalars_dict = {}
    for key, value in input_dict.items():
        scalars_dict[key] = value.item()
        writer.add_scalar(key, scalars_dict[key], global_step)

# Record the image summaries
def record_image_summaries(input_dict_raw, writer, global_step, num_image):
    if num_image == 1:
        dataformat = 'HWC'
    else:
        dataformat = 'NHWC'

    # convert images to float
    input_dict = {}
    for key, value in input_dict_raw.items():
        input_dict[key] = value / 255.
        writer.add_images(key, value / 255., global_step, dataformats=dataformat)


# Convert tensor of bgr images to rgb images
def bgr_to_rgb(input_images):
    # Fetch individual channels
    b_channel = input_images[..., 0][..., None]
    g_channel = input_images[..., 1][..., None]
    r_channel = input_images[..., 2][..., None]

    return np.concatenate((r_channel, g_channel, b_channel), axis=-1) 
