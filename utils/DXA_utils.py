"""
File Name: DXA_utils.py
Author: Lambert T Leong
Description: Contains DXA utility functions
"""

import random as rnd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from typing import List

def load_npys(list_of_paths: List[str]) -> np.ndarray: ### system has 1TB or RAM. Use flow from methods on smaller systems
    """
    Load and stack NumPy arrays from a list of file paths.

    Args:
        list_of_paths (List[str]): List of file paths to load NumPy arrays from.

    Returns:
        np.ndarray: Stacked NumPy arrays.
    """
    out = []
    pad=np.zeros((150,1,6))
    for i in list_of_paths:
        npy=np.load(i)/16383
        npy= npy[:,:,:]
        out+=[npy]
    out = np.stack(out)
    return out

### AUGMENTATION FUNC FOR DXA ###
def random_blackout(image,min_channels=4,boxmin=5,boxmax=75):
    black_height, black_width = rnd.randint(boxmin,boxmax),rnd.randint(boxmin,boxmax)
    blackbox_one = np.ones((black_height, black_width))
    im_height, im_width, num_channels = image.shape
    height, width = im_height, im_width
    channel_list = rnd.sample(range(num_channels),rnd.randint(min_channels,num_channels))
    for ch in channel_list:
        black_height, black_width = rnd.randint(boxmin,boxmax),rnd.randint(boxmin,boxmax)
        blackbox_one = np.ones((black_height, black_width))
        height_bound, width_bound = im_height-black_height, im_width-black_width
        while height > (height_bound) and width > (width_bound):
            height, width = rnd.randint(0,height_bound), rnd.randint(0,width_bound)
        flat=image[height:height+black_height,width:width+black_width,ch].flatten()
        np.random.shuffle(flat)
        image[height:height+black_height,width:width+black_width,ch]=flat.reshape((black_height, black_width))
        height,width = im_height,im_width
    return image 

### GEN DXA R IMAGES ###
def process_raw_dxa(input_image: tf.Tensor) -> dict:
    """
    Processes Hologic image data by calculating attenuations and R-values for different phases.

    Args:
        input_image (tf.Tensor): The input image tensor, assumed to be a 4D tensor where the last 
                                 dimension represents different attenuation channels.

    Returns:
        dict: A dictionary of processed images and R-values for air, tissue, and bone.
    """
    # Splitting input image into high and low attenuations for air, bone, and tissue
    atten_air, atten_bone, atten_tissue = input_image[..., :2], input_image[..., 2:4], input_image[..., 4:]

    # Calibration
    calib_bone, calib_tissue = atten_bone - atten_air, atten_tissue - atten_air

    # Adjusted attenuation values
    atten_bone += calib_bone
    atten_tissue += calib_tissue

    # Clipping attenuations
    atten_air = tf.clip_by_value(atten_air, 1 / (2 ** 14 - 1), 1.0)
    atten_bone = tf.clip_by_value(atten_bone, 1 / (2 ** 14 - 1), 1.0)
    atten_tissue = tf.clip_by_value(atten_tissue, 1 / (2 ** 14 - 1), 1.0)

    # Calculate R-Values
    R_values = {
        'air': atten_air[..., 1] / atten_air[..., 0],
        'bone': atten_bone[..., 1] / atten_bone[..., 0],
        'tissue': atten_tissue[..., 1] / atten_tissue[..., 0]
    }

    # Stacking R-values and high/low attenuations
    processed_data = [
        tf.stack([R_values['air'], R_values['bone'], R_values['tissue']], axis=-1),
        atten_air[..., 0],
        atten_air[..., 1],
        tf.stack([R_values['bone'], R_values['air'], R_values['tissue']], axis=-1),
        atten_bone[..., 0],
        atten_bone[..., 1],
        tf.stack([R_values['tissue'], R_values['air'], R_values['bone']], axis=-1),
        atten_tissue[..., 0],
        atten_tissue[..., 1]
    ]

    return processed_data