"""
File Name: dxa_generator.py
Author: Lambert T Leong
Description: Contains code to create the DXA generator
"""

import tensorflow as tf
from tensorflow.keras import layers, models, initializers, regularizers, initializers
import numpy as np
from functools import reduce
from typing import Tuple, Union

class DXAGenerator(tf.keras.Model):
    def __init__(self, reshape_dim: Tuple[int, int, int], upscale: int, model_path: Union[None, str] = None):
        """
        DXA Image Generator Model.

        Args:
            reshape_dim (Tuple[int, int, int]): Dimensions to reshape the input.
            upscale (int): Upscaling factor for the generated image.
            model_path (Union[None, str]): Path to a pre-trained model if available, otherwise None.
        """
        super(DXAGenerator, self).__init__()
        
        self.reshape_dim = reshape_dim
        self.upscale = upscale
        self.latent_dim = reduce(lambda x, y: x * y, reshape_dim)
        self.gen_input = layers.Input(shape=(self.latent_dim,))
        
        if not model_path: 
            self.get_generator()
        else:
            from tensorflow import keras
            self.model = keras.models.load_model(model_path, custom_objects={'tf': tf})

    def get_generator(self):
        """
        Build the DXA image generator model.
        """
        x = layers.Reshape(target_shape=self.reshape_dim)(self.gen_input)
        x = layers.PReLU(shared_axes=[1, 2])(x)
        filters = self.reshape_dim[2]
        
        for i in range(5):
            x = layers.Conv2D(filters, (1, 1), padding="same")(x)
            if i != 0:
                x = layers.Conv2D(filters, (1, 2), padding="same")(x)
            x = layers.PReLU(shared_axes=[1, 2])(x)
            x = self.tf_resize(x.shape, self.upscale)(x)
            x = layers.Conv2D(filters, (2, 2), padding="same")(x)
            x = layers.PReLU(shared_axes=[1, 2])(x)
            x = layers.Conv2D(filters, (4, 4), padding="same")(x)
            x = layers.PReLU(shared_axes=[1, 2])(x)
            
            if i > 2:
                x = layers.Conv2D(filters, (8, 8), padding="same")(x)
                x = layers.PReLU(shared_axes=[1, 2])(x)
            filters //= self.upscale
        
        x = layers.Conv2D(filters, (8, 8), padding="same")(x)
        x = layers.Cropping2D(cropping=((5, 5), (9, 10)))(x)
        x = layers.Conv2D(6, (1, 1), padding="same")(x)
        self.model = tf.keras.models.Model(self.gen_input, x, name="dxa_gen")
    
    def tf_resize(self, in_shape: Tuple[int, int, int], scale: int = 2) -> layers.Lambda:
        """
        Custom TensorFlow layer for resizing.

        Args:
            in_shape (Tuple[int, int, int]): Input shape to be resized.
            scale (int): Upscaling factor.

        Returns:
            layers.Lambda: Lambda layer for resizing.
        """
        new_row, new_col = in_shape[1] * scale, in_shape[2] * scale
        return layers.Lambda(lambda x: tf.image.resize(x, (new_row, new_col), method=tf.image.ResizeMethod.BILINEAR))
