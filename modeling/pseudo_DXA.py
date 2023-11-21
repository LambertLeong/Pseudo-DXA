"""
File Name: pseudo_DXA.py
Author: Lambert T Leong
Description: Code to create the Pseudo-DXA model
"""

import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import List
from tensorflow.keras.models import Model
from typing import Tuple, Union

sys.path.append('../utils/')
import DXA_utils as dxa

class PseudoDXA(tf.keras.Model):
    def __init__(self, encoder: Model, generator: Model):
        """
        PseudoDXA Model.

        Args:
            encoder (Model): Encoder model.
            generator (Model): Generator model.
        """        
        super(PseudoDXA, self).__init__()
        assert isinstance(encoder, models.Model) and isinstance(generator, models.Model), \
            "Both encoder and generator must be instances of tf.keras.models.Model"
        self.encoder = encoder
        self.generator = generator
        self.lossModel = self.get_loss_model() 
    
    def build_model(self):
        """
        Build the PseudoDXA model by unrolling layers of encoder and generator.
        """
        self.encoder._name = 'mesh_encoder_ll'
        self.generator._name = 'dxa_generator_ll'
        inputs = layers.Input(shape=self.encoder.input_shape[1:], name='input_fitted_mesh')
        encoded = self.encoder(inputs)
        gen = self.generator(encoded)
        self.model = models.Model(inputs, gen)
    
    def freeze_generator(self, freeze: bool = True):
        """
        Freeze or unfreeze the generator's layers.

        Args:
            freeze (bool): If True, freeze the generator's layers. If False, unfreeze them.
        """
        self.generator.trainable = not freeze
        for layer in self.generator.layers:
            layer.trainable = not freeze
    
    def get_loss_model(self) -> Model:
        """
        Get the loss model for perceptual loss.

        Returns:
            Model: The loss model.
        """       
        # Initialize VGG16 model
        lossModel = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        # Set the entire model as non-trainable
        lossModel.trainable = False
        for layer in lossModel.layers:
            layer.trainable = False
        # Define selected layers and their corresponding weights, hardcoded for now but can fix to read from config file
        self.selectedLayers = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18]
        self.selected_layer_weights = [.5, .75, .5, .5, .8, .8, .5, .5, .3, .3, .5, .5, .3, .8, .8, .5]
        # Extract outputs for each selected layer
        self.selectedOutputs = [lossModel.layers[i].output for i in self.selectedLayers]
        # Create a new model with multiple outputs
        return tf.keras.models.Model(inputs=lossModel.inputs, outputs=self.selectedOutputs)

    def call(self, inputs):
        # Encode the inputs using the encoder
        latent_space = self.encoder(inputs)
        # Decode the latent space using the generator
        reconstructed = self.generator(latent_space)
        return reconstructed
    
    @tf.function
    def perceptual_loss_leong(self, input_image: tf.Tensor, reconstruct_image: tf.Tensor) -> tf.Tensor:
        """
        Calculate the perceptual loss between an input image and its reconstruction.

        Args:
            input_image (tf.Tensor): The original input image.
            reconstruct_image (tf.Tensor): The reconstructed image.

        Returns:
            tf.Tensor: The calculated perceptual loss.
        """
        lossModel=self.lossModel
        rc_loss = 0.0
        transform_loss = 0.0
        mse_loss=0.0
        # Process the images through the Hologic function
        #act_ims = self.process_raw_dxa(input_image)
        #rec_ims = self.process_raw_dxa(reconstruct_image)
        act_ims = self.transform_raw_dxa(input_image)
        rec_ims = self.transform_raw_dxa(reconstruct_image)
        #print(act_ims[0],'\n\n\n')
        #print(act_ims)
        #sys.exit(0)
        # Calculate transform loss for processed images
        for i in range(len(act_ims)):
            h1_list = lossModel(act_ims[i])  
            h2_list = lossModel(rec_ims[i])  
            for h1, h2, weight in zip(h1_list, h2_list, self.selected_layer_weights[1:]):
                h1 = tf.reshape(h1, [tf.shape(h1)[0], -1])
                h2 = tf.reshape(h2, [tf.shape(h2)[0], -1]) 
                transform_loss += weight * tf.reduce_sum(tf.abs(h1 - h2), axis=-1)
                mse_loss += tf.keras.losses.mean_squared_error(h1, h2)
        # 6 phase loss for the first three channels
        h1_list = lossModel(input_image[:,:,:,:3])
        h2_list = lossModel(reconstruct_image[:,:,:,:3])
        for h1, h2, weight in zip(h1_list, h2_list, self.selected_layer_weights):
            h1 = tf.reshape(h1, [tf.shape(h1)[0], -1])
            h2 = tf.reshape(h2, [tf.shape(h2)[0], -1])
            rc_loss += weight * tf.reduce_sum(tf.abs(h1 - h2), axis=-1)

        # 6 phase loss for the last three channels
        h1_list = lossModel(input_image[:,:,:,-3:])
        h2_list = lossModel(reconstruct_image[:,:,:,-3:])
        for h1, h2, weight in zip(h1_list, h2_list, self.selected_layer_weights):
            h1 = tf.reshape(h1, [tf.shape(h1)[0], -1])
            h2 = tf.reshape(h2, [tf.shape(h2)[0], -1])
            rc_loss += weight * tf.reduce_sum(tf.abs(h1 - h2), axis=-1)
        
        return (rc_loss*.2)+(transform_loss*.5) + (mse_loss)
    
    def transform_raw_dxa(self, input_image: tf.Tensor):
        """
        Transform raw DXA image data.

        Args:
            input_image (tf.Tensor): Input DXA image.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Transformed images.
        """
        min_clip = 1 / (2 ** 14 - 1)  # Hardcoded minimum clip value

        # Splitting input image into high and low attenuations for air, bone, and tissue
        atten_air_hi, atten_air_lo = input_image[..., 0], input_image[..., 1]
        atten_bone_hi, atten_bone_lo = input_image[..., 2], input_image[..., 3]
        atten_tissue_hi, atten_tissue_lo = input_image[..., 4], input_image[..., 5]

        # Background correction
        calib_tissue_hi = atten_tissue_hi - atten_air_hi
        calib_tissue_lo = atten_tissue_lo - atten_air_lo
        calib_bone_hi = atten_bone_hi - atten_air_hi
        calib_bone_lo = atten_bone_lo - atten_air_lo

        # Adjusted attenuation values
        atten_bone_air_hi = atten_air_hi + calib_bone_hi
        atten_bone_tissue_hi = atten_tissue_hi + calib_bone_hi
        atten_bone_air_lo = atten_air_lo + calib_bone_lo
        atten_bone_tissue_lo = atten_tissue_lo + calib_bone_lo
        atten_tissue_air_hi = atten_air_hi + calib_tissue_hi
        atten_tissue_bone_hi = atten_bone_hi + calib_tissue_hi
        atten_tissue_air_lo = atten_air_lo + calib_tissue_lo
        atten_tissue_bone_lo = atten_bone_lo + calib_tissue_lo

        # Final adjusted images
        atten_bone_hi += calib_bone_hi
        atten_bone_lo += calib_bone_lo
        atten_tissue_hi += calib_tissue_hi
        atten_tissue_lo += calib_tissue_lo

        # Clipping to avoid divide-by-zero errors
        atten_air_hi = tf.clip_by_value(atten_air_hi, min_clip, 1.0)
        atten_air_lo = tf.clip_by_value(atten_air_lo, min_clip, 1.0)
        atten_air_bone_hi = tf.clip_by_value(atten_bone_hi, min_clip, 1.0) 
        atten_air_bone_lo = tf.clip_by_value(atten_bone_lo, min_clip, 1.0)
        atten_air_tissue_hi = tf.clip_by_value(atten_tissue_hi, min_clip, 1.0)
        atten_air_tissue_lo = tf.clip_by_value(atten_tissue_lo, min_clip, 1.0)
        atten_tissue_hi = tf.clip_by_value(atten_tissue_hi, min_clip, 1.0)
        atten_tissue_lo = tf.clip_by_value(atten_tissue_lo, min_clip, 1.0)
        atten_tissue_air_hi = tf.clip_by_value(atten_tissue_air_hi, min_clip, 1.0)
        atten_tissue_air_lo = tf.clip_by_value(atten_tissue_air_lo, min_clip, 1.0)
        atten_tissue_bone_hi = tf.clip_by_value(atten_tissue_bone_hi, min_clip, 1.0)
        atten_tissue_bone_lo = tf.clip_by_value(atten_tissue_bone_lo, min_clip, 1.0)
        atten_bone_hi = tf.clip_by_value(atten_bone_hi, min_clip, 1.0)
        atten_bone_lo = tf.clip_by_value(atten_bone_lo, min_clip, 1.0)
        atten_bone_air_hi = tf.clip_by_value(atten_bone_air_hi, min_clip, 1.0)
        atten_bone_air_lo = tf.clip_by_value(atten_bone_air_lo, min_clip, 1.0)
        atten_bone_tissue_hi = tf.clip_by_value(atten_bone_tissue_hi, min_clip, 1.0)
        atten_bone_tissue_lo = tf.clip_by_value(atten_bone_tissue_lo, min_clip, 1.0)
        #atten_bone_tissue_lo = tf.clip_by_value(atten_bone_tissue_lo, min_clip, 1.0)

        # R-Value calculation
        R_air = atten_air_lo / atten_air_hi
        R_air_bone = atten_bone_air_lo / atten_bone_air_hi
        R_air_tissue = atten_air_tissue_lo / atten_air_tissue_hi
        R_bone = atten_bone_lo / atten_bone_hi
        R_bone_air = atten_bone_air_lo / atten_bone_air_hi
        R_bone_tissue = atten_bone_tissue_lo / atten_bone_tissue_hi
        R_tissue = atten_tissue_lo / atten_tissue_hi
        R_tissue_air = atten_tissue_air_lo / atten_tissue_air_hi
        R_tissue_bone = atten_tissue_bone_lo / atten_tissue_bone_hi

        # Stacking R-values and high/low attenuations
        air_r = tf.stack([R_air, R_air_bone, R_air_tissue], axis=-1)
        air_hi = tf.stack([atten_air_hi, atten_bone_air_hi, atten_air_tissue_hi], axis=-1)
        air_lo = tf.stack([atten_air_lo, atten_bone_air_lo, atten_air_tissue_lo], axis=-1)
        tissue_r = tf.stack([R_tissue, R_tissue_air, R_tissue_bone], axis=-1)
        tissue_hi = tf.stack([atten_tissue_hi, atten_tissue_bone_hi, atten_tissue_air_hi], axis=-1)
        tissue_lo = tf.stack([atten_tissue_lo, atten_tissue_bone_lo, atten_tissue_air_lo], axis=-1)
        bone_r = tf.stack([R_bone, R_bone_air, R_bone_tissue], axis=-1)
        bone_hi = tf.stack([atten_bone_hi, atten_bone_air_hi, atten_bone_tissue_hi], axis=-1)
        bone_lo = tf.stack([atten_bone_lo, atten_bone_air_lo, atten_bone_tissue_lo], axis=-1)

        return air_r, air_hi, air_lo, tissue_r, tissue_hi, tissue_lo, bone_r, bone_hi, bone_lo