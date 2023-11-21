"""
File Name: mesh_encoder.py
Author: Lambert T Leong
Description: Creates the mesh encoder which is based off the pointsnet++ model
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras import layers, models, regularizers,  initializers 

class MeshEncoder(tf.keras.Model):
    def __init__(
            self,
            latent_dim: int,
            num_pts: int,
            dropout_rate: float = 0.5,
            filters: int = 32,
            params_factor: int = 2
        ):
            """
            Initialize the MeshDecoder.

            Args:
                latent_dim (int): The dimension of the latent space.
                num_pts (int): The number of points in the input.
                dropout_rate (float, optional): Dropout rate for the model. Defaults to 0.5.
                filters (int, optional): The number of filters. Defaults to 2 ** 5.
                params_factor (int, optional): The parameters factor. Defaults to 2.
            """
            super(MeshEncoder, self).__init__()

            self.latent_dim = latent_dim
            self.num_pts = num_pts
            self.dropout_rate = dropout_rate
            self.filters = filters
            self.params_factor = params_factor
            self.im_input = layers.Input(shape=(self.num_pts, 3))
            self.x = self.tnet(self.im_input, 3)
            self.x = self.conv_bn(self.x, 32 * self.params_factor)
            self.x = self.conv_bn(self.x, 32 * self.params_factor)
            self.x = self.tnet(self.x, 32 * self.params_factor)
            self.x = self.conv_bn(self.x, 32 * self.params_factor)
            self.x = self.conv_bn(self.x, 64 * self.params_factor)
            self.x = self.conv_bn(self.x, 512 * self.params_factor)
            self.x = layers.GlobalMaxPooling1D()(self.x)
            self.x = self.dense_bn(self.x, 256 * self.params_factor)
            self.x = layers.Dropout(self.dropout_rate)(self.x)
            self.x = self.dense_bn(self.x, 128 * 8)
            self.x = layers.Dropout(self.dropout_rate)(self.x)
            self.mu = layers.Dense(self.latent_dim)(self.x)
            self.sigma = layers.Dense(self.latent_dim)(self.x)
            self.latent= self.compute_latent([self.mu, self.sigma])
            self.model = models.Model(self.im_input, self.latent)

    def conv_bn(self, x, filters):
        """
        Convolutional layer followed by BatchNormalization and ReLU activation.

        Args:
            x (tf.Tensor): Input tensor.
            filters (int): Number of filters for the convolutional layer.

        Returns:
            tf.Tensor: Output tensor.
        """
        x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    def dense_bn(self, x, filters):
        """
        Dense layer followed by BatchNormalization and ReLU activation.

        Args:
            x (tf.Tensor): Input tensor.
            filters (int): Number of filters for the dense layer.

        Returns:
            tf.Tensor: Output tensor.
        """
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    def tnet(self, inputs, num_features):
        """
        T-Net architecture for feature transformation.

        Args:
            inputs (tf.Tensor): Input tensor.
            num_features (int): Number of features.

        Returns:
            tf.Tensor: Transformed feature tensor.
        """
        bias = initializers.Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)
        x = self.conv_bn(inputs, 32 * self.params_factor)
        x = self.conv_bn(x, 64 * self.params_factor)
        x = self.conv_bn(x, 512 * self.params_factor)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.dense_bn(x, 256 * self.params_factor)
        x = self.dense_bn(x, 128 * self.params_factor)
        x = layers.Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=bias,
        )(x)
        feat_T = layers.Reshape((num_features, num_features))(x)
        return layers.Dot(axes=(2, 1))([inputs, feat_T])

    def compute_latent(self, x):
        mu, sigma = x
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(sigma / 2) * eps

    def kl_reconstruction_loss(self, true, pred, mu, sigma):
        # Reconstruction loss
        true_flat = tf.reshape(true, [tf.shape(true)[0], -1])
        pred_flat = tf.reshape(pred, [tf.shape(pred)[0], -1])
        reconstruction_loss = tf.reduce_sum(tf.abs(true_flat - pred_flat), axis=-1)
        # KL divergence loss
        kl_loss = 1 + sigma - tf.square(mu) - tf.exp(sigma)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        
        return tf.reduce_mean(reconstruction_loss + kl_loss)

class OrthogonalRegularizer(regularizers.Regularizer):
    """
    Orthogonal Regularizer for enforcing orthogonality in weight matrices.

    Args:
        num_features (int): Number of features.
        l2reg (float): L2 regularization strength (default: 0.001).
    """

    def __init__(self, num_features: int, l2reg: float = 0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the regularization loss.

        Args:
            x (tf.Tensor): Weight tensor.

        Returns:
            tf.Tensor: Orthogonal regularization loss.
        """
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))