import tensorflow as tf
import numpy as np


def fold_batch_norm_conv2d(conv: tf.keras.layers.Conv2D, bn: tf.keras.layers.BatchNormalization, epsilon=1e-6):
    gamma = bn.gamma.numpy()
    gamma = gamma.reshape((1, 1, 1, gamma.shape[0]))
    beta = bn.beta.numpy()
    mean = bn.moving_mean.numpy()
    variance = bn.moving_variance.numpy()
    variance = variance.reshape((1, 1, 1, variance.shape[0]))

    bottom_part = np.sqrt(variance + epsilon)

    new_kernel = (conv.kernel * gamma) / bottom_part
    conv.kernel.assign(new_kernel)

    new_bias = beta + (conv.bias - mean) * gamma / bottom_part
    conv.bias.assign(tf.reshape(new_bias, (-1, )))
