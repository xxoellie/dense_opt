import pytest
import os
os.environ["CUDA_VISIBLE_DEVICeS"] = ""

import tensorflow as tf
import numpy as np
from dense_optimization.transforms.fold_bn import fold_conv2d_batch_norm, fold_dense_batch_norm


class TestFoldBN(object):

    def test_conv2d_batch_norm(self):
        input_shape = [1, 80, 80, 3]
        dummy_input = tf.random.uniform(input_shape, dtype=tf.float32)
        conv_layer = tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), )
        bn_layer = tf.keras.layers.BatchNormalization()
        _ = bn_layer(conv_layer(dummy_input))

        conv_layer.bias.assign(tf.random.uniform(conv_layer.bias.shape, dtype=tf.float32))
        conv_layer.kernel.assign(tf.random.uniform(conv_layer.kernel.shape, dtype=tf.float32))
        bn_layer.gamma.assign(tf.ones(bn_layer.gamma.shape, dtype=tf.float32))
        bn_layer.beta.assign(tf.zeros(bn_layer.beta.shape, dtype=tf.float32))
        bn_layer.moving_mean.assign(tf.random.uniform(bn_layer.moving_mean.shape, dtype=tf.float32))
        bn_layer.moving_variance.assign(tf.random.uniform(bn_layer.moving_variance.shape, dtype=tf.float32))

        result_before = bn_layer(conv_layer(dummy_input), training=False)
        fold_conv2d_batch_norm(conv_layer, bn_layer)
        results_after = conv_layer(dummy_input)

        assert np.allclose(result_before.numpy(), results_after.numpy(), atol=1e-5)

    def test_dense_batch_norm(self):
        input_shape = [1, 512]
        dummy_input = tf.random.uniform(input_shape, dtype=tf.float32)
        dense_layer = tf.keras.layers.Dense(units=1024)
        bn_layer = tf.keras.layers.BatchNormalization()
        _ = bn_layer(dense_layer(dummy_input))

        dense_layer.bias.assign(tf.random.uniform(dense_layer.bias.shape, dtype=tf.float32))
        dense_layer.kernel.assign(tf.random.uniform(dense_layer.kernel.shape, dtype=tf.float32))

        bn_layer.gamma.assign(tf.ones(bn_layer.gamma.shape, dtype=tf.float32))
        bn_layer.beta.assign(tf.zeros(bn_layer.beta.shape, dtype=tf.float32))
        bn_layer.moving_mean.assign(tf.random.uniform(bn_layer.moving_mean.shape, dtype=tf.float32))
        bn_layer.moving_variance.assign(tf.random.uniform(bn_layer.moving_variance.shape, dtype=tf.float32))

        result_before = bn_layer(dense_layer(dummy_input), training=False)
        fold_dense_batch_norm(dense_layer, bn_layer)
        results_after = dense_layer(dummy_input)

        assert np.allclose(result_before.numpy(), results_after.numpy(), atol=1e-5)