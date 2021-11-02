import pytest
import os
os.environ["CUDA_VISIBLE_DEVICeS"] = ""

import tensorflow as tf
import numpy as np
from dense_optimization.transforms.fold_bn import fold_batch_norm_conv2d


class TestFoldBN(object):

    def test_fold_bn(self):
        input_shape = [1, 80, 80, 1]
        dummy_input = tf.random.uniform(input_shape, dtype=tf.float32)
        conv_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), )
        bn_layer = tf.keras.layers.BatchNormalization()
        _ = bn_layer(conv_layer(dummy_input))

        conv_layer.bias.assign(tf.random.uniform(conv_layer.bias.shape))
        conv_layer.kernel.assign(tf.random.uniform(conv_layer.kernel.shape))
        bn_layer.gamma.assign(tf.random.uniform(bn_layer.gamma.shape))
        bn_layer.beta.assign(tf.random.uniform(bn_layer.beta.shape))
        bn_layer.moving_mean.assign(tf.random.uniform(bn_layer.moving_mean.shape))
        bn_layer.moving_variance.assign(tf.random.uniform(bn_layer.moving_variance.shape))

        result_before = bn_layer(conv_layer(dummy_input), training=False)
        fold_batch_norm_conv2d(conv_layer, bn_layer)
        results_after = conv_layer(dummy_input)

        assert np.allclose(result_before.numpy(), results_after.numpy(), atol=1e-1)
