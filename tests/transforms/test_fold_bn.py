import pytest
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import numpy as np
from dense_optimization.transforms.fold_bn import fold_conv2d_batch_norm, fold_dense_batch_norm, FoldBatchNormTransform
from dense_optimization.network.network import Network
from tests.driver.model import model

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

    def test_call(self):
        input_shape = [1, 28, 28, 1]
        dummy_input = tf.random.uniform(input_shape, dtype=tf.float32)

        network = Network()
        network.convert(model)
        before_model = network._model
        before_bn = len(network.nodes())
        before_output = before_model(dummy_input)
        Fold_bn = FoldBatchNormTransform()
        Fold_bn(network)

        after_model = network.build_model()
        after_output = after_model(dummy_input)
        after_bn = len(network.nodes())
        # network.draw_network(network._graph)
        print(dummy_input)
        print(before_output)
        print(after_output)
        # assert before_bn == after_bn + 4
        #
        # assert np.allclose(before_output.numpy(), after_output.numpy(), atol=1e-5)

    def test_bn(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), input_shape=(28, 28, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3)),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(10),
        ])

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # data Normalization
        x_train, x_test = tf.cast(x_train, tf.float32) / 255, tf.cast(x_test, tf.float32) / 255
        x_train = tf.reshape(x_train, [-1, 28, 28, 1])
        x_test = tf.reshape(x_test, [-1, 28, 28, 1])
        # y_train = tf.one_hot(y_train, 10)
        # y_test = tf.one_hot(y_test, 10)
        training_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
        validation_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000).batch(32)

        for data in training_dataset:
            x, y = data
            pred = model(x)

            print(model.layers[1].gamma.numpy())