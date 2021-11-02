import tensorflow as tf
import numpy as np
import abc

from dense_optimization.network.network import Network


class Transform(abc.ABC):  # Not to touch

    def __init__(self):
        pass

    def __call__(self, network: Network):
        return self.call(network=network)

    @abc.abstractmethod
    def call(self, network: Network):
        pass


class FoldBatchNormTransform(Transform):  # TODO For Later. Optional if you finish early.

    def call(self, network: Network):
        # TODO 1. Find all BatchNorm layers.
        # TODO 2. Fold them into the previous 'Conv2D' or 'Dense' layer.
        raise NotImplementedError()


def fold_dense_batch_norm(dense: tf.keras.layers.Dense, bn: tf.keras.layers.BatchNormalization):
    gamma = tf.constant(bn.gamma)
    beta = tf.constant(bn.beta)
    mu = tf.constant(bn.moving_mean)
    var = tf.constant(bn.moving_variance)

    W = tf.constant(dense.kernel)
    W = tf.reshape(W, (1, 1, W.shape[0], W.shape[1]))
    bias = tf.constant(dense.bias)

    denom = tf.sqrt(var + bn.epsilon)
    gamma_denom = (gamma / denom)

    new_W = W * gamma_denom
    new_bias = (bias - mu) * gamma_denom + beta

    dense.kernel.assign(tf.squeeze(new_W))
    dense.bias.assign(new_bias)


def fold_conv2d_batch_norm(conv: tf.keras.layers.Conv2D, bn: tf.keras.layers.BatchNormalization):
    gamma = tf.constant(bn.gamma)
    beta = tf.constant(bn.beta)
    mu = tf.constant(bn.moving_mean)
    var = tf.constant(bn.moving_variance)

    W = tf.constant(conv.kernel)
    bias = tf.constant(conv.bias)

    denom = tf.sqrt(var + bn.epsilon)
    gamma_denom = (gamma / denom)

    new_W = W * gamma_denom
    new_bias = (bias-mu)*gamma_denom + beta

    conv.kernel.assign(new_W)
    conv.bias.assign(new_bias)
