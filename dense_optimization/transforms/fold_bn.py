import tensorflow as tf
import numpy as np
import abc

from dense_optimization.network.network import Network


class Transform(abc.ABC):

    def __init__(self):
        pass

    def __call__(self, network: Network):
        return self.call(network=network)

    @abc.abstractmethod
    def call(self, network: Network):
        pass


class FoldBatchNormTransform(Transform):

    def call(self, network: Network):
        # TODO 1. Find all BatchNorm layers.
        # TODO 2. Fold them into the previous 'Conv2D' or 'Dense' layer.
        raise NotImplementedError()


def fold_batch_norm_dense(dense: tf.keras.layers.Dense, bn: tf.keras.layers.BatchNormalization):
    pass  # TODO to Implement



def fold_batch_norm_conv2d(conv: tf.keras.layers.Conv2D, bn: tf.keras.layers.BatchNormalization, epsilon=1e-6):
    gamma = bn.gamma.numpy()
    gamma = gamma.reshape((1, 1, 1, gamma.shape[0]))
    beta = bn.beta.numpy()
    mean = bn.moving_mean.numpy()
    variance = bn.moving_variance.numpy()
    variance = variance.reshape((1, 1, 1, variance.shape[0]))

    denom = np.sqrt(variance + epsilon)

    W = conv.kernel.numpy()
    bias = conv.bias.numpy()

    # denom = torch.sqrt(var + eps)
    b = beta - gamma * mean / denom
    A = gamma / denom
    bias *= A.reshape((bias.shape[0], ))
    # A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

    W *= A
    bias += b.squeeze()

    conv.kernel.assign(W)

    conv.bias.assign(bias)
