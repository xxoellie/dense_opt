import tensorflow as tf

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

        def is_batch_norm_layer(x):
            return isinstance(x, (
                tf.keras.layers.BatchNormalization,)) and x.__class__.__name__ == tf.keras.layers.BatchNormalization.__name__
        def is_conv2d_layer(x):
            return isinstance(x, (tf.keras.layers.Conv2D,)) and x.__class__.__name__ == tf.keras.layers.Conv2D.__name__
        def is_dense_layer(x):
            return isinstance(x, (tf.keras.layers.Dense,)) and x.__class__.__name__ == tf.keras.layers.Dense.__name__

        G = network._graph
        nodes = network.nodes
        LAYER_KEY_ATTRIBUTE = network.LAYER_KEY_ATTRIBUTE

        nodes_list = list(nodes)
        for i in range(len(nodes)):
            layer_obj = nodes[nodes_list[i]][LAYER_KEY_ATTRIBUTE]
            if is_batch_norm_layer(layer_obj):
                predecessors = list(G.predecessors(layer_obj.name))
                for pred in predecessors:
                    before_layer = nodes[pred][LAYER_KEY_ATTRIBUTE]
                    if is_dense_layer(before_layer):
                        folded_layer = fold_dense_batch_norm(before_layer, layer_obj)
                    elif is_conv2d_layer(before_layer):
                        folded_layer = fold_conv2d_batch_norm(before_layer, layer_obj)


                successors = list(G.successors(layer_obj.name))
                for succ in successors:
                    after_layer = nodes[succ][LAYER_KEY_ATTRIBUTE]
                    if successors:
                        network.remove_layer(layer_obj)
                        network.remove_layer(before_layer)
                        network.insert_layer_before(after_layer, folded_layer)

        # raise NotImplementedError()


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
    return dense

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
    return conv