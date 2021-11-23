import pytest
import networkx as nx
import gc
import numpy as np
import tensorflow as tf
from dense_optimization.network.network import Network
from dense_optimization.transforms.fold_bn import fold_conv2d_batch_norm, fold_dense_batch_norm

LAYER_KEY_ATTRIBUTE = "layer"
LAYER_OUT_KEY_ATTRIBUTE = "output"


class TestNetwork(object):

    @staticmethod
    def build_model():
        input_layer = tf.keras.layers.Input(shape=[80, 80, 1])
        conv_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3))(input_layer)
        batch_norm = tf.keras.layers.BatchNormalization()(conv_layer)
        flatten = tf.keras.layers.Flatten()(batch_norm)
        dense1 = tf.keras.layers.Dense(units=10)(flatten)
        batch_norm2 = tf.keras.layers.BatchNormalization()(dense1)
        dense2 = tf.keras.layers.Dense(units=5)(batch_norm2)
        return tf.keras.Model(inputs=input_layer, outputs=dense2)

    def remove_neighbor(self, G: nx.DiGraph, node):
        predec = list(G.predecessors(node))
        neighs = list(G.neighbors(node))
        for p in predec:
            G.remove_edge(p, node)
        for n in neighs:
            G.remove_edge(node, n)
        G.remove_node(node)

        for p in predec:
            for n in neighs:
                G.add_edge(p, n)

    def test_build_model_again(self):
        # Building a model:
        model = self.build_model()
        for i in range(5):  # Making sure BatchNorm has new values
            model(np.random.uniform(-1, 1, [1, 80, 80, 1]), training=True)

        random_input = np.random.uniform(-1, 1, [1, 80, 80, 1])  # Generating random input for Comparison
        orig_model_outputs = model(random_input, training=False).numpy()  # Saving for later comparison

        # Building a DiGraph out of a Keras Model, manually
        G = nx.DiGraph()
        G.add_node(model.layers[0].name, **{LAYER_KEY_ATTRIBUTE: model.layers[0], "is_input": True})
        G.add_node(model.layers[1].name, **{LAYER_KEY_ATTRIBUTE: model.layers[1]})
        G.add_node(model.layers[2].name, **{LAYER_KEY_ATTRIBUTE: model.layers[2]})
        G.add_node(model.layers[3].name, **{LAYER_KEY_ATTRIBUTE: model.layers[3]})
        G.add_node(model.layers[4].name, **{LAYER_KEY_ATTRIBUTE: model.layers[4]})
        G.add_node(model.layers[5].name, **{LAYER_KEY_ATTRIBUTE: model.layers[5]})
        G.add_node(model.layers[6].name, **{LAYER_KEY_ATTRIBUTE: model.layers[6], "is_output": True})

        G.add_edge(model.layers[0].name, model.layers[1].name)
        G.add_edge(model.layers[1].name, model.layers[2].name)
        G.add_edge(model.layers[2].name, model.layers[3].name)
        G.add_edge(model.layers[3].name, model.layers[4].name)
        G.add_edge(model.layers[4].name, model.layers[5].name)
        G.add_edge(model.layers[5].name, model.layers[6].name)

        # Folding all Batch Norms by calling them manually
        fold_conv2d_batch_norm(G.nodes["conv2d"][LAYER_KEY_ATTRIBUTE], G.nodes["batch_normalization"][LAYER_KEY_ATTRIBUTE])
        self.remove_neighbor(G, "batch_normalization")
        fold_dense_batch_norm(G.nodes["dense"][LAYER_KEY_ATTRIBUTE], G.nodes["batch_normalization_1"][LAYER_KEY_ATTRIBUTE])
        self.remove_neighbor(G, "batch_normalization_1")

        # We're clearing any Session info of Keras, clean page.
        tf.keras.backend.clear_session()

        weights_dict = {}  # To save the Weights to assign to the new Layer object
        for n in G.nodes:
            layer_obj = G.nodes[n][LAYER_KEY_ATTRIBUTE]
            old_weights = [w.numpy() for w in layer_obj.weights]
            weights_dict[n] = old_weights
            layer_class = layer_obj.__class__
            layer_config = layer_obj.get_config()
            G.nodes[n][LAYER_KEY_ATTRIBUTE] = layer_class.from_config(layer_config)
        gc.collect()  # Force call Garbage Collection, just in-case.

        topo_sorted = nx.topological_sort(G)
        for i, n in enumerate(topo_sorted):
            prede = list(G.predecessors(n))
            if len(prede) == 0:
                continue
            for p in prede:
                G.nodes[n][LAYER_KEY_ATTRIBUTE](G.nodes[p][LAYER_KEY_ATTRIBUTE].output)
            for nw, ow in zip(G.nodes[n][LAYER_KEY_ATTRIBUTE].weights, weights_dict[n]):  # Assign previous weights
                nw.assign(ow)

        all_inputs = [G.nodes[n][LAYER_KEY_ATTRIBUTE].output for n in G.nodes if "is_input" in G.nodes[n]]
        all_outputs = [G.nodes[n][LAYER_KEY_ATTRIBUTE].output for n in G.nodes if "is_output" in G.nodes[n]]
        new_model = tf.keras.Model(inputs=all_inputs, outputs=all_outputs)
        new_model_outputs = new_model(random_input)

        assert np.allclose(new_model_outputs, orig_model_outputs)
        print()