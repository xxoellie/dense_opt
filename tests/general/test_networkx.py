import pytest
import networkx as nx
import tensorflow as tf

LAYER_KEY_ATTRIBUTE = "layer"

class TestNetworkX(object):

    def test_attributes(self):
        """
        Testing adding attribute on creation of new node
        """
        G = nx.DiGraph()
        conv_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3))

        attrs_to_add = {
            LAYER_KEY_ATTRIBUTE: conv_layer
        }
        # How to add attributes to a node:
        G.add_node(conv_layer.name, **attrs_to_add)

        # How to get the attribute:
        layer_attr_from_node = G.nodes[conv_layer.name][LAYER_KEY_ATTRIBUTE]  # type: tf.keras.layers.Conv2D

        layer_name_from_node_attr = layer_attr_from_node.name
        assert layer_name_from_node_attr == conv_layer.name
        kernel_size_from_attr = layer_attr_from_node.kernel_size
        assert all(map(lambda xs: xs[0] == xs[1], zip(kernel_size_from_attr, conv_layer.kernel_size)))

    def test_attributes_2(self):
        """
        Testing adding attributes to a node after it's creation.
        """
        G = nx.DiGraph()
        conv_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3))

        G.add_node(conv_layer.name)

        # How to add attributes to a node after it's creation:
        attrs_to_add = {
            LAYER_KEY_ATTRIBUTE: conv_layer
        }
        nx.set_node_attributes(G, {conv_layer.name: attrs_to_add})

        # How to get the attribute:
        layer_attr_from_node = G.nodes[conv_layer.name][LAYER_KEY_ATTRIBUTE]  # type: tf.keras.layers.Conv2D

        layer_name_from_node_attr = layer_attr_from_node.name
        assert layer_name_from_node_attr == conv_layer.name

    def test_searching_all_conv_in_graph(self):
        """
        Testing adding attributes to a node after it's creation.
        """

        def is_conv_layer(x):
            return isinstance(x, (tf.keras.layers.Conv2D, )) and x.__class__.__name__ == tf.keras.layers.Conv2D.__name__

        def is_batch_norm_layer(x):
            return isinstance(x, (tf.keras.layers.BatchNormalization, )) and x.__class__.__name__ == tf.keras.layers.BatchNormalization.__name__


        G = nx.DiGraph()
        input_layer = tf.keras.layers.InputLayer(input_shape=[80, 80, 1])
        conv_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3))
        batch_norm = tf.keras.layers.BatchNormalization()
        flatten = tf.keras.layers.Flatten()
        dense1 = tf.keras.layers.Dense(units=10)
        batch_norm2 = tf.keras.layers.Dense(units=10)
        dense2 = tf.keras.layers.Dense(units=5)

        G.add_node(input_layer.name, **{LAYER_KEY_ATTRIBUTE: input_layer})
        G.add_node(conv_layer.name, **{LAYER_KEY_ATTRIBUTE: conv_layer})
        G.add_node(batch_norm.name, **{LAYER_KEY_ATTRIBUTE: batch_norm})
        G.add_node(flatten.name, **{LAYER_KEY_ATTRIBUTE: flatten})
        G.add_node(dense1.name, **{LAYER_KEY_ATTRIBUTE: dense1})
        G.add_node(batch_norm2.name, **{LAYER_KEY_ATTRIBUTE: batch_norm2})
        G.add_node(dense2.name, **{LAYER_KEY_ATTRIBUTE: dense2})

        G.add_edge(input_layer.name, conv_layer.name)
        G.add_edge(conv_layer.name, batch_norm.name)
        G.add_edge(batch_norm.name, flatten.name)
        G.add_edge(flatten.name, dense1.name)
        G.add_edge(dense1.name, batch_norm2.name)
        G.add_edge(batch_norm2.name, dense2.name)

        # How to get the attribute:
        all_conv_layers = []
        for n in G.nodes:
            layer_obj = G.nodes[n][LAYER_KEY_ATTRIBUTE]
            if is_conv_layer(layer_obj):
                all_conv_layers.append(n)
        assert len(all_conv_layers) == 1

    def test_find_all_batch_norms_with_conv_or_dense_before_them(self):
        G = nx.DiGraph()
        input_layer = tf.keras.layers.InputLayer(input_shape=[80, 80, 1])
        conv_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3))
        batch_norm = tf.keras.layers.BatchNormalization()
        flatten = tf.keras.layers.Flatten()
        dense1 = tf.keras.layers.Dense(units=10)
        batch_norm2 = tf.keras.layers.Dense(units=10)
        dense2 = tf.keras.layers.Dense(units=5)

        G.add_node(input_layer.name, **{LAYER_KEY_ATTRIBUTE: input_layer})
        G.add_node(conv_layer.name, **{LAYER_KEY_ATTRIBUTE: conv_layer})
        G.add_node(batch_norm.name, **{LAYER_KEY_ATTRIBUTE: batch_norm})
        G.add_node(flatten.name, **{LAYER_KEY_ATTRIBUTE: flatten})
        G.add_node(dense1.name, **{LAYER_KEY_ATTRIBUTE: dense1})
        G.add_node(batch_norm2.name, **{LAYER_KEY_ATTRIBUTE: batch_norm2})
        G.add_node(dense2.name, **{LAYER_KEY_ATTRIBUTE: dense2})

        G.add_edge(input_layer.name, conv_layer.name)
        G.add_edge(conv_layer.name, batch_norm.name)
        G.add_edge(batch_norm.name, flatten.name)
        G.add_edge(flatten.name, dense1.name)
        G.add_edge(dense1.name, batch_norm2.name)
        G.add_edge(batch_norm2.name, dense2.name)

        def is_batch_norm_layer(x):
            return isinstance(x, (
                tf.keras.layers.BatchNormalization,)) and x.__class__.__name__ == tf.keras.layers.BatchNormalization.__name__
        def is_conv2d_layer(x):
            return isinstance(x, (tf.keras.layers.Conv2D,)) and x.__class__.__name__ == tf.keras.layers.Conv2D.__name__
        def is_dense_layer(x):
            return isinstance(x, (tf.keras.layers.Dense,)) and x.__class__.__name__ == tf.keras.layers.Dense.__name__

        batchs = []
        nodes_list = list(G.nodes)
        for i in range(len(G.nodes())):
            layer_obj = G.nodes[nodes_list[i]][LAYER_KEY_ATTRIBUTE]
            if is_batch_norm_layer(layer_obj):
                predecssors = list(G.predecessors(layer_obj.name))
                for pred in predecssors:
                    before_layer = G.nodes[pred][LAYER_KEY_ATTRIBUTE]
                    if is_dense_layer(before_layer) or is_conv2d_layer(before_layer):
                        batchs.append(layer_obj)
        print(batchs)
        assert len(batchs) == 1
