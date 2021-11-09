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