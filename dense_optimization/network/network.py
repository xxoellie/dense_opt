import networkx as nx
import tensorflow as tf
from tensorflow.keras.layers import Layer

class Network(object):
    
    def __init__(self):
        self._graph = nx.DiGraph()
        self._model = None

    def insert_layer_before(self, layer: Layer, layer_to_add: Layer):
        """
        Inserts a new layer *before* layer
        :param layer:
        :param layer_to_add:
        :return: None
        """
        raise NotImplementedError()  # TODO Insert a layer before and connect

    def insert_layer_after(self, layer: Layer, layer_to_add: Layer):
        """
        Inserts a new layer *after* layer
        :param layer:
        :param layer_to_add:
        :return: None
        """
        raise NotImplementedError()  # TODO Insert a new layer after layer and connect those

    def remove_layer(self, layer: Layer):
        """
        Removes the layer from the graph and connects accordingly
        :param layer:
        :return:
        """
        raise NotImplementedError()  # TODO Remove a layer and connect the previous to the next

    def convert(self, model: tf.keras.Model) -> nx.DiGraph:
        """
        Convert a Keras model to a networkx graph.
        Model should be Sequential only now.
        :param model:
        :return:
        """
        raise NotImplementedError()  # TODO To convert to nx.DiGraph()

    def build_model(self) -> tf.keras.Model:  # TODO Not for now!
        """
        Returns a new Keras Model
        :return:
        """
        raise NotImplementedError()  # TODO build a new model from the current nx.DiGraph (self.graph)