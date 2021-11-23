import networkx as nx
import tensorflow as tf
from matplotlib import pyplot as plt
import gc

class Network(object):

    LAYER_KEY_ATTRIBUTE = 'layer'

    def __init__(self):

        self._graph = nx.DiGraph()
        self._model = None
        self.weights_dict = {}

    @property
    def nodes(self):
        return self._graph.nodes()

    def insert_layer_before(self, layer:tf.keras.layers.Layer, layer_to_add:tf.keras.layers.Layer):
        """
        Inserts a new layer *before* layer
        :param layer:
        :param layer_to_add:
        :return: None
        """

        pred = list(self._graph.predecessors(layer.name))
        self._graph.add_node(layer_to_add.name, **{self.LAYER_KEY_ATTRIBUTE:layer_to_add})
        if pred:
            for i in pred:
                self._graph.remove_edge(i, layer.name)
            for i in pred:
                self._graph.add_edge(i, layer_to_add.name)
            self._graph.add_edge(layer_to_add.name, layer.name)
        else:
            self._graph.add_edge(layer_to_add.name, layer.name)

        # raise NotImplementedError()  # TODO Insert a layer before and connect

    def insert_layer_after(self, layer:tf.keras.layers.Layer, layer_to_add:tf.keras.layers.Layer):
        """
        Inserts a new layer *after* layer
        :param layer:
        :param layer_to_add:
        :return: None
        """

        succ = list(self._graph.successors(layer.name))
        self._graph.add_node(layer_to_add.name, **{self.LAYER_KEY_ATTRIBUTE:layer_to_add})
        if succ:
            for i in succ:
                self._graph.remove_edge(layer.name, i)
            for i in succ:
                self._graph.add_edge(layer_to_add.name, i)
            self._graph.add_edge(layer.name, layer_to_add.name)
        else:
            self._graph.add_edge(layer.name, layer_to_add.name)

        # raise NotImplementedError()  # TODO Insert a new layer after layer and connect those

    def remove_layer(self, layer:tf.keras.layers.Layer):
        """
        Removes the layer from the graph and connects accordingly
        :param layer:
        :return:
        """

        pred = list(self._graph.predecessors(layer.name))
        succ = list(self._graph.successors(layer.name))
        self._graph.remove_node(layer.name)
        for i in pred:
            for j in succ:
                self._graph.add_edge(i, j)

        # raise NotImplementedError()  # TODO Remove a layer and connect the previous to the next

    def draw_network(self, graph):
        nx.drawing.draw_networkx(graph)
        plt.show()

    def convert(self, model: tf.keras.Model) -> nx.DiGraph:
        """
        Convert a Keras model to a networkx graph.
        Model should be Sequential only now.
        :param model:
        :return:
        """
        self._model = model

        #if isinstance(model, tf.keras.Sequential):
        if model.__class__.__name__ == 'Sequential':

            for i in range(len(self._model.layers)):
                layer = self._model.layers[i]
                layer_attr = {self.LAYER_KEY_ATTRIBUTE:layer}
                self._graph.add_node(layer.name, **layer_attr)
                # nx.set_node_attributes(self._graph, layer_attr)
            for i in range(len(self._model.layers) - 1):
                self._graph.add_edge(self._model.layers[i].name, self._model.layers[i+1].name)

        # raise NotImplementedError()  # TODO To convert to nx.DiGraph()
    def save_weights(self):
        for n in self._graph.nodes:
            layer_obj = self._graph.nodes[n][self.LAYER_KEY_ATTRIBUTE]
            old_weights = [w.numpy() for w in layer_obj.weights]
            self.weights_dict[n] = old_weights
            # layer_class = layer_obj.__class__
            # layer_config = layer_obj.get_config()
            # self._graph.nodes[n][self.LAYER_KEY_ATTRIBUTE] = layer_class.from_config(layer_config)

    def assign_old_weights(self):
        topo_sorted = nx.topological_sort(self._graph)
        for i, n in enumerate(topo_sorted):
            prede = list(self._graph.predecessors(n))
            if len(prede) == 0:
                continue
            for p in prede:
                self._graph.nodes[n][self.LAYER_KEY_ATTRIBUTE](self._graph.nodes[p][self.LAYER_KEY_ATTRIBUTE].output)
            for nw, ow in zip(self._graph.nodes[n][self.LAYER_KEY_ATTRIBUTE].weights, self.weights_dict[n]):  # Assign previous weights
                nw.assign(ow)

    def build_model(self) -> tf.keras.Model:  # TODO Not for now!
        """
        Returns a new Keras Model
        :return:
        """
        tf.keras.backend.clear_session()
        # for sequential model
        sorted_nodes = nx.topological_sort(self._graph)
        new_layer_obj_list = []
        # nodes = self.nodes()
        weights_dict = {}
        for n in sorted_nodes:
            layer_obj = self.nodes[n][self.LAYER_KEY_ATTRIBUTE]
            weights_dict[layer_obj.name] = layer_obj.weights
            cfg = layer_obj.get_config()
            new_obj = type(layer_obj).from_config(cfg)
            new_layer_obj_list.append(new_obj)

        new_model = tf.keras.Sequential(new_layer_obj_list)
        new_model(tf.random.uniform([1, *new_model.input_shape[1:]]))

        for k, old_weights in weights_dict.items():
            new_weights = new_model.get_layer(k)
            for ow, nw in zip(old_weights, new_weights.weights):
                nw.assign(ow)

        return new_model

        # raise NotImplementedError()  # TODO build a new model from the current nx.DiGraph (self.graph)

