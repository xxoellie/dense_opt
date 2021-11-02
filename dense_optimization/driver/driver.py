import tensorflow as tf

from typing import Dict, Any


class Driver(object):

    def __init__(self):
        pass

    def evaluate(self,
                 model: tf.keras.Model,
                 dataset: tf.data.Dataset,
                 steps: int,
                 metric_fn: tf.keras.metrics.Metric) -> Dict[str, Any]:
        """

        :param model: Compiled model
        :param dataset: Dataset
        :param steps: Number of steps to run evaluation on
        :param metric_fn: the metric to measure performance of the model
        :return: Results of the evaluation from Model
        """
        raise NotImplementedError()
