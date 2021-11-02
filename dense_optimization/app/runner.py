import tensorflow as tf

from dense_optimization.driver.driver import Driver
from dense_optimization.network.network import Network
from dense_optimization.transforms import fold_bn
from typing import Callable


class Runner(object):

    def __init__(self):
        self._driver = Driver()
        self._network = Network()

    def run(self,
            model: tf.keras.Model,
            training_dataset: tf.data.Dataset,
            validation_dataset: tf.data.Dataset,
            steps_for_evaluation: int,
            metric_fn: tf.keras.metrics.Metric):
        results_before_folding = self._driver.evaluate(model=model,
                                                       dataset=validation_dataset,
                                                       steps=steps_for_evaluation,
                                                       metric_fn=metric_fn)
        print(results_before_folding)
        self._network.convert(model=model)

        results_after_folding = self._driver.evaluate(model=model,
                                                      dataset=validation_dataset,
                                                      steps=steps_for_evaluation,
                                                      metric_fn=metric_fn)
        print(results_after_folding)
