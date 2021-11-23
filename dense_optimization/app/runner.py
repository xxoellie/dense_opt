import numpy as np
import tensorflow as tf

from dense_optimization.driver.driver import Driver
from dense_optimization.network.network import Network
from dense_optimization.transforms.fold_bn import FoldBatchNormTransform
from typing import Callable
import time

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

        model = self._network.assign_weights(model)

        results_before_folding = self._driver.evaluate(model=model,
                                                       dataset=validation_dataset,
                                                       steps=steps_for_evaluation,
                                                       metric_fn=metric_fn)

        self._network.convert(model=model)

        # TODO Fold BatchNorm

        bn_transform = FoldBatchNormTransform()
        bn_transform(network=self._network)

        new_model = self._network.build_model()

        results_after_folding = self._driver.evaluate(model=new_model,
                                                      dataset=validation_dataset,
                                                      steps=steps_for_evaluation,
                                                      metric_fn=metric_fn)

        print("output_comparison: ", end='')
        print(np.allclose(results_before_folding['predicted output'].numpy(), results_after_folding['predicted output'].numpy(), atol=1e-2))

        return new_model