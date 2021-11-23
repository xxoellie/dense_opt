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

    @staticmethod
    def get_some_examples_from_dataset(dataset, num_examples):
        examples = []
        for i, data in enumerate(dataset):
            if i == num_examples:
                break
            examples.append(data)
        return examples

    def run(self,
            model: tf.keras.Model,
            training_dataset: tf.data.Dataset,
            validation_dataset: tf.data.Dataset,
            steps_for_evaluation: int,
            metric_fn: tf.keras.metrics.Metric):

        # This is to make sure we're consistent and checking on the same examples in both "evaluations" of Before&After
        validation_examples = self.get_some_examples_from_dataset(dataset=validation_dataset,
                                                                  num_examples=steps_for_evaluation)

        results_before_folding = self._driver.evaluate(model=model,
                                                       dataset=validation_examples,
                                                       metric_fn=metric_fn)

        self._network.convert(model=model)

        bn_transform = FoldBatchNormTransform()
        bn_transform(network=self._network)

        new_model = self._network.build_model()

        results_after_folding = self._driver.evaluate(model=new_model,
                                                      dataset=validation_examples,
                                                      metric_fn=metric_fn)

        is_close = np.allclose(results_before_folding['predicted output'].numpy(), results_after_folding['predicted output'].numpy(), atol=1e-4)
        assert is_close, "Did not manage to Fold BatchNorms successfully :("

        return new_model