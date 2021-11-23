import shutil

import numpy as np
import tensorflow as tf
import os
import tempfile

from dense_optimization.driver.driver import Driver
from dense_optimization.network.network import Network
from dense_optimization.transforms.fold_bn import FoldBatchNormTransform


from dense_optimization.utils import benchmark_inference, benchmark_training, visualize_model, get_model_size, get_number_of_parameters, warmup_model
from typing import Callable
import time


class Runner(object):

    def __init__(self):
        self._driver = Driver()
        self._network = Network()
        self._temp_dir = tempfile.mkdtemp()
        self._model_before_path = os.path.join(self._temp_dir, "model_before")
        self._model_after_path = os.path.join(self._temp_dir, "model_after")

    @staticmethod
    def get_some_examples_from_dataset(dataset, num_examples):
        examples = []
        for i, data in enumerate(dataset):
            if i == num_examples:
                break
            examples.append(data)
        return examples

    def do_pretty_stuff(self):
        model_before = tf.keras.models.load_model(self._model_before_path)
        warmup_model(model=model_before)
        model_after = tf.keras.models.load_model(self._model_after_path)
        warmup_model(model=model_after)

        output_dir = self._temp_dir  # TODO Choose Output Dir
        assert output_dir is not None
        num_steps = 5

        inference_times_before = benchmark_inference(model=model_before, num_steps=num_steps)
        training_times_before = benchmark_training(model=model_before, num_steps=num_steps)
        visualize_model(model=model_before, prefix="model_before", output_dir=output_dir)
        model_size_before_megabytes = get_model_size(model=model_before)
        num_params_before = get_number_of_parameters(model=model_before)

        inference_times_after = benchmark_inference(model=model_after, num_steps=num_steps)
        training_times_after = benchmark_training(model=model_after, num_steps=num_steps)
        visualize_model(model=model_after, prefix="model_after", output_dir=output_dir)
        model_size_after_megabytes = get_model_size(model=model_after)
        num_params_after = get_number_of_parameters(model=model_after)

        # TODO Plot, graph, however you'd want.

    def run(self,
            model: tf.keras.Model,
            training_dataset: tf.data.Dataset,
            validation_dataset: tf.data.Dataset,
            steps_for_evaluation: int,
            metric_fn: tf.keras.metrics.Metric):
        model.save(filepath=self._model_before_path)
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

        new_model.save(filepath=self._model_after_path)
        self.do_pretty_stuff()

        if os.path.exists(self._temp_dir):  # To remove the temporary directory
            shutil.rmtree(self._temp_dir)

        return new_model