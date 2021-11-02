import pytest
import tensorflow as tf

from typing import Tuple

from dense_optimization.app.runner import Runner

class TestRunner(object):

    @staticmethod
    def build_mnist_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), input_shape=(28, 28, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3)),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(10),
        ])
        return model

    @staticmethod
    def create_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        # TODO Return Training Dataset and Validation Dataset
        return (None, None)

    def test_runner(self):
        model = self.build_mnist_model()

        metric_fn = tf.keras.metrics.CategoricalAccuracy()

        training_dataset, validation_dataset = self.create_datasets()
        runner = Runner()

        new_model = runner.run(model=model, training_dataset=training_dataset, validation_dataset=validation_dataset,
                               steps_for_evaluation=100, metric_fn=metric_fn)
        assert new_model is not None
        assert len(new_model.layers) == 6




