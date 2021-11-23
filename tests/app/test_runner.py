import pytest
import tensorflow as tf
from typing import Tuple
import time
from dense_optimization.app.runner import Runner

class TestRunner(object):

    @staticmethod
    def build_mnist_model():

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), input_shape=(28, 28, 1)),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3)),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.ReLU(),

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
        # data load
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # data Normalization
        x_train, x_test = tf.cast(x_train, tf.float32) / 255, tf.cast(x_test, tf.float32) / 255
        x_train = tf.reshape(x_train, [-1, 28, 28, 1])
        x_test = tf.reshape(x_test, [-1, 28, 28, 1])
        y_train = tf.one_hot(y_train, 10)
        y_test = tf.one_hot(y_test, 10)
        training_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
        validation_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000).batch(32)

        return training_dataset, validation_dataset


    def test_runner(self):
        print()
        model = self.build_mnist_model()
        metric_fn = tf.keras.metrics.CategoricalAccuracy()
        training_dataset, validation_dataset = self.create_datasets()

        runner = Runner()

        new_model = runner.run(model=model, training_dataset=training_dataset, validation_dataset=validation_dataset,
                               steps_for_evaluation=1, metric_fn=metric_fn)

        assert new_model is not None
        # assert len(new_model.layers) == 6


