import numpy as np
import shutil
import os
import tempfile
import tensorflow as tf


def get_number_of_parameters(model: tf.keras.Model):
    num_of_params = 0
    weights = model.get_weights()
    for w in weights:
        num_of_params += np.prod(w.shape)
    return num_of_params


def get_model_size(model: tf.keras.Model):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "something.h5")
    model.save_weights(path)
    size = os.path.getsize(path)
    shutil.rmtree(temp_dir)
    return size / 1000**2  # The size in Megabytes


def warmup_model(model: tf.keras.Model):
    input_shape = (1, *model.input_shape[1:])
    inp = tf.random.uniform(input_shape)
    model(inp, training=False)
