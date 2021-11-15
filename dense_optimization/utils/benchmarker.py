import tensorflow as tf
from time import perf_counter
from tensorflow.keras.utils import Progbar


def benchmark_model(model: tf.keras.Model, num_steps: int = 10):
    input_shape = (1, *model.input_shape[1:])

    # Warmup
    model(tf.random.uniform(shape=input_shape))

    times = []
    pbar = Progbar(target=num_steps, unit_name="benchmarking")
    for i in range(1, num_steps+1):
        inp = tf.random.uniform(shape=input_shape, minval=-1, maxval=1, dtype=tf.float32)
        s = perf_counter()
        model(inp)
        e = perf_counter()
        delta = e-s
        times.append(delta)
        pbar.update(current=i)
    return times
