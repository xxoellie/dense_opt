import tensorflow as tf
from time import perf_counter
from tensorflow.keras.utils import Progbar


def _benchmark_model(func_to_run, num_steps):

    times = []
    pbar = Progbar(target=num_steps, unit_name="benchmarking")
    for i in range(1, num_steps+1):
        s = perf_counter()
        func_to_run()
        e = perf_counter()
        delta = e-s
        times.append(delta)
        pbar.update(current=i)
    return times


def benchmark_inference(model: tf.keras.Model, num_steps: int):
    input_shape = (1, *model.input_shape[1:])
    model(tf.random.uniform(shape=input_shape))
    inp = tf.random.uniform(shape=input_shape, minval=-1, maxval=1, dtype=tf.float32)

    def _run_inference():
        model(inp, training=False)

    return _benchmark_model(func_to_run=_run_inference,
                            num_steps=num_steps)


def benchmark_training(model: tf.keras.Model, num_steps: int):
    input_shape = (1, *model.input_shape[1:])
    output_shape = (1, *model.output_shape[1:])
    model(tf.random.uniform(shape=input_shape))
    inp = tf.random.uniform(shape=input_shape, minval=-1, maxval=1, dtype=tf.float32)
    out = tf.random.uniform(shape=output_shape, minval=-1, maxval=1, dtype=tf.float32)
    opt = tf.keras.optimizers.SGD()

    def _run_train_step():
        with tf.GradientTape() as tape:
            y_true = model(inp, training=False)
            loss = tf.keras.losses.mean_absolute_error(y_true=y_true, y_pred=out)
        opt.minimize(loss=loss, var_list=model.trainable_variables, tape=tape)

    return _benchmark_model(func_to_run=_run_train_step,
                            num_steps=num_steps)
