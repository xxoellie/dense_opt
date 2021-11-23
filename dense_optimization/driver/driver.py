import tensorflow as tf

from typing import Dict, Any, List


class Driver(object):

    def evaluate(self,
                 model: tf.keras.Model,
                 dataset: List,
                 metric_fn: tf.keras.metrics.Metric) -> Dict[str, Any]:
        """

        :param model: Compiled model
        :param dataset: Dataset
        :param steps: Number of steps to run evaluation on
        :param metric_fn: the metric to measure performance of the model
        :return: Results of the evaluation from Model
        """
        # return {
        #     metric_fn.name: VALUE
        # }
        result_dic = {}
        for i, data in enumerate(dataset):
            x, y = data
            pred = model(x, training=False)
            loss = metric_fn(y, pred)
            result_dic[f'{metric_fn.name}'] = loss
            result_dic['predicted output'] = pred

        return result_dic
        #if metric_fn is several
        # metric_fns: [metric_fn1, metric_fn2...]
        # for train_data in dataset:
        #     x, y = train_data
        #     pred = model(x)
        #     for metric_fn in metric_fns:
        #         loss = metric_fn(y, pred)
        #         loss_dic[f'{metric_fn.name}'] = loss
        #
        # return loss_dic
        # raise NotImplementedError()
