# coding: utf-8
import logging
import numpy as np
import tensorflow as tf

class EarlyStoppingHook(tf.train.SessionRunHook):
    """Build EarlyStoppingHook inherited from tf.train.SessionRunHook,
    which is to be used for evaluation_hooks in tf.estimatro.EstimatorSpec.
    This implementation is based on EarlyStopping in keras.callbacks.
    """
    def __init__(self,
                 monitor_name = "loss",
                 min_delta = 0.001,
                 patience = 10,
                 mode = "min"):
        """Initialization.

        Args:
            monitor_name: string, op name in graph, denoting quality to be
                monitored.
            min_delta: float, minimum change in the monitored quality
                to qualify as an improvement.
            patience: int, numbers of steps with no improvement after
                which will be stopped.
            mode: enum{min, max}, denoting the monitred quality
                should increase or decrease when measuring it.
        """
        self.monitor_name = monitor_name
        self.min_delta = min_delta
        self.patience = patience
        # define monitor_op
        if mode == "min":
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == "max":
            self.monitor_op = np.greater
        self.wait = 1
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def begin(self):
        """Called once before using the session.
        """
        graph = tf.get_default_graph()
        self.monitor = graph.get_operation_by_name(self.monitor_name)

    def before_run(self, run_context):
        """Called before each call to run().
        """
        self.element = self.monitor.outputs[0]
        return tf.train.SessionRunArgs([self.element])

    def after_run(self, run_context, run_values):
        """Called after each call to run().
        """
        current = run_values.results[0]
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 1
        else:
            self.wait += 1
            if self.wait > self.patience:
                run_context.request_stop()
                raise ValueError("Early stopping occurred.")
