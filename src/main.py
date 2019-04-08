# coding: utf-8

import os
import sys
import json
import hooks
import logging
import numpy as np
import tensorflow as tf

# import your model file here, make sure model_fn and input_fn exist


# libiomp5.dylib config for mac, needed or kernel would be killed
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# logging config, using level to control different log outputs
logging.basicConfig(format = "[%(levelname)s] - %(filename)s - %(funcName)s " +
                             "- %(message)s",
                    level = logging.DEBUG)

logging.info("defining model_fn and input_fn")
# sepecify your model_fn and input_fn here
model_fn = None
input_fn = None

logging.info("defining running configs.")
params = {
    # estimator defining config
    "num_train_samples"     : 988129,
    "num_valid_samples"     : 3536,
    "num_test_samples"      : 3557,
    "num_users"             : 6040,
    "num_items"             : 3706,
    "batch_size"            : 32,
    "embedding_size"        : 32,
    # other estimator configs according to special model

    # estimator running config
    "num_epochs"            : 2,
    "patience"              : 10,
    "monitor_name"          : "loss",
    "model_dir"             : os.path.join("..", "model"),
    "save_summary_steps"    : 10,
    "save_checkpoints_steps": 10,
    "evaluation_interval"   : 1,
}

logging.info("clearing model path.")
os.system("rm -rf %s" % params["model_dir"])

logging.info("defining run config.")
run_config = tf.estimator.RunConfig(
    model_dir = params["model_dir"],
    save_summary_steps = params["save_summary_steps"],
    save_checkpoints_steps = params["save_checkpoints_steps"])

logging.info("building train estimator.")
train_estimator = tf.estimator.Estimator(
    model_fn  = params["model_fn"],
    model_dir = params["model_dir"],
    config    = run_config)

logging.info("defining train spec.")
train_spec = tf.estimator.TrainSpec(
    input_fn  = lambda: input_fn(tf.estimator.ModeKeys.TRAIN, params),
    max_steps = params["num_train_samples"] / params["batch_size"] *
                params["num_epochs"])

logging.info("defining early stopping hook.")
early_stopping_hook = hooks.EarlyStoppingHook(
    monitor_name = params["monitor_name"],
    patience     = params["patience"])

logging.info("defining eval spec.")
eval_spec = tf.estimator.EvalSpec(
    input_fn = lambda: input_fn(tf.estimator.ModeKeys.EVAL, params),
    hooks    = [early_stopping_hook],
    throttle_secs = params["eval_interval"])

logging.info("training and evaluating.")
tf.estimator.train_and_evaluate(
    train_estimator, train_spec, eval_spec)

logging.info("rebuilding test estimator.")
test_estimator = tf.estimator.Estimator(
    model_fn  = lambda: input_fn(tf.estimator.ModeKeys.PREDICT, params),
    model_dir = params["model_dir"],
    warm_start_from = params["model_dir"])

logging.info("computing metrics.")
# TODO
