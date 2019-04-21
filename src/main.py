# coding: utf-8

import os
import sys
import json
import evals
import hooks
import logging
import numpy as np
import tensorflow as tf

# import your model file here, make sure model_fn and input_fn exist
import NeuMF

# libiomp5.dylib config for mac, needed or kernel would be killed
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# logging config, using level to control different log outputs
logging.basicConfig(format = "[%(levelname)s] - %(filename)s - %(funcName)s " +
                             "- %(message)s",
                    level = logging.DEBUG)

logging.info("defining model_fn and input_fn")
# sepecify your model_fn and input_fn here
model_fn = NeuMF.model_fn
input_fn = NeuMF.input_fn

logging.info("defining running configs.")
params = {
    # estimator defining config
    "num_train_samples"     : 988129,
    "num_valid_samples"     : 6040,
    "num_test_samples"      : 355700,
    "num_neg_samples"       : 3,
    "num_users"             : 6041,
    "num_items"             : 3953,
    "batch_size"            : 32,
    "embedding_size"        : 32,

    # other estimator configs according to special model
    # NeuMF special configs
    "hidden_layers"         : 2,   # more than 1, or exception will be raised
    "hidden_units"          : 128,

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
    model_fn  = model_fn,
    model_dir = params["model_dir"],
    params    = params,
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
params["batch_size"] = 100
test_estimator = tf.estimator.Estimator(
    model_fn  = model_fn,
    model_dir = params["model_dir"],
    params    = params,
    warm_start_from = params["model_dir"])

logging.info("predicting.")
predictions = test_estimator.predict(
    input_fn = lambda: input_fn(tf.estimator.ModeKeys.PREDICT, params))

logging.info("computing NDCG and HR metrics with K in range(1, 11).")
for K in range(1, 11):
    HR, NDCG = evals.evaluate_at_K(predictions, K)
    print("HR@{}: {}, NDCG@{}: {}\n".format(K, HR, K, NDCG))
