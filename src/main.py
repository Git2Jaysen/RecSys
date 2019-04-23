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
import WideDeep

# libiomp5.dylib config for mac, needed or kernel would be killed
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# logging config, using level to control different log outputs
logging.basicConfig(format = "[%(levelname)s] - %(filename)s - %(funcName)s " +
                             "- %(message)s",
                    level = logging.INFO)

logging.info("defining model_fn and input_fn")
# sepecify your model_fn and input_fn here
model_fn = NeuMF.model_fn
input_fn = NeuMF.input_fn
# model_fn = WideDeep.model_fn
# input_fn = WideDeep.input_fn

logging.info("defining running configs.")
params = {
    # estimator defining config
    "num_train_samples"     : 988129,
    "num_eval_samples"      : 6040,
    "num_pred_samples"      : 355700,
    "num_neg_samples"       : 1,
    "num_users"             : 6041,
    "num_items"             : 3953,
    "batch_size"            : 128,
    "embedding_size"        : 16,

    # optimizer config
    "SGD_init_lr"           : 0.1,
    "SGD_decay_steps"       : 1000,
    "SGD_decay_rate"        : 0.96,

    # other estimator configs according to special model
    # using for NeuMF and Wide&Deep config
    "num_factors"           : 8,

    # other configs using for model of ultilizing profiles
    "num_genders"           : 2,
    "num_ages"              : 8,
    "num_occupations"       : 22,
    "num_zipcodes"          : 3440,
    "num_years"             : 82,
    "num_genres"            : 20,

    # estimator running config
    "num_epochs"            : 2,
    "patience"              : 100,
    "monitor_name"          : "log_loss/value",
    "model_dir"             : os.path.join("..", "model"),
    "save_summary_steps"    : 10,
    "save_checkpoints_steps": 10,
    "evaluation_interval"   : 10,
}

# ========================= training part =========================

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
    max_steps = int(params["num_train_samples"] *       # attention !!!
                    (1 + params["num_neg_samples"]) /
                    params["batch_size"] * params["num_epochs"]))
    # max_steps = int(params["num_eval_samples"] *        # using for debug
    #                 (1 + params["num_neg_samples"]) /   # make sure generator in
    #                 params["batch_size"] * params["num_epochs"])) # input_fn
    #                                                               # changed.

logging.info("defining early stopping hook.")
early_stopping_hook = hooks.EarlyStoppingHook(
    monitor_name = params["monitor_name"],
    patience     = params["patience"])

logging.info("defining eval spec.")
eval_spec = tf.estimator.EvalSpec(
    input_fn = lambda: input_fn(tf.estimator.ModeKeys.EVAL, params),
    # hooks    = [early_stopping_hook],    # early stopping setting
    throttle_secs = params["evaluation_interval"])

logging.info("training and evaluating.")
try:
    tf.estimator.train_and_evaluate(
        train_estimator, train_spec, eval_spec)
except ValueError:
    logging.info("early stopping occurred.")

# ========================= testing part =========================

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

logging.info("storing predictions as a list.")
# this step is important, because predictions acually is a generator,
# we will get none samples when we run into the second loop, unless
# we do predicting again, which is inefficient.
samples = []
for sample in predictions:
    samples.append(sample)

logging.info("computing NDCG and HR metrics with K in range(1, 51).")
print("\nStarting to compute metrics...\n")
for K in range(1, 51):
    HR, NDCG = evals.evaluate_at_K(samples, K)
    print("HR@{}: {},\tNDCG@{}: {}".format(K, HR, K, NDCG))
print("\n")
