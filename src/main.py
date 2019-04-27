# coding: utf-8

import os
import sys
import json
import math
import evals
import logging
import numpy as np
import tensorflow as tf

# import your model file here, make sure model_fn and input_fn exist
import NeuMF
import Wide_Deep

# libiomp5.dylib config for mac, need or kernel would be killed
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# logging config, using level to control different log outputs
logging.basicConfig(format = "[%(levelname)s] - %(filename)s - %(funcName)s " +
                             "- %(message)s",
                    level = logging.INFO)

logging.info("defining model_fn and input_fn")
# sepecify your model_fn and input_fn here
# model_fn = NeuMF.model_fn
# input_fn = NeuMF.input_fn
model_fn = Wide_Deep.model_fn
input_fn = Wide_Deep.input_fn

logging.info("defining running configs.")
params = {
    # estimator defining config
    "num_neg_samples"       : 1,
    "num_users"             : 6040,
    "num_items"             : 3953,
    "num_genders"           : 2,
    "num_ages"              : 8,
    "num_occupations"       : 22,
    "num_zipcodes"          : 3440,
    "num_years"             : 82,
    "num_genres"            : 18,
    "batch_size"            : 128,
    "embedding_size"        : 16,

    # other estimator configs according to special model
    # using for NeuMF and Wide&Deep config
    "num_factors"           : 64,

    # optimizer config
    "SGD_init_lr"           : 0.1,
    "SGD_decay_steps"       : 100,
    "SGD_decay_rate"        : 0.96,

    # estimator running config
    "num_epochs"            : 4,
    "patience"              : 4,
    "model_dir"             : os.path.join("..", "model"),
    "save_summary_steps"    : 1,
    "save_checkpoints_steps": 1,
    "evaluation_interval"   : 5,
}

# ========================= training part =========================

logging.info("clearing model path.")
os.system("rm -rf %s" % params["model_dir"])     # attention !!!

logging.info("defining run config.")
run_config = tf.estimator.RunConfig(
    model_dir = params["model_dir"],
    save_summary_steps = params["save_summary_steps"],
    save_checkpoints_steps = params["save_checkpoints_steps"])

logging.info("computing max steps.")
max_steps = int(params["num_users"] * (1 + params["num_neg_samples"]) /
            params["batch_size"] * params["num_epochs"])

logging.info("computing total_evaluation nums.")
total_eval_nums = math.ceil(max_steps / params["evaluation_interval"])

logging.info("building train estimator.")
estimator = tf.estimator.Estimator(
    model_fn  = model_fn,
    model_dir = params["model_dir"],
    params    = params,
    config    = run_config)

logging.info("training and testing every {} steps with early stopping.".format(
             params["evaluation_interval"]))
best, wait = 0, 0
for i in range(total_eval_nums):
    logging.info("training with {}/{} -th steps.".format(i, total_eval_nums))
    estimator.train(
        input_fn  = lambda: input_fn(tf.estimator.ModeKeys.TRAIN, params),
        steps = params["evaluation_interval"])

    logging.info("evaluating with current model parameters.")
    predictions = estimator.predict(
        input_fn = lambda: input_fn(tf.estimator.ModeKeys.EVAL, params))

    logging.info("evaluating with evaluation samples.")
    _, NDCG_at_10 = evals.evaluate_at_K(predictions, 10)
    logging.info("evaluation with NDCG_at_10: {}.".format(NDCG_at_10))
    # HR_at_10,_ = evals.evaluate_at_K(predictions, 10)
    # logging.info("evaluation with HR_at_10: {}.".format(HR_at_10))
    logging.info("testing and updating early stopping conditions.")
    if NDCG_at_10 > best + 0.0001:
        best, wait = NDCG_at_10, 0
    else:
        wait += 1
        logging.info("early stopping wait: {}".format(wait))
        if wait >= params["patience"]:
            logging.info("early stopping occured.")
            break

# ========================= testing part =========================

logging.info("predicting with prediction samples.")
predictions = estimator.predict(
    input_fn = lambda: input_fn(tf.estimator.ModeKeys.PREDICT, params))

logging.info("storing predictions as a list.")
# this step is important, because predictions acually is a generator,
# we will get none samples when we run into the second loop, unless
# we do predicting again, which is inefficient.
samples = []
for sample in predictions:
    samples.append(sample)

logging.info("computing NDCG and HR metrics with K in range(1, 11).")
print("\nStarting to compute metrics...\n")
for K in range(1, 11):
    HR, NDCG = evals.evaluate_at_K(samples, K)
    print("HR@{}: {},\tNDCG@{}: {}".format(K, HR, K, NDCG))
print("\n")
