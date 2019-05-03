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
import GMF
import MLP
import NeuMF
import WideDeep
import ProAttn

# libiomp5.dylib config for mac in jupyter notebook,
# need or kernel would be killed
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# logging config, using level to control different log outputs
logging.basicConfig(format = "[%(levelname)s] - %(filename)s - %(funcName)s " +
                             "- %(message)s",
                    level = logging.INFO)

logging.info("defining model_fn and input_fn")
# sepecify your model_fn and input_fn here
# model_fn = GMF.model_fn
# input_fn = GMF.input_fn
# model_fn = MLP.model_fn
# input_fn = MLP.input_fn
model_fn = NeuMF.model_fn
input_fn = NeuMF.input_fn
# model_fn = WideDeep.model_fn
# input_fn = WideDeep.input_fn
# model_fn = ProAttn.model_fn
# input_fn = ProAttn.input_fn

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
    "num_epochs"            : 2,
    "patience"              : 10,
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

logging.info("building train estimator.")
train_estimator = tf.estimator.Estimator(
    model_fn  = model_fn,
    model_dir = params["model_dir"],
    params    = params,
    config    = run_config)

logging.info("training model.")
train_estimator.train(
    input_fn = lambda: input_fn(tf.estimator.ModeKeys.TRAIN, params),
    max_steps = max_steps)

# logging.info("computing total_evaluation nums.")
# total_eval_nums = math.ceil(max_steps / params["evaluation_interval"])

# logging.info("training and testing every {} steps with early stopping.".format(
#              params["evaluation_interval"]))
# best, wait = 0, 0
# for i in range(total_eval_nums):
#     logging.info("training with {}/{} -th steps.".format(i, total_eval_nums))
#     estimator.train(
#         input_fn  = lambda: input_fn(tf.estimator.ModeKeys.TRAIN, params),
#         steps = params["evaluation_interval"])
#
#     logging.info("evaluating with current model parameters.")
#     predictions = estimator.predict(
#         input_fn = lambda: input_fn(tf.estimator.ModeKeys.EVAL, params))
#
#     logging.info("evaluating with evaluation samples.")
#     # _, NDCG_at_20 = evals.evaluate_at_K(predictions, 20)
#     # logging.info("evaluation with NDCG_at_20: {}.".format(NDCG_at_20))
#     HR_at_20,_ = evals.evaluate_at_K(predictions, 20)
#     logging.info("evaluation with HR_at_20: {}, best: {}.".format(HR_at_20, best))
#     logging.info("testing and updating early stopping conditions.")
#     # if NDCG_at_20 > best + 0.0001:
#     #     best, wait = NDCG_at_20, 0
#     if HR_at_20 > best + 0.00001:
#         best, wait = HR_at_20, 0
#     else:
#         wait += 1
#         logging.info("early stopping wait: {}".format(wait))
#         if wait >= params["patience"]:
#             logging.info("early stopping occured.")
#             break

# ========================= testing part =========================
logging.info("refreshing batch size.")
params["batch_size"] = 100

logging.info("rebuilding test estimator.")
test_estimator = tf.estimator.Estimator(
    model_fn  = model_fn,
    model_dir = params["model_dir"],
    params    = params,
    config    = run_config)

logging.info("predicting with prediction samples.")
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
    print("HR@{}: {},\tNDCG@{}: {}".format(K, round(HR, 5), K, round(NDCG, 5)))
print("\n")
