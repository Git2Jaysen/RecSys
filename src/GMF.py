# coding: utf-8

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_ranking as tfr

__all__ = ["input_fn", "model_fn"]

# train_file = "movielens1m.pointwise.trn.988129n.implicit.csv"
train_file = "movielens1m.pointwise.trn.994169n.implicit.csv"
eval_file  = "movielens1m.pointwise.val.353600n.implicit.csv"
# eval_file  = "movielens1m.pointwise.val.176800n.implicit.csv"
# eval_file  = "movielens1m.pointwise.val.70720n.implicit.csv"
pred_file  = "movielens1m.pointwise.prd.355700n.implicit.csv"

def _train_generator(params):
    """Yielding samples one by one of MovieLens 1M dataset for training, with
    the following info:

        ((UserID, MovieID), Label).

    Args:
        params: dict, need "num_neg_samples" key.

    Returns:
        Samples include the info mentioned above.
    """
    logging.info("reading data from data file.")
    data = pd.read_csv(os.path.join("..", os.path.join("data", train_file)))
    logging.debug("data columns: {}".format(" ".join(data.columns)))
    logging.info("yielding training samples with negative sampling.")
    users = list(set(data["UserID"].values.tolist()))
    for u in users:
        user_data = data[data["UserID"] == u]
        user_pos_data = user_data[user_data["Label"] == 1]
        user_neg_data = user_data[user_data["Label"] == 0]
        for i in user_pos_data.index:
            yield ((user_pos_data.loc[i, "UserID"],
                       user_pos_data.loc[i, "MovieID"]), 1)
            if len(user_neg_data) > 0:
                user_neg_samples = user_neg_data.sample(
                    params["num_neg_samples"], replace=True, random_state=42)
                user_neg_samples.reset_index(inplace=True)
                for j in user_neg_samples.index:
                    yield ((user_neg_samples.loc[j, "UserID"],
                                    user_neg_samples.loc[j, "MovieID"]), 0)

def _eval_generator(params):
    """Yielding samples one by one of MovieLens 1M dataset for evaluating, with
    the following info:

        ((UserID, MovieID), Label).

    Args:
        params: dict, unsed in this function.

    Returns:
        Samples include the info mentioned above.
    """
    logging.info("reading data from data file.")
    data = pd.read_csv(os.path.join("..", os.path.join("data", eval_file)))
    logging.info("yielding evaluation samples.")
    for i in data.index:
        yield ((data.loc[i, "UserID"], data.loc[i, "MovieID"]),
                   data.loc[i, "Label"])

def _pred_generator(params):
    """Yielding samples one by one of MovieLens 1M dataset for predicting, with
    the following info:

        ((UserID, MovieID), Label).

    Args:
        params: dict, unsed in this function.

    Returns:
        Samples include the info mentioned above.
    """
    logging.info("reading data from data file.")
    data = pd.read_csv(os.path.join("..", os.path.join("data", pred_file)))
    logging.info("yielding prediction samples.")
    for i in data.index:
        yield ((data.loc[i, "UserID"], data.loc[i, "MovieID"]),
                data.loc[i, "Label"])

def input_fn(mode, params):
    """Building input_fn for tf.estimator.Estimator instances with MovieLens 1M.

    Args:
        mode: tf.estimator.ModeKeys instance, TRAIN, EVAL or PREDICT.
        params: dict, needed "batch_size" key.

    Returns:
        A tf.data.Dataset instance.
    """
    logging.info("generating dataset.")
    if mode == tf.estimator.ModeKeys.TRAIN:
        generator = _train_generator
        # generator = _eval_generator
    elif mode == tf.estimator.ModeKeys.EVAL:
        generator = _eval_generator
    else:
        generator = _pred_generator
    dataset = tf.data.Dataset.from_generator(
        generator = lambda: generator(params),
        output_types = ((tf.int64, tf.int64), tf.int64))
    logging.info("shuffing dataset.")
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(params["batch_size"] * 1000, seed=2019)
    dataset = (dataset.batch(batch_size = params["batch_size"],
                             drop_remainder = True)
                      .prefetch(params["batch_size"]))
    return dataset.repeat() if mode == tf.estimator.ModeKeys.TRAIN else dataset

def model_fn(features, labels, mode, params):
    """Building model_fn for tf.estimator.Estimator instances with NeuMF model.

    Args:
        features: the first item returned from the input_fn.
        labels  : the second item returned from the input_fn.
        mode    : specifies if this training, evaluation or prediction.
        params  : model configs.

    Returns:
        A tf.estimator.EstimatorSpec instance.
    """
    logging.info("parsing features.")
    # shape of labels: [batch_size]
    # shape of users: [batch_size, 1], denoting user ids
    # shape of items: [batch_size, 1], denoting item ids
    users, items = features

    logging.info("defining user and item embedding lookup table.")
    # shape: [num_users, embedding_size]
    user_embedding = tf.Variable(
        tf.truncated_normal([params["num_users"] + 1, params["embedding_size"]],
                            dtype=tf.float64),
        dtype = tf.float64,
        name = "user_embedding"
    )
    # shape: [num_items, embedding_size]
    item_embedding = tf.Variable(
        tf.truncated_normal([params["num_items"] + 1, params["embedding_size"]],
                            dtype=tf.float64),
        dtype = tf.float64,
        name = "item_embedding"
    )

    logging.info("looking up user and item embedding.")
    # shape: [batch_size, embedding_size]
    user_emb_inp = tf.nn.embedding_lookup(user_embedding, users)
    # shape: [batch_size, embedding_size]
    item_emb_inp = tf.nn.embedding_lookup(item_embedding, items)

    logging.info("defining GMF component.")
    # shape: [batch_size, embedding_size]
    GMF_output = user_emb_inp * item_emb_inp

    logging.info("reshaping GMF output.")
    # a bug of tf, if not doing this, a error will be raised
    GMF_output = tf.reshape(GMF_output, shape = [-1, params["embedding_size"]])

    logging.info("defining output layer.")
    # shape: [batch_size, 1]
    logits = tf.layers.Dense(
        units = 1,
        activation = tf.nn.sigmoid,
        use_bias = False)(GMF_output)

    logging.info("reshaping logits.")
    # shape: [batch_size, ]
    logits = tf.reshape(logits, shape = [-1, ], name = "logits")

    logging.info("defining loss.")
    if (mode == tf.estimator.ModeKeys.TRAIN or
        mode == tf.estimator.ModeKeys.EVAL):
        loss = tf.losses.log_loss(labels = labels, predictions = logits)
    else:
        loss = None

    logging.info("defining training op.")
    if mode == tf.estimator.ModeKeys.TRAIN:
        logging.info("creating global step.")
        global_step = tf.train.get_or_create_global_step()

        # logging.info("using SGD optimizer.")
        # learning_rate = tf.train.exponential_decay(
        #     learning_rate = params["SGD_init_lr"],
        #     global_step   = global_step,
        #     decay_steps   = params["SGD_decay_steps"],
        #     decay_rate    = params["SGD_decay_rate"],
        #     staircase     = True)
        # train_op = tf.train.GradientDescentOptimizer(learning_rate) \
        #                    .minimize(loss = loss, global_step = global_step)

        logging.info('using Adam optimizer.')
        train_op = tf.train.AdamOptimizer().minimize(loss = loss,
                                                     global_step = global_step)
    else:
        train_op = None

    logging.info("defining predictions.")
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = logits
    else:
        predictions = None

    logging.info("returning estimator spec.")
    return tf.estimator.EstimatorSpec(
      mode = mode,
      predictions = predictions,
      loss = loss,
      train_op = train_op)
