# coding: utf-8

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

__all__ = ["NeuCF_movielens1m_input_fn", "NeuCF_model_fn"]

def NeuCF_movielens1m_input_fn(mode, params):
    """Building input_fn for tf.estimator.Estimator instances with MovieLens 1M.

    Args:
        mode: tf.estimator.ModeKeys instance, TRAIN, EVAL or PREDICT.
        params: dict, needed "batch_size" key.

    Returns:
        A tf.data.Dataset instance containing ((user_id, item_id), label) pair.
    """
    logging.info("looking for data file.")
    data_dir = os.path.join("..", "data")
    if mode == tf.estimator.ModeKeys.TRAIN:
        data_file = os.path.join(data_dir,
            "movielens.1m.train.20190408.idx.implicit.csv")
        logging.debug("data_file: %s" % data_file)
        logging.info("reading data according to datafile.")
        data = pd.read_csv(data_file)
        users, items, labels = (
            data["UserID"].values, data["MovieID"].values, data["Label"].values)
        logging.info("generating tf.data.Dataset instance.")
        features_dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(users.astype(np.int64)),
            tf.data.Dataset.from_tensor_slices(items.astype(np.int64))))
    else:
        if mode == mode == tf.estimator.ModeKeys.EVAL:
            data_file = os.path.join(data_dir,
                "movielens.1m.valid.20190408.idx.implicit.csv")
        else:
            data_file = os.path.join(data_dir,
                "movielens.1m.test.20190408.idx.implicit.csv")
        logging.debug("data_file: %s" % data_file)
        logging.info("reading data according to datafile.")
        data = pd.read_csv(data_file)
        users, items, labels, samples = (
            data["UserID"].values, data["MovieID"].values,
            data["Label"].values , data["Samples"].values)
        logging.info("generating feature dataset.")
        features_dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(users.astype(np.int64)),
            tf.data.Dataset.from_tensor_slices(items.astype(np.int64)),
            tf.data.Dataset.from_tensor_slices(
                np.array([sample.split(" ")
                          for sample in samples]).astype(np.int64))))
    logging.info("generating label dataset.")
    labels_dataset = tf.data.Dataset.from_tensor_slices(
        labels.astype(np.float64))
    logging.info("merging features and labels.")
    dataset =  tf.data.Dataset.zip((features_dataset, labels_dataset))
    logging.info("shuffling, batching, prefetching of dataset.")
    dataset = (
        dataset.shuffle(256)
               .batch(batch_size = params["batch_size"]
                      if mode == tf.estimator.ModeKeys.TRAIN else len(users),
                      drop_remainder = True)
               .prefetch(params["batch_size"])
    )
    return dataset.repeat() if mode == tf.estimator.ModeKeys.TRAIN else dataset

def NeuCF_model_fn(features, labels, mode, params):
    """Building model_fn for tf.estimator.Estimator instances with NeuCF model.

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
    if mode == tf.estimator.ModeKeys.TRAIN:
        # shape of users: [batch_size, 1], denoting user ids
        # shape of items: [batch_size, 1], denoting item ids
        users, items = features
    else:
        # shape of samples: [batch_size, 100]
        users, items, samples = features

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

    logging.info("defining MLP componet.")
    MLP_output = None

    logging.info("defining NeuCF model.")
    GMF_MLP_concat = tf.concat(GMF_output, MLP_output)

    logging.info("defining output layer.")
    preds = tf.layers.Dense(
        units = 1,
        activation = tf.nn.sigmoid,
        use_bias = True)(GMF_MLP_concat)

    logging.info("defining loss.")
    if (mode == tf.estimator.ModeKeys.TRAIN or
        mode == tf.estimator.ModeKeys.EVAL):
        loss = None
    else:
        loss = None

    logging.info("defining training op.")
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer().minimize(
            loss, tf.train.get_or_create_global_step())
    else:
        train_op = None

    logging.info("defining predictions.")
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = preds
    else:
        predictions = None

    logging.info("returning estimator spec.")
    return tf.estimator.EstimatorSpec(
      mode = mode,
      predictions = predictions,
      loss = loss,
      train_op = train_op)
