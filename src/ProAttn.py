# coding: utf-8

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

# encoder Movie Genres to binarize form, i.e. suppose we have 18 genres, then a
# sample with [1, 2] will be convert to [1, 1, 0, 0, ...]
from sklearn.preprocessing import MultiLabelBinarizer
encoder = MultiLabelBinarizer()
encoder.fit([[i] for i in range(18)])

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

        ((UserID, MovieID, Gender, Age, Occupation,
          Zip-code, Year, Genres(bin)), Label).

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
        logging.debug("computing user data according to user id.")
        user_data = data[data["UserID"] == u]
        user_pos_data = user_data[user_data["Label"] == 1]
        user_neg_data = user_data[user_data["Label"] == 0]
        logging.debug("start to sample for user %s." % str(u))
        for i in user_pos_data.index:
            logging.debug("yielding a user positive sample for %s." % str(u))
            yield ((user_pos_data.loc[i, "UserID"],
                    user_pos_data.loc[i, "MovieID"],
                    user_pos_data.loc[i, "Gender"],
                    user_pos_data.loc[i, "Age"],
                    user_pos_data.loc[i, "Occupation"],
                    user_pos_data.loc[i, "Zip-code"],
                    user_pos_data.loc[i, "Year"],
                    encoder.transform(
                        [[int(i) for i in
                          user_pos_data.loc[i, "Genres"].split(" ")]]
                    ).tolist()[0]), 1)
            if len(user_neg_data) > 0:
                user_neg_samples = user_neg_data.sample(
                    params["num_neg_samples"], replace=True, random_state=42)
                user_neg_samples.reset_index(inplace=True)
                logging.debug("yielding negative samples for %s." % str(u))
                for j in user_neg_samples.index:
                    yield ((user_neg_samples.loc[j, "UserID"],
                            user_neg_samples.loc[j, "MovieID"],
                            user_neg_samples.loc[j, "Gender"],
                            user_neg_samples.loc[j, "Age"],
                            user_neg_samples.loc[j, "Occupation"],
                            user_neg_samples.loc[j, "Zip-code"],
                            user_neg_samples.loc[j, "Year"],
                            encoder.transform(
                                [[int(i) for i in
                                  user_neg_samples.loc[j, "Genres"].split(" ")]]
                            ).tolist()[0]), 0)

def _eval_generator(params):
    """Yielding samples one by one of MovieLens 1M dataset for evaluating, with
    the following info:

        ((UserID, MovieID, Gender, Age, Occupation,
          Zip-code, Year, Genres(bin)), Label).

    Args:
        params: dict, unsed in this function.

    Returns:
        Samples include the info mentioned above.
    """
    logging.info("reading data from data file.")
    data = pd.read_csv(os.path.join("..", os.path.join("data", eval_file)))
    logging.info("yielding evaluation samples.")
    for i in data.index:
        yield ((data.loc[i, "UserID"],
                data.loc[i, "MovieID"],
                data.loc[i, "Gender"],
                data.loc[i, "Age"],
                data.loc[i, "Occupation"],
                data.loc[i, "Zip-code"],
                data.loc[i, "Year"],
                encoder.transform(
                    [[int(i) for i in data.loc[i, "Genres"].split(" ")]]
                ).tolist()[0]),
                data.loc[i, "Label"])

def _pred_generator(params):
    """Yielding samples one by one of MovieLens 1M dataset for predicting, with
    the following info:

        ((UserID, MovieID, Gender, Age, Occupation,
          Zip-code, Year, Genres(bin)), Label).

    Args:
        params: dict, unsed in this function.

    Returns:
        Samples include the info mentioned above.
    """
    logging.info("reading data from data file.")
    data = pd.read_csv(os.path.join("..", os.path.join("data", pred_file)))
    logging.info("yielding prediction samples.")
    for i in data.index:
        yield ((data.loc[i, "UserID"],
                data.loc[i, "MovieID"],
                data.loc[i, "Gender"],
                data.loc[i, "Age"],
                data.loc[i, "Occupation"],
                data.loc[i, "Zip-code"],
                data.loc[i, "Year"],
                encoder.transform(
                    [[int(i) for i in data.loc[i, "Genres"].split(" ")]]
                ).tolist()[0]),
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
        output_types = ((tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                         tf.int64, tf.int64, tf.int64), tf.int64),
        output_shapes = (([], [], [], [], [], [], [], [None]), []))
    logging.info("batching dataset.")
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(params["batch_size"] * 1000, seed=2019)
    dataset = (
        dataset.batch(
            batch_size = params["batch_size"], drop_remainder = True))
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat().prefetch(params["batch_size"])
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
    # ...
    # shape of genres: [batch_size, num_genres]
    (users,    items, genders, ages,  occupations,
     zipcodes, years, genres) = features

    logging.info("defining user embedding lookup table.")
    # shape: [num_users, embedding_size]
    user_embedding = tf.Variable(
        tf.truncated_normal([params["num_users"] + 1, params["embedding_size"]],
                            dtype=tf.float64),
        dtype = tf.float64,
        name = "user_embedding")
    logging.info("defining item embedding lookup table.")
    # shape: [num_items, embedding_size]
    item_embedding = tf.Variable(
        tf.truncated_normal([params["num_items"] + 1, params["embedding_size"]],
                            dtype=tf.float64),
        dtype = tf.float64,
        name = "item_embedding")
    logging.info("defining gender embedding lookup table.")
    # shape: [num_genders, embedding_size]
    gender_embedding = tf.Variable(
        tf.truncated_normal([params["num_genders"] + 1,
                             params["embedding_size"]],
                            dtype=tf.float64),
        dtype = tf.float64,
        name = "gender_embedding")
    logging.info("defining age embedding lookup table.")
    # shape: [num_ages, embedding_size]
    age_embedding = tf.Variable(
        tf.truncated_normal([params["num_ages"] + 1, params["embedding_size"]],
                            dtype=tf.float64),
        dtype = tf.float64,
        name = "age_embedding")
    logging.info("defining occupation embedding lookup table.")
    # shape: [num_occupations, embedding_size]
    occupation_embedding = tf.Variable(
        tf.truncated_normal([params["num_occupations"] + 1,
                             params["embedding_size"]],
                            dtype=tf.float64),
        dtype = tf.float64,
        name = "occupation_embedding")
    logging.info("defining zipcode embedding lookup table.")
    # shape: [num_zipcodes, embedding_size]
    zipcode_embedding = tf.Variable(
        tf.truncated_normal([params["num_zipcodes"] + 1,
                             params["embedding_size"]],
                            dtype=tf.float64),
        dtype = tf.float64,
        name = "zipcode_embedding")
    logging.info("defining year embedding lookup table.")
    # shape: [num_years, embedding_size]
    year_embedding = tf.Variable(
        tf.truncated_normal([params["num_years"] + 1,
                             params["embedding_size"]],
                            dtype=tf.float64),
        dtype = tf.float64,
        name = "year_embedding")
    logging.info("defining genres embedding lookup table.")
    # shape: [num_genres, embedding_size]
    genres_embedding = tf.Variable(
        tf.truncated_normal([params["num_genres"],
                             params["embedding_size"]],
                            dtype=tf.float64),
        dtype = tf.float64,
        name = "genres_embedding")

    logging.info("looking up user embedding.")
    # shape: [batch_size, embedding_size]
    user_emb_inp = tf.nn.embedding_lookup(user_embedding, users)
    logging.info("looking up item embedding.")
    # shape: [batch_size, embedding_size]
    item_emb_inp = tf.nn.embedding_lookup(item_embedding, items)
    logging.info("looking up gender embedding.")
    # shape: [batch_size, 1, embedding_size]
    gender_emb_inp = tf.reshape(
        tf.nn.embedding_lookup(gender_embedding, genders),
        shape = [params["batch_size"], -1, params["embedding_size"]])
    logging.info("looking up age embedding.")
    # shape: [batch_size, 1, embedding_size]
    age_emb_inp = tf.reshape(
        tf.nn.embedding_lookup(age_embedding, ages),
        shape = [params["batch_size"], -1, params["embedding_size"]])
    logging.info("looking up occupation embedding.")
    # shape: [batch_size, 1, embedding_size]
    occupation_emb_inp = tf.reshape(
        tf.nn.embedding_lookup(occupation_embedding, occupations),
        shape = [params["batch_size"], -1, params["embedding_size"]])
    logging.info("looking up zipcode embedding.")
    # shape: [batch_size, 1, embedding_size]
    zipcode_emb_inp = tf.reshape(
        tf.nn.embedding_lookup(zipcode_embedding, zipcodes),
        shape = [params["batch_size"], -1, params["embedding_size"]])
    logging.info("looking up year embedding.")
    # shape: [batch_size, embedding_size]
    year_emb_inp = tf.reshape(
        tf.nn.embedding_lookup(year_embedding, years),
        shape = [params["batch_size"], -1, params["embedding_size"]])
    logging.info("looking up genres embedding.")
    # shape: [batch_size, num_genres, embedding_size]
    genres_emb_inp = tf.reshape(
        tf.reshape(tf.cast(genres, dtype = tf.float64),
                   shape=[params["batch_size"], -1, 1]) * genres_embedding,
        shape = [params["batch_size"], -1, params["embedding_size"]])

    logging.info("computing truth of user-item record.")
    # shape: [batch_size, 1, embedding_size]
    truth = tf.reshape(
        user_emb_inp * item_emb_inp,
        shape = [params["batch_size"], -1, params["embedding_size"]])

    logging.info("computing consideration factors.")
    # shape: [batch_size, 5 + num_genres, embedding_size]
    considerations = tf.concat([
        gender_emb_inp,
        age_emb_inp,
        occupation_emb_inp,
        zipcode_emb_inp,
        year_emb_inp,
        genres_emb_inp
    ], axis = 1)

    logging.info("pay attention to considerations using truth.")
    # shape: [batch_size, 5 + num_genres, 1]
    scores = tf.nn.softmax(
        # sum(query * key)
        tf.reduce_sum(truth * considerations, axis = -1, keepdims = True),
        axis = 1)
    # shape: [batch_size, (5 + num_genres) * embedding_size]
    attn_output = tf.reshape(
        scores * considerations,
        shape = [params["batch_size"], -1])

    logging.info("passing attention output to MLP.")
    hidden1   = tf.layers.Dense(
        units = 2 * 2 * params["num_factors"], activation = tf.nn.relu)(attn_output)
    hidden2   = tf.layers.Dense(
        units = 2 * params["num_factors"], activation = tf.nn.relu)(hidden1)
    MLP_output = tf.layers.Dense(
        units = params["num_factors"], activation = tf.nn.relu)(hidden2)

    logging.info("defining output layer.")
    # shape: [batch_size, 1]
    logits = tf.layers.Dense(
        units = 1,
        activation = tf.nn.sigmoid,
        use_bias = False)(MLP_output)
    logging.info("reshaping logits.")
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
