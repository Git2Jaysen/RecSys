# coding: utf-8

import os
import sys
import json
import math
import heapq
import numpy as np

__all__ = ["evaluate_at_K"]

def _compute_Hit_Ratio(ranklist, real_item):
    for item in ranklist:
        if item == real_item:
            return 1
    return 0

def _compute_NDCG(ranklist, real_item):
    for i, item in enumerate(ranklist):
        if item == real_item:
            return math.log(2) / math.log(i + 2)
    return 0

def evaluate_at_K(preds, K):
    """Evaluating predictions with HR@K and NDCG@K. This work is copying from
    https://github.com/hexiangnan/neural_collaborative_filtering/blob/master/
    evaluate.py.

    Args:
        preds: generator, predictions.
        K    : the position need to be evaluated.

    Returns:
        HR@K and NDCG@K.
    """
    Hit_Ratio, NDCG, num_samples, scores, count = 0.0, 0.0, 0, {}, 0
    for i, pred in enumerate(preds):
        score[i], count = pred, count + 1
        if count % 100 == 0:
            num_samples, real_item, count = num_samples + 1, i - 99, 0
            ranklist = heapq.nlargest(K, scores, key = scores.get)
            Hit_Ratio += _compute_Hit_Ratio(ranklist, real_item)
            NDCG += _compute_NDCG(ranklist, real_item)
            scores = {}
    return Hit_Ratio / num_samples, NDCG / num_samples
