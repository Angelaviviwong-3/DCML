#!/usr/bin/env python3
"""Conditional popularity-aware BPR model for the 20260717 CIRS rerun."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths


import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class ConditionalBPRMF:
    def __init__(self, users: int, items: int, dim: int, learning_rate: float):
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.positive_items = tf.placeholder(tf.int32, shape=(None,))
        self.negative_items = tf.placeholder(tf.int32, shape=(None,))
        self.positive_popularity = tf.placeholder(tf.float32, shape=(None,))
        self.negative_popularity = tf.placeholder(tf.float32, shape=(None,))
        initializer = tf.glorot_uniform_initializer()
        self.user_embedding = tf.get_variable("user_embedding", (users, dim), initializer=initializer)
        self.item_embedding = tf.get_variable("item_embedding", (items, dim), initializer=initializer)
        self.popularity_weight = tf.get_variable("popularity_weight", (1,), initializer=tf.zeros_initializer())
        self.preference_scale = tf.get_variable("preference_scale", (1,), initializer=tf.zeros_initializer())
        user = tf.nn.embedding_lookup(self.user_embedding, self.users)
        positive = tf.nn.embedding_lookup(self.item_embedding, self.positive_items)
        negative = tf.nn.embedding_lookup(self.item_embedding, self.negative_items)
        positive_score = tf.reduce_sum(user * positive, axis=1) * tf.exp(self.preference_scale)
        negative_score = tf.reduce_sum(user * negative, axis=1) * tf.exp(self.preference_scale)
        positive_score += self.popularity_weight * tf.log(self.positive_popularity + 1e-10) * 50.0
        negative_score += self.popularity_weight * tf.log(self.negative_popularity + 1e-10) * 50.0
        regularization = 1e-4 * (tf.nn.l2_loss(user) + tf.nn.l2_loss(positive) + tf.nn.l2_loss(negative))
        self.loss = -tf.reduce_mean(tf.log(tf.sigmoid(positive_score - negative_score) + 1e-10)) + regularization
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.debiased_scores = tf.matmul(user, self.item_embedding, transpose_b=True) * tf.exp(self.preference_scale)
