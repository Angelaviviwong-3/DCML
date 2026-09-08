#!/usr/bin/env python3
"""Multimodal group-conformity encoder for the 20260718 MGCE rerun."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class MGCEEngine:
    def __init__(self, users: int, items: int, image_features, popularity_features, dim: int, learning_rate: float):
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.positive_items = tf.placeholder(tf.int32, shape=(None,))
        self.negative_items = tf.placeholder(tf.int32, shape=(None,))
        initializer = tf.glorot_uniform_initializer()
        self.user_embedding = tf.get_variable("user_embedding", (users, dim), initializer=initializer)
        self.item_embedding = tf.get_variable("item_embedding", (items, dim), initializer=initializer)
        image_projection = tf.get_variable("image_projection", (image_features.shape[1], dim), initializer=initializer)
        popularity_projection = tf.get_variable("popularity_projection", (popularity_features.shape[1], dim), initializer=initializer)
        image_embedding = tf.matmul(tf.cast(image_features, tf.float32), image_projection)
        popularity_embedding = tf.matmul(tf.cast(popularity_features, tf.float32), popularity_projection)
        self.fused_item_embedding = self.item_embedding + image_embedding + popularity_embedding
        user = tf.nn.embedding_lookup(self.user_embedding, self.users)
        positive = tf.nn.embedding_lookup(self.fused_item_embedding, self.positive_items)
        negative = tf.nn.embedding_lookup(self.fused_item_embedding, self.negative_items)
        positive_score = tf.reduce_sum(user * positive, axis=1)
        negative_score = tf.reduce_sum(user * negative, axis=1)
        regularization = 1e-5 * (tf.nn.l2_loss(user) + tf.nn.l2_loss(positive) + tf.nn.l2_loss(negative))
        self.loss = -tf.reduce_mean(tf.log(tf.sigmoid(positive_score - negative_score) + 1e-10)) + regularization
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.scores = tf.matmul(user, self.fused_item_embedding, transpose_b=True)
