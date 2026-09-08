#!/usr/bin/env python3
"""Train one CIRS model per Suning funnel outcome and export audited scores."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths


import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

for parent in Path(__file__).resolve().parents:
    if (parent / "baseline_utils.py").exists():
        sys.path.insert(0, str(parent))
        break

from baseline_utils import (  # noqa: E402
    STAGES,
    baseline_data_dir,
    cache_dir,
    candidate_items,
    export_prediction_bundle,
    find_causal_root,
    load_candidates,
    setup_logger,
    stage_seed,
    write_json,
)
from cirs_2_model_api import ConditionalBPRMF  # noqa: E402


def main() -> None:
    tf.disable_v2_behavior()
    root = find_causal_root(__file__)
    logger = setup_logger(root, "cirs_stage_specific_train_export")
    cache = cache_dir(root, "cirs_cache")
    users, items = map(int, (cache / "data_size_20260717.txt").read_text(encoding="utf-8").split())
    diagnostics = {"script_version": "20260717", "stage_specific_training": True, "stages": {}}
    for stage in STAGES:
        seed = stage_seed(stage, 61)
        rng = np.random.default_rng(seed)
        data = pd.read_csv(cache / f"train_with_time_{stage}_20260717.txt", sep=r"\s+", header=None, names=["user", "item", "slot"])
        user_items = data.groupby("user")["item"].apply(list).to_dict()
        user_slots = data.groupby("user")["slot"].apply(list).to_dict()
        user_positive = {int(user): set(map(int, values)) for user, values in user_items.items()}
        popularity = np.load(cache / f"item_popularity_{stage}_20260717.npy")
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        model = ConditionalBPRMF(users, items, 64, 1e-3)
        session = tf.Session(config=tf.ConfigProto(device_count={"GPU": 0}, allow_soft_placement=True))
        session.run(tf.global_variables_initializer())
        training_users = np.asarray(sorted(user_items), dtype=np.int64)
        losses = []
        for epoch in range(20):
            rng.shuffle(training_users)
            total, batches = 0.0, 0
            for offset in range(0, len(training_users), 2048):
                batch_users = training_users[offset : offset + 2048]
                positive, negative, positive_pop, negative_pop = [], [], [], []
                for user in batch_users:
                    index = int(rng.integers(len(user_items[int(user)])))
                    pos = int(user_items[int(user)][index])
                    slot = int(user_slots[int(user)][index])
                    neg = int(rng.integers(items))
                    while neg in user_positive[int(user)]:
                        neg = int(rng.integers(items))
                    positive.append(pos)
                    negative.append(neg)
                    positive_pop.append(float(popularity[slot, pos]))
                    negative_pop.append(float(popularity[slot, neg]))
                _, loss = session.run(
                    (model.optimizer, model.loss),
                    feed_dict={
                        model.users: batch_users,
                        model.positive_items: positive,
                        model.negative_items: negative,
                        model.positive_popularity: positive_pop,
                        model.negative_popularity: negative_pop,
                    },
                )
                total += float(loss)
                batches += 1
            epoch_loss = total / max(batches, 1)
            losses.append(epoch_loss)
            logger.info("CIRS/%s epoch=%02d loss=%.6f", stage, epoch, epoch_loss)

        predictions: dict[str, dict[str, float]] = {}
        for user, candidate in load_candidates(root, stage).items():
            ids = candidate_items(candidate)
            scores = session.run(model.debiased_scores, feed_dict={model.users: [int(user)]})[0]
            predictions[str(user)] = {str(item): float(scores[item]) for item in ids}
        metadata = export_prediction_bundle(
            root,
            "CIRS",
            stage,
            predictions,
            training_edges=len(data),
            training_protocol="Independent stage-specific conditional popularity-aware BPR objective on Set B",
            extra_metadata={
                "epochs": 20,
                "seed": seed,
                "time_slots": int(popularity.shape[0]),
                "first_epoch_loss": losses[0],
                "last_epoch_loss": losses[-1],
            },
        )
        diagnostics["stages"][stage] = metadata
        session.close()
    write_json(diagnostics, baseline_data_dir(root) / "CIRS_training_diagnostics_20260717.json", indent=2)


if __name__ == "__main__":
    main()
