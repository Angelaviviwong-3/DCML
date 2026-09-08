#!/usr/bin/env python3
"""Train one MGCE model per Amazon Purchase outcome and export audited scores."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

for parent in Path(__file__).resolve().parents:
    if (parent / "amazon_baseline_utils.py").exists():
        sys.path.insert(0, str(parent))
        break

from amazon_baseline_utils import (  # noqa: E402
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
from mgce_2_model_api import MGCEEngine  # noqa: E402


def main() -> None:
    tf.disable_v2_behavior()
    root = find_causal_root(__file__)
    logger = setup_logger(root, "mgce_stage_specific_train_export")
    cache = cache_dir(root, "mgce_cache")
    users, items = map(int, (cache / "data_size_20260718.txt").read_text(encoding="utf-8").split())
    image = np.load(cache / "image_feat_20260718.npy")
    diagnostics = {"script_version": "20260718", "stage_specific_training": True, "stages": {}}
    for stage in STAGES:
        seed = stage_seed(stage, 71)
        rng = np.random.default_rng(seed)
        data = np.loadtxt(cache / f"train_{stage}_20260718.txt", dtype=np.int64).reshape(-1, 2)
        popularity = np.load(cache / f"pop_feat_{stage}_20260718.npy")
        user_positive: dict[int, set[int]] = {}
        for user, item in data:
            user_positive.setdefault(int(user), set()).add(int(item))
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        model = MGCEEngine(users, items, image, popularity, 64, 1e-3)
        session = tf.Session(config=tf.ConfigProto(device_count={"GPU": 0}, allow_soft_placement=True))
        session.run(tf.global_variables_initializer())
        losses = []
        for epoch in range(25):
            order = rng.permutation(len(data))
            total, batches = 0.0, 0
            for offset in range(0, len(order), 2048):
                batch = data[order[offset : offset + 2048]]
                user, positive = batch[:, 0], batch[:, 1]
                negative = rng.integers(0, items, size=len(batch), dtype=np.int64)
                for index, uid in enumerate(user):
                    while int(negative[index]) in user_positive[int(uid)]:
                        negative[index] = rng.integers(items)
                _, loss = session.run(
                    (model.optimizer, model.loss),
                    feed_dict={model.users: user, model.positive_items: positive, model.negative_items: negative},
                )
                total += float(loss)
                batches += 1
            epoch_loss = total / max(batches, 1)
            losses.append(epoch_loss)
            logger.info("MGCE/%s epoch=%02d loss=%.6f", stage, epoch, epoch_loss)

        predictions: dict[str, dict[str, float]] = {}
        for user, candidate in load_candidates(root, stage).items():
            ids = candidate_items(candidate)
            scores = session.run(model.scores, feed_dict={model.users: [int(user)]})[0]
            predictions[str(user)] = {str(item): float(scores[item]) for item in ids}
        metadata = export_prediction_bundle(
            root,
            "MGCE",
            stage,
            predictions,
            training_edges=len(data),
            training_protocol="Independent stage-specific MGCE multimodal conformity objective on Set B",
            extra_metadata={
                "epochs": 25,
                "seed": seed,
                "training_inference_embedding_match": True,
                "first_epoch_loss": losses[0],
                "last_epoch_loss": losses[-1],
            },
        )
        diagnostics["stages"][stage] = metadata
        session.close()
    write_json(diagnostics, baseline_data_dir(root) / "MGCE_training_diagnostics_20260718.json", indent=2)


if __name__ == "__main__":
    main()
