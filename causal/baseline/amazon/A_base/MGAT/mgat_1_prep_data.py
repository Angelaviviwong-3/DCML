#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare independent stage-specific MGAT caches for Amazon Purchase-only."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


import sys
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

for parent in Path(__file__).resolve().parents:
    if (parent / "amazon_baseline_utils.py").exists():
        sys.path.insert(0, str(parent))
        break

from amazon_baseline_utils import (  # noqa: E402
    SCRIPT_VERSION,
    STAGES,
    cache_dir,
    feature_matrix,
    find_causal_root,
    load_id_maps,
    load_train_table,
    map_positive_pairs,
    setup_logger,
    amazon_feature_columns,
    unique_pairs,
)


def main() -> None:
    root = find_causal_root(__file__)
    logger = setup_logger(root, "mgat_prep_data")
    out_dir = cache_dir(root, "mgat_cache")
    user2id, item2id = load_id_maps(root)
    user_num, item_num = len(user2id), len(item2id)
    train = load_train_table(root)
    columns = amazon_feature_columns(train)
    np.save(out_dir / "FeatureVideo_normal_20260718.npy", feature_matrix(train, item2id, columns["visual"]))
    np.save(out_dir / "FeatureText_normal_20260718.npy", feature_matrix(train, item2id, columns["text"]))
    np.save(out_dir / "FeatureAudio_normal_20260718.npy", feature_matrix(train, item2id, columns["audio"]))
    (out_dir / "data_size_20260718.txt").write_text(f"{user_num} {item_num}", encoding="utf-8")

    for stage in STAGES:
        pairs = unique_pairs(map_positive_pairs(train, user2id, item2id, stage))
        edges = []
        adjacency: dict[int, list[int]] = {}
        for user, item in tqdm(pairs, desc=f"MGAT {stage} edges", leave=False):
            offset_item = int(item) + user_num
            edges.append([int(user), offset_item])
            adjacency.setdefault(int(user), []).append(offset_item)
        np.save(out_dir / f"train_{stage}_20260718.npy", np.asarray(edges, dtype=np.int64))
        np.save(out_dir / f"adjacency_{stage}_20260718.npy", adjacency)
        logger.info("Prepared MGAT %s %s cache | edges=%d", stage, SCRIPT_VERSION, len(edges))


if __name__ == "__main__":
    main()
