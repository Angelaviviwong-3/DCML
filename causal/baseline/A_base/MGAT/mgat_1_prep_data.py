#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare independent stage-specific MGAT caches for Suning."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
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
    if (parent / "baseline_utils.py").exists():
        sys.path.insert(0, str(parent))
        break

from baseline_utils import (  # noqa: E402
    SCRIPT_VERSION,
    STAGES,
    cache_dir,
    feature_matrix,
    find_causal_root,
    load_id_maps,
    load_master_table,
    load_train_table,
    map_positive_pairs,
    setup_logger,
    suning_feature_columns,
    unique_pairs,
)


def main() -> None:
    root = find_causal_root(__file__)
    logger = setup_logger(root, "mgat_prep_data")
    out_dir = cache_dir(root, "mgat_cache")
    user2id, item2id = load_id_maps(root)
    user_num, item_num = len(user2id), len(item2id)
    train = load_train_table(root)
    master = load_master_table(root)
    columns = suning_feature_columns(master)
    np.save(out_dir / "FeatureVideo_normal_20260717.npy", feature_matrix(master, item2id, columns["visual"]))
    np.save(out_dir / "FeatureText_normal_20260717.npy", feature_matrix(master, item2id, columns["text"]))
    np.save(out_dir / "FeatureAudio_normal_20260717.npy", feature_matrix(master, item2id, columns["audio"]))
    (out_dir / "data_size_20260717.txt").write_text(f"{user_num} {item_num}", encoding="utf-8")

    for stage in STAGES:
        pairs = unique_pairs(map_positive_pairs(train, user2id, item2id, stage))
        edges = []
        adjacency: dict[int, list[int]] = {}
        for user, item in tqdm(pairs, desc=f"MGAT {stage} edges", leave=False):
            offset_item = int(item) + user_num
            edges.append([int(user), offset_item])
            adjacency.setdefault(int(user), []).append(offset_item)
        np.save(out_dir / f"train_{stage}_20260717.npy", np.asarray(edges, dtype=np.int64))
        np.save(out_dir / f"adjacency_{stage}_20260717.npy", adjacency)
        logger.info("Prepared MGAT %s %s cache | edges=%d", stage, SCRIPT_VERSION, len(edges))


if __name__ == "__main__":
    main()
