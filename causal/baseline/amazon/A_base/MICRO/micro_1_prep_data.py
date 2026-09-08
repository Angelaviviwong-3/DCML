#!/usr/bin/env python3
"""Prepare independent stage-specific Amazon Purchase-only inputs for MICRO."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


import sys
from pathlib import Path

import numpy as np

BASELINE_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = BASELINE_ROOT / "amazon_baseline_utils.py"
if not UTILS_PATH.is_file():
    raise FileNotFoundError(f"Missing 20260718 baseline utilities: {UTILS_PATH}")
sys.path.insert(0, str(BASELINE_ROOT))

from amazon_baseline_utils import (  # noqa: E402
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
    write_json,
)


def main() -> None:
    root = find_causal_root(__file__)
    logger = setup_logger(root, "micro_prep_data")
    output = cache_dir(root, "micro_cache")
    user2id, item2id = load_id_maps(root)
    train = load_train_table(root)
    columns = amazon_feature_columns(train)
    np.save(output / "image_feat_20260718.npy", feature_matrix(train, item2id, columns["image"]))
    np.save(output / "text_feat_20260718.npy", feature_matrix(train, item2id, columns["text"]))
    (output / "data_size_20260718.txt").write_text(f"{len(user2id)} {len(item2id)}", encoding="utf-8")
    for stage in STAGES:
        pairs = unique_pairs(map_positive_pairs(train, user2id, item2id, stage))
        interactions: dict[str, list[int]] = {}
        for user, item in pairs:
            interactions.setdefault(str(int(user)), []).append(int(item))
        write_json(interactions, output / f"train_data_{stage}_20260718.json")
        logger.info("MICRO %s: %d training edges, %d training users", stage, len(pairs), len(interactions))


if __name__ == "__main__":
    main()
