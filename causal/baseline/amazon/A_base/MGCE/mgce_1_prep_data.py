#!/usr/bin/env python3
"""Prepare independent stage-specific Amazon Purchase-only inputs for MGCE."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


import sys
from collections import Counter
from pathlib import Path

import numpy as np

for parent in Path(__file__).resolve().parents:
    if (parent / "amazon_baseline_utils.py").exists():
        sys.path.insert(0, str(parent))
        break

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
    write_pair_file,
)


def main() -> None:
    root = find_causal_root(__file__)
    logger = setup_logger(root, "mgce_prep_data")
    output = cache_dir(root, "mgce_cache")
    user2id, item2id = load_id_maps(root)
    train = load_train_table(root)
    columns = amazon_feature_columns(train)
    np.save(output / "image_feat_20260718.npy", feature_matrix(train, item2id, columns["image"]))
    (output / "data_size_20260718.txt").write_text(f"{len(user2id)} {len(item2id)}", encoding="utf-8")
    for stage in STAGES:
        pairs = unique_pairs(map_positive_pairs(train, user2id, item2id, stage))
        write_pair_file(output / f"train_{stage}_20260718.txt", pairs)
        counts = Counter(int(item) for _, item in pairs)
        popularity = np.zeros((len(item2id), 1), dtype=np.float32)
        for item, count in counts.items():
            popularity[item, 0] = np.log1p(count)
        scale = float(popularity.std())
        popularity = ((popularity - popularity.mean()) / (scale if scale > 1e-12 else 1.0)).astype(np.float32)
        np.save(output / f"pop_feat_{stage}_20260718.npy", np.nan_to_num(popularity))
        logger.info("MGCE %s: %d training edges", stage, len(pairs))


if __name__ == "__main__":
    main()
