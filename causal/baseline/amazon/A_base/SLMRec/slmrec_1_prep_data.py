#!/usr/bin/env python3
"""Prepare independent stage-specific Amazon Purchase-only inputs for SLMRec."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


import sys
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
    logger = setup_logger(root, "slmrec_prep_data")
    output = cache_dir(root, "slmrec_cache")
    user2id, item2id = load_id_maps(root)
    train = load_train_table(root)
    columns = amazon_feature_columns(train)
    np.save(output / "v_feat_20260718.npy", feature_matrix(train, item2id, columns["visual"]))
    np.save(output / "t_feat_20260718.npy", feature_matrix(train, item2id, columns["text"]))
    np.save(output / "a_feat_20260718.npy", feature_matrix(train, item2id, columns["audio"]))
    (output / "data_size_20260718.txt").write_text(f"{len(user2id)} {len(item2id)}", encoding="utf-8")
    for stage in STAGES:
        pairs = unique_pairs(map_positive_pairs(train, user2id, item2id, stage))
        write_pair_file(output / f"train_{stage}_20260718.txt", pairs)
        logger.info("SLMRec %s: %d training edges", stage, len(pairs))


if __name__ == "__main__":
    main()
