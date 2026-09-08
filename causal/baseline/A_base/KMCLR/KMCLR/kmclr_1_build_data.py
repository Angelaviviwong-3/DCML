#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build independent stage-specific KMCLR matrices for Suning."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

for parent in Path(__file__).resolve().parents:
    if (parent / "baseline_utils.py").exists():
        sys.path.insert(0, str(parent))
        break

from baseline_utils import (  # noqa: E402
    SCRIPT_VERSION,
    STAGES,
    cache_dir,
    find_causal_root,
    load_id_maps,
    load_train_table,
    map_positive_pairs,
    setup_logger,
    unique_pairs,
)


def main() -> None:
    root = find_causal_root(__file__)
    logger = setup_logger(root, "kmclr_build_data")
    out_dir = cache_dir(root, "kmclr_cache")
    user2id, item2id = load_id_maps(root)
    user_num, item_num = len(user2id), len(item2id)
    train = load_train_table(root)
    for stage in STAGES:
        pairs = unique_pairs(map_positive_pairs(train, user2id, item2id, stage))
        rows = pairs[:, 0] if len(pairs) else np.asarray([], dtype=int)
        cols = pairs[:, 1] if len(pairs) else np.asarray([], dtype=int)
        matrix = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(user_num, item_num))
        with open(out_dir / f"trn_{stage}.pkl", "wb") as handle:
            pickle.dump(matrix, handle)
        logger.info("Prepared KMCLR %s %s matrix | edges=%d", stage, SCRIPT_VERSION, len(pairs))
    (out_dir / "data_size_20260717.txt").write_text(f"{user_num} {item_num}", encoding="utf-8")


if __name__ == "__main__":
    main()
