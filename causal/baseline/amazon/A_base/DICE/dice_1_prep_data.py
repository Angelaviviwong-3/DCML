#!/usr/bin/env python3
"""Prepare independent stage-specific Amazon Purchase-only inputs for DICE."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


import sys
from collections import Counter
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "amazon_baseline_utils.py").exists():
        sys.path.insert(0, str(parent))
        break

from amazon_baseline_utils import (  # noqa: E402
    STAGES,
    cache_dir,
    find_causal_root,
    load_id_maps,
    load_train_table,
    map_positive_pairs,
    setup_logger,
    unique_pairs,
    write_json,
    write_pair_file,
)


def main() -> None:
    root = find_causal_root(__file__)
    logger = setup_logger(root, "dice_prep_data")
    output = cache_dir(root, "dice_cache")
    user2id, item2id = load_id_maps(root)
    train = load_train_table(root)
    (output / "data_size_20260718.txt").write_text(f"{len(user2id)} {len(item2id)}", encoding="utf-8")
    for stage in STAGES:
        pairs = unique_pairs(map_positive_pairs(train, user2id, item2id, stage))
        write_pair_file(output / f"train_{stage}_20260718.txt", pairs)
        popularity = Counter(int(item) for _, item in pairs)
        write_json({str(item): int(count) for item, count in popularity.items()}, output / f"item_pop_{stage}_20260718.json")
        logger.info("DICE %s: %d training edges", stage, len(pairs))


if __name__ == "__main__":
    main()
