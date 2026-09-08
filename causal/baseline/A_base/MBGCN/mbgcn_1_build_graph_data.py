#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build independent stage-specific MBGCN caches for Suning."""

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
    find_causal_root,
    load_id_maps,
    load_train_table,
    map_positive_pairs,
    setup_logger,
    stage_cache_dir,
    stage_seed,
    unique_pairs,
    write_pair_file,
)


def main() -> None:
    root = find_causal_root(__file__)
    logger = setup_logger(root, "mbgcn_build_graph_data")
    user2id, item2id = load_id_maps(root)
    user_num, item_num = len(user2id), len(item2id)
    train = load_train_table(root)

    for stage in STAGES:
        out_dir = stage_cache_dir(root, "mbgcn_cache", stage)
        sample_dir = out_dir / "sample_file"
        sample_dir.mkdir(parents=True, exist_ok=True)
        pairs = unique_pairs(map_positive_pairs(train, user2id, item2id, stage))
        if not len(pairs):
            raise ValueError(f"MBGCN {stage} has no training interactions")
        (out_dir / "data_size_20260717.txt").write_text(f"{user_num}\t{item_num}\n", encoding="utf-8")
        write_pair_file(out_dir / f"{stage}_20260717.txt", pairs, sep="\t")
        interacted: dict[int, set[int]] = {}
        for user, item in pairs:
            interacted.setdefault(int(user), set()).add(int(item))
        rng = np.random.default_rng(stage_seed(stage))
        all_items = np.arange(item_num)
        for sample_index in tqdm(range(5), desc=f"MBGCN {stage} sampling", leave=False):
            with open(sample_dir / f"sample_{sample_index}_20260717.txt", "w", encoding="utf-8") as handle:
                for user, positives in interacted.items():
                    for positive in sorted(positives):
                        negative = int(rng.choice(all_items))
                        while negative in positives:
                            negative = int(rng.choice(all_items))
                        handle.write(f"{user}\t{positive}\t{negative}\n")
        logger.info("Prepared MBGCN %s cache %s | edges=%d", stage, SCRIPT_VERSION, len(pairs))


if __name__ == "__main__":
    main()
