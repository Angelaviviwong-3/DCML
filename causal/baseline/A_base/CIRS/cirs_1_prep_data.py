#!/usr/bin/env python3
"""Prepare independent stage-specific Suning inputs for CIRS."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths


import sys
from pathlib import Path

import numpy as np

for parent in Path(__file__).resolve().parents:
    if (parent / "baseline_utils.py").exists():
        sys.path.insert(0, str(parent))
        break

from baseline_utils import (  # noqa: E402
    STAGES,
    cache_dir,
    find_causal_root,
    load_id_maps,
    load_train_table,
    setup_logger,
)

N_SLOTS = 5


def main() -> None:
    root = find_causal_root(__file__)
    logger = setup_logger(root, "cirs_prep_data")
    output = cache_dir(root, "cirs_cache")
    user2id, item2id = load_id_maps(root)
    train = load_train_table(root)
    (output / "data_size_20260717.txt").write_text(f"{len(user2id)} {len(item2id)}", encoding="utf-8")
    for stage in STAGES:
        frame = train[train[f"y_{stage}"].fillna(0).astype(int) == 1].copy()
        if frame.empty:
            raise ValueError(f"CIRS requires non-empty y_{stage} interactions")
        if "timestamp" in frame.columns:
            frame = frame.sort_values("timestamp", kind="mergesort")
        frame = frame.reset_index(drop=True)
        frame["slot"] = np.minimum(np.arange(len(frame)) * N_SLOTS // len(frame), N_SLOTS - 1)
        slot_items: dict[int, list[int]] = {slot: [] for slot in range(N_SLOTS)}
        edge_count = 0
        with open(output / f"train_with_time_{stage}_20260717.txt", "w", encoding="utf-8") as handle:
            for row in frame[["user_id", "item_id", "slot"]].itertuples(index=False):
                user_raw, item_raw = str(row.user_id), str(row.item_id)
                if user_raw not in user2id or item_raw not in item2id:
                    continue
                user, item, slot = user2id[user_raw], item2id[item_raw], int(row.slot)
                handle.write(f"{user} {item} {slot}\n")
                slot_items[slot].append(item)
                edge_count += 1
        popularity = np.zeros((N_SLOTS, len(item2id)), dtype=np.float32)
        for slot, values in slot_items.items():
            counts = np.bincount(values, minlength=len(item2id)).astype(np.float32)
            smoothed = (counts + 1.0) / (len(values) + len(item2id))
            width = float(smoothed.max() - smoothed.min())
            popularity[slot] = (smoothed - smoothed.min()) / (width + 1e-10)
        np.save(output / f"item_popularity_{stage}_20260717.npy", popularity)
        logger.info("CIRS %s: %d training events", stage, edge_count)


if __name__ == "__main__":
    main()
