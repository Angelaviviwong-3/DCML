#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Amazon multimodal scalar merge without plotting, 2026-07-13.

This script exists only to make the 20260713 Amazon pipeline non-overwriting.
It writes ``*_item_multimodal_scalars_merged_full_20260713`` beside historical
MCRE outputs and does not generate visualization files.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths


import argparse
import glob
import sys
import time
from pathlib import Path

import pandas as pd


CAUSAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_utils import ensure_dir, find_causal_root, read_table, setup_logging, write_table, zero_preserved_rank_norm


ROOT = find_causal_root(__file__)
LOG_DIR = ensure_dir(_dcml_paths.workspace_path(ROOT, 'log'))
CALIBRATE_COLS = [
    "T_con_mkt",
    "T_int_vis",
    "T_int_fac",
    "atomic_a_pri",
    "atomic_a_gft",
    "atomic_a_sub",
    "atomic_a_urg",
    "atomic_a_spec",
    "atomic_a_str",
]


def existing_merged_path(save_dir: Path, dataset: str) -> Path:
    if dataset == "amazon_appliances":
        return save_dir / f"{dataset}_item_multimodal_scalars_merged_full_2.parquet"
    return save_dir / f"{dataset}_item_multimodal_scalars_merged_full.parquet"


def load_source(save_dir: Path, dataset: str, logger) -> pd.DataFrame:
    patterns = [
        save_dir / f"{dataset}_item_multimodal_scalars_full_20260713_*.csv",
        save_dir / f"{dataset}_item_multimodal_scalars_full_20260713_*.parquet",
        save_dir / f"{dataset}_item_multimodal_scalars_full_2_*.csv",
        save_dir / f"{dataset}_item_multimodal_scalars_full_2_*.parquet",
    ]
    files: list[str] = []
    for pattern in patterns:
        files = [f for f in glob.glob(str(pattern)) if "checkpoint" not in Path(f).name]
        if files:
            break
    if files:
        logger.info("Merging %d split MCRE files.", len(files))
        parts = [read_table(f) for f in sorted(files)]
        return pd.concat(parts, ignore_index=True)
    fallback = existing_merged_path(save_dir, dataset)
    logger.info("No split files found; cloning historical merged MCRE table: %s", fallback)
    return read_table(fallback)


def run(dataset: str) -> None:
    logger = setup_logging(LOG_DIR, f"merge_mcre_{dataset}_20260713")
    start = time.time()
    save_dir = ensure_dir(_dcml_paths.workspace_path(ROOT, 'processed_data') / dataset / "item_multimodal_scalars")
    logger.info("=" * 80)
    logger.info("Amazon MCRE merge without plotting, dataset=%s", dataset)
    df = load_source(save_dir, dataset, logger)
    df["item_id"] = df["item_id"].astype(str)
    before = len(df)
    df = df.drop_duplicates("item_id").reset_index(drop=True)
    logger.info("Rows after item de-duplication: %d (removed %d)", len(df), before - len(df))

    for col in CALIBRATE_COLS:
        if col in df.columns:
            df[f"{col}_calibrated"] = zero_preserved_rank_norm(pd.to_numeric(df[col], errors="coerce").fillna(0.0), seed=20260713)

    out_path = save_dir / f"{dataset}_item_multimodal_scalars_merged_full_20260713.parquet"
    write_table(df, out_path, logger)
    logger.info("Completed in %.2f seconds", time.time() - start)
    logger.info("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="amazon_appliances", choices=["amazon_appliances", "amazon_beauty"])
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
