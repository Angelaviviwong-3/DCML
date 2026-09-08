#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Amazon Reviews'23 user/item confounder preprocessing, 2026-07-13.

This safe rerun writes suffixed outputs beside the historical files:
``confounder_user_matrix_20260713`` and ``item_text_price_matrix_20260713``.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CAUSAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_utils import ensure_dir, find_causal_root, setup_logging, write_table


ROOT = find_causal_root(__file__)
RAW_BASE = _dcml_paths.workspace_path(ROOT, 'raw_data')
OUT_BASE = _dcml_paths.workspace_path(ROOT, 'processed_data')
LOG_DIR = ensure_dir(_dcml_paths.workspace_path(ROOT, 'log'))
logger = setup_logging(LOG_DIR, "process_amazon_confounders_20260713")
DATASETS = ("amazon_appliances", "amazon_beauty")


def raw_paths(dataset: str) -> tuple[Path, Path]:
    if dataset == "amazon_appliances":
        return RAW_BASE / f"{dataset}_raw" / "meta_Appliances.jsonl", RAW_BASE / f"{dataset}_raw" / "reviews_Appliances.jsonl"
    return RAW_BASE / f"{dataset}_raw" / "meta_All_Beauty.jsonl", RAW_BASE / f"{dataset}_raw" / "All_Beauty.jsonl"


def parse_price(price_val) -> float:
    if price_val is None:
        return np.nan
    if isinstance(price_val, (int, float)):
        return float(price_val)
    match = re.search(r"[\d.]+", str(price_val))
    if match:
        try:
            return float(match.group())
        except ValueError:
            return np.nan
    return np.nan


def process_dataset(dataset: str) -> None:
    start = time.time()
    logger.info("=" * 80)
    logger.info("Processing Amazon confounders: %s", dataset)
    meta_path, review_path = raw_paths(dataset)
    out_user_dir = ensure_dir(OUT_BASE / dataset / "Confounder_user")
    out_item_dir = ensure_dir(OUT_BASE / dataset / "item_feature&Confounder_price")

    user_records: list[dict] = []
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_id = obj.get("user_id")
            rating = obj.get("rating")
            if user_id and rating is not None:
                user_records.append({"user_id": str(user_id), "rating": float(rating)})
    df_users = pd.DataFrame(user_records)
    df_user_agg = (
        df_users.groupby("user_id")
        .agg(user_review_count=("rating", "count"), user_avg_rating=("rating", "mean"))
        .reset_index()
    )
    scaler = StandardScaler()
    df_user_agg[["user_review_count_scaled", "user_avg_rating_scaled"]] = scaler.fit_transform(
        df_user_agg[["user_review_count", "user_avg_rating"]]
    )
    x_user = df_user_agg[["user_id", "user_review_count_scaled", "user_avg_rating_scaled"]]
    write_table(x_user, out_user_dir / "confounder_user_matrix_20260713.parquet", logger)

    item_records: list[dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = obj.get("parent_asin")
            if not asin:
                continue
            brand = str(obj.get("store") or "Unknown brand").strip()
            category = str(obj.get("main_category") or "Unknown category").strip()
            title = str(obj.get("title") or "").strip()
            features = obj.get("features", [])
            features_str = " ".join(map(str, features))[:300] if features else "None"
            text_w = f"Brand: {brand} | Category: {category} | Title: {title} | Product features: {features_str}"
            item_records.append(
                {
                    "item_id": str(asin),
                    "cat_name": category,
                    "text_W": text_w,
                    "raw_price": obj.get("price"),
                    "rating_number": obj.get("rating_number") or 0,
                    "average_rating": obj.get("average_rating") or 0.0,
                }
            )
    df_items = pd.DataFrame(item_records).drop_duplicates("item_id")
    df_items["price"] = df_items["raw_price"].apply(parse_price)
    cat_median = df_items.groupby("cat_name")["price"].transform("median")
    df_items["price"] = df_items["price"].fillna(cat_median).fillna(df_items["price"].median())
    df_items["x_pri_log"] = np.log1p(df_items["price"])
    df_items["social_signal_log"] = np.log1p(pd.to_numeric(df_items["rating_number"], errors="coerce").fillna(0))
    out_item = df_items[["item_id", "cat_name", "text_W", "x_pri_log", "social_signal_log", "average_rating"]]
    write_table(out_item, out_item_dir / "item_text_price_matrix_20260713.parquet", logger)
    logger.info("Completed %s in %.2f seconds", dataset, time.time() - start)


def main() -> None:
    for dataset in DATASETS:
        process_dataset(dataset)


if __name__ == "__main__":
    main()
