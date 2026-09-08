#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Amazon Reviews'23 rating/review preprocessing, 2026-07-13.

Writes suffixed outputs beside historical files:
``user_item_rating_20260713`` and ``user_item_pure_reviews_20260713``.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


CAUSAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_utils import ensure_dir, find_causal_root, setup_logging, write_table


ROOT = find_causal_root(__file__)
RAW_BASE = _dcml_paths.workspace_path(ROOT, 'raw_data')
OUT_BASE = _dcml_paths.workspace_path(ROOT, 'processed_data')
LOG_DIR = ensure_dir(_dcml_paths.workspace_path(ROOT, 'log'))
logger = setup_logging(LOG_DIR, "process_amazon_reviews_20260713")
DATASETS = ("amazon_appliances", "amazon_beauty")


def review_path(dataset: str) -> Path:
    if dataset == "amazon_appliances":
        return RAW_BASE / f"{dataset}_raw" / "reviews_Appliances.jsonl"
    return RAW_BASE / f"{dataset}_raw" / "All_Beauty.jsonl"


def convert_timestamp(ts) -> str:
    if ts is None:
        return "19700101"
    try:
        value = float(ts)
        if value > 1e11:
            value = value / 1000.0
        return datetime.fromtimestamp(value).strftime("%Y%m%d")
    except Exception:
        return "19700101"


def clean_review_text(text) -> tuple[bool, str]:
    if not text or not isinstance(text, str):
        return False, ""
    cleaned = text.strip()
    garbage = {
        "none",
        "na",
        "n/a",
        "good",
        "great",
        "ok",
        "okay",
        "nice",
        "love it",
        "works great",
        "five stars",
        "five star",
        "as expected",
    }
    if len(cleaned) < 10:
        return False, cleaned
    if cleaned.lower() in garbage:
        return False, cleaned
    return True, cleaned


def process_dataset(dataset: str) -> None:
    start = time.time()
    logger.info("=" * 80)
    logger.info("Processing Amazon reviews/ratings: %s", dataset)
    path = review_path(dataset)
    out_rating_dir = ensure_dir(OUT_BASE / dataset / "rating")
    out_reviews_dir = ensure_dir(OUT_BASE / dataset / "pure_reviews")

    ratings: list[dict] = []
    reviews: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_id = obj.get("user_id")
            item_id = obj.get("parent_asin")
            rating = obj.get("rating")
            timestamp = obj.get("timestamp")
            if not user_id or not item_id or rating is None:
                continue
            date = convert_timestamp(timestamp)
            ratings.append(
                {
                    "user_id": str(user_id),
                    "item_id": str(item_id),
                    "rating": float(rating),
                    "timestamp": date,
                }
            )
            valid, text = clean_review_text(obj.get("text", ""))
            if valid:
                reviews.append(
                    {
                        "user_id": str(user_id),
                        "item_id": str(item_id),
                        "review_text": text,
                        "is_image_review": int(len(obj.get("images", [])) > 0),
                        "timestamp": date,
                    }
                )

    df_rating = pd.DataFrame(ratings)
    df_reviews = pd.DataFrame(reviews)
    if not df_rating.empty:
        df_rating = df_rating.drop_duplicates(["user_id", "item_id", "timestamp"])
    if not df_reviews.empty:
        df_reviews = df_reviews.drop_duplicates(["user_id", "item_id", "timestamp"])

    write_table(df_rating, out_rating_dir / "user_item_rating_20260713.parquet", logger)
    write_table(df_reviews, out_reviews_dir / "user_item_pure_reviews_20260713.parquet", logger)
    logger.info("Ratings: %d | pure reviews: %d", len(df_rating), len(df_reviews))
    logger.info("Completed %s in %.2f seconds", dataset, time.time() - start)


def main() -> None:
    for dataset in DATASETS:
        process_dataset(dataset)


if __name__ == "__main__":
    main()
