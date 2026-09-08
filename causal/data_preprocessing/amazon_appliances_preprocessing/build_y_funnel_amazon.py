#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Amazon Reviews'23 purchase-only outcome builder, 2026-07-13.

Amazon Reviews'23 does not contain native click or add-to-cart logs. This
script therefore builds only the bottom-funnel Purchase outcome used for
cross-category Amazon validation. For naming consistency with Suning, the
outcome column remains ``y_purchase``; its source is an observed review/rating
record from Amazon Reviews'23, not a logged checkout event.

Outputs are written beside the historical files and never overwrite them:
``processed_data/{dataset}/Y/y_behavior_20260713.{parquet,csv}``.
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
from pathlib import Path

import numpy as np
import pandas as pd


CAUSAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_utils import ensure_dir, find_causal_root, setup_logging, write_table


ROOT = find_causal_root(__file__)
RAW_BASE = _dcml_paths.workspace_path(ROOT, 'raw_data')
OUT_BASE = _dcml_paths.workspace_path(ROOT, 'processed_data')
LOG_DIR = ensure_dir(_dcml_paths.workspace_path(ROOT, 'log'))
logger = setup_logging(LOG_DIR, "process_amazon_purchase_outcome_20260713")

DATASETS = ("amazon_appliances", "amazon_beauty")
NEG_PER_PURCHASE = 5
RANDOM_SEED = 20260713


def raw_paths(dataset: str) -> tuple[Path, Path]:
    if dataset == "amazon_appliances":
        candidates = [
            (RAW_BASE / dataset / "meta_Appliances.jsonl", RAW_BASE / dataset / "reviews_Appliances.jsonl"),
            (RAW_BASE / f"{dataset}_raw" / "meta_Appliances.jsonl", RAW_BASE / f"{dataset}_raw" / "reviews_Appliances.jsonl"),
        ]
    else:
        candidates = [
            (RAW_BASE / dataset / "meta_All_Beauty.jsonl", RAW_BASE / dataset / "All_Beauty.jsonl"),
            (RAW_BASE / f"{dataset}_raw" / "meta_All_Beauty.jsonl", RAW_BASE / f"{dataset}_raw" / "All_Beauty.jsonl"),
        ]
    for meta_path, review_path in candidates:
        if meta_path.exists() and review_path.exists():
            return meta_path, review_path
    return candidates[-1]


def read_meta(meta_path: Path) -> tuple[list[str], dict[str, float], dict[str, str]]:
    item_pool: list[str] = []
    popularity: dict[str, float] = {}
    category: dict[str, str] = {}
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
            asin = str(asin)
            item_pool.append(asin)
            popularity[asin] = float(obj.get("rating_number") or 0.0)
            cats = obj.get("categories") or obj.get("main_category") or "Unknown"
            if isinstance(cats, list):
                category[asin] = "|".join(map(str, cats)) if cats else "Unknown"
            else:
                category[asin] = str(cats)
    return sorted(set(item_pool)), popularity, category


def make_position_proxy(popularity: dict[str, float]):
    pop = pd.Series(popularity, dtype=float)
    if len(pop) == 0:
        q33 = q66 = 0.0
    else:
        q33, q66 = pop.quantile([0.33, 0.66])

    def _position(item_id: str) -> str:
        value = popularity.get(item_id, 0.0)
        if value > q66:
            return "PopularityProxy_Top"
        if value > q33:
            return "PopularityProxy_Mid"
        return "PopularityProxy_Tail"

    return _position


def read_purchase_records(review_path: Path, get_position, category: dict[str, str]) -> pd.DataFrame:
    rows: list[dict] = []
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_id = obj.get("user_id")
            item_id = obj.get("parent_asin")
            timestamp = obj.get("timestamp")
            if not user_id or not item_id or timestamp is None:
                continue
            item_id = str(item_id)
            rows.append(
                {
                    "user_id": str(user_id),
                    "item_id": item_id,
                    "timestamp": int(timestamp),
                    "y_purchase": 1,
                    "y_click": np.nan,
                    "y_cart": np.nan,
                    "rating": float(obj.get("rating") or np.nan),
                    "verified_purchase": obj.get("verified_purchase"),
                    "position": get_position(item_id),
                    "category_20260713": category.get(item_id, "Unknown"),
                    "outcome_source": "observed_review_rating_record",
                    "is_sampled_nonpurchase": 0,
                    "amazon_click_available": 0,
                    "amazon_cart_available": 0,
                    "amazon_purchase_definition_20260713": "review_rating_observed_bottom_funnel_purchase_signal",
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.drop_duplicates(["user_id", "item_id", "timestamp"])
        .sort_values(["user_id", "timestamp", "item_id"])
        .reset_index(drop=True)
    )


def add_sampled_nonpurchase_alternatives(pos_df: pd.DataFrame, item_pool: list[str], get_position, category: dict[str, str]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    pool = np.array(item_pool, dtype=object)
    user_seen = pos_df.groupby("user_id")["item_id"].apply(set).to_dict()
    negatives: list[dict] = []
    for row in pos_df.itertuples(index=False):
        seen = user_seen.get(row.user_id, set())
        added = 0
        attempts = 0
        while added < NEG_PER_PURCHASE and attempts < NEG_PER_PURCHASE * 30:
            attempts += 1
            item = str(rng.choice(pool))
            if item in seen:
                continue
            negatives.append(
                {
                    "user_id": row.user_id,
                    "item_id": item,
                    "timestamp": int(row.timestamp),
                    "y_purchase": 0,
                    "y_click": np.nan,
                    "y_cart": np.nan,
                    "rating": np.nan,
                    "verified_purchase": np.nan,
                    "position": get_position(item),
                    "category_20260713": category.get(item, "Unknown"),
                    "outcome_source": "sampled_nonpurchase_alternative",
                    "is_sampled_nonpurchase": 1,
                    "amazon_click_available": 0,
                    "amazon_cart_available": 0,
                    "amazon_purchase_definition_20260713": "review_rating_observed_bottom_funnel_purchase_signal",
                }
            )
            added += 1
    out = pd.concat([pos_df, pd.DataFrame(negatives)], ignore_index=True)
    out = (
        out.groupby(["user_id", "item_id", "timestamp"], as_index=False)
        .agg(
            {
                "y_purchase": "max",
                "y_click": "first",
                "y_cart": "first",
                "rating": "first",
                "verified_purchase": "first",
                "position": "first",
                "category_20260713": "first",
                "outcome_source": "first",
                "is_sampled_nonpurchase": "max",
                "amazon_click_available": "first",
                "amazon_cart_available": "first",
                "amazon_purchase_definition_20260713": "first",
            }
        )
        .sort_values(["user_id", "timestamp", "item_id"])
        .reset_index(drop=True)
    )
    return out


def diagnostics(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"dataset": dataset, "metric": "rows", "value": int(len(df))},
            {"dataset": dataset, "metric": "users", "value": int(df["user_id"].nunique())},
            {"dataset": dataset, "metric": "items", "value": int(df["item_id"].nunique())},
            {"dataset": dataset, "metric": "purchase_positive_records", "value": int(df["y_purchase"].sum())},
            {"dataset": dataset, "metric": "sampled_nonpurchase_records", "value": int(df["is_sampled_nonpurchase"].sum())},
            {"dataset": dataset, "metric": "native_click_available", "value": 0},
            {"dataset": dataset, "metric": "native_cart_available", "value": 0},
        ]
    )


def process_dataset(dataset: str) -> None:
    start = time.time()
    logger.info("=" * 80)
    logger.info("Processing Amazon purchase-only outcome: %s", dataset)
    meta_path, review_path = raw_paths(dataset)
    logger.info("Meta path: %s", meta_path)
    logger.info("Review path: %s", review_path)
    out_dir = ensure_dir(OUT_BASE / dataset / "Y")

    item_pool, popularity, category = read_meta(meta_path)
    get_position = make_position_proxy(popularity)
    positives = read_purchase_records(review_path, get_position, category)
    logger.info("Observed purchase positives from review/rating records: %d", len(positives))
    y = add_sampled_nonpurchase_alternatives(positives, item_pool, get_position, category)

    write_table(y, out_dir / "y_behavior_20260713.parquet", logger)
    diagnostics(dataset, y).to_csv(out_dir / "amazon_purchase_outcome_diagnostics_20260713.csv", index=False)
    logger.info("Completed %s in %.2f seconds", dataset, time.time() - start)


def main() -> None:
    for dataset in DATASETS:
        process_dataset(dataset)


if __name__ == "__main__":
    main()
