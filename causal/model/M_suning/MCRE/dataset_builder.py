#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suning DCML dataset builder, 2026-07-12 correction.

Corrections:
1. Keep exactly 12 treatment variables; no ``*_calibrated_calib`` duplicates.
2. Use timestamp-block chronological split, so A/B/C do not share boundary dates.
3. Define H_level as 0=no pre-exposure history and 1/2/3=positive-history tertiles.
4. Retain raw social-proof columns for interpretation, but later DML scripts exclude
   them from X because they are treatment aliases of T_con_soc.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths


import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


CAUSAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CAUSAL_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dcml_utils import (
    CANONICAL_STRUCTURED_TREATMENTS,
    assign_pre_history_h_levels,
    calibrate_canonical_mllm_treatments,
    ensure_dir,
    extract_category,
    find_causal_root,
    normalize_ids,
    numeric_timestamp,
    read_table,
    setup_logging,
    temporal_abc_split_without_time_overlap,
    write_json,
    write_table,
    zero_preserved_rank_norm,
)

ROOT = find_causal_root(__file__)
BASE_DIR = _dcml_paths.workspace_path(ROOT, 'processed_data') / "suning"
SAVE_DIR = ensure_dir(BASE_DIR / "build_dataset")
LOG_DIR = ensure_dir(_dcml_paths.workspace_path(ROOT, 'log'))
logger = setup_logging(LOG_DIR, "dataset_builder_suning_temporal_20260712")

Y_PATH = BASE_DIR / "Y" / "y_behavior_20260712.parquet"
USER_PATH = BASE_DIR / "Confounder_user" / "confounder_user_matrix.parquet"
ITEM_PATH = BASE_DIR / "item_feature&Confounder_price" / "item_text_price_matrix.parquet"
RATING_PATH = BASE_DIR / "rating" / "user_item_rating.parquet"
TREATMENT_PATH = BASE_DIR / "item_multimodal_scalars" / "item_multimodal_scalars_full_3.parquet"


def add_pre_exposure_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add user history counts observed strictly before each event timestamp."""
    out = df.copy()
    time_counts = (
        out.groupby(["user_id", "timestamp"], as_index=False)
        .agg(
            events_at_t=("item_id", "size"),
            clicks_at_t=("y_click", "sum"),
            carts_at_t=("y_cart", "sum"),
            purchases_at_t=("y_purchase", "sum"),
        )
        .sort_values(["user_id", "timestamp"])
    )
    for col in ["events_at_t", "clicks_at_t", "carts_at_t", "purchases_at_t"]:
        before_col = col.replace("_at_t", "_pre")
        time_counts[before_col] = (
            time_counts.groupby("user_id")[col]
            .cumsum()
            .groupby(time_counts["user_id"])
            .shift(fill_value=0)
        )
    out = out.merge(
        time_counts[
            ["user_id", "timestamp", "events_pre", "clicks_pre", "carts_pre", "purchases_pre"]
        ],
        on=["user_id", "timestamp"],
        how="left",
    )
    for col in ["events_pre", "clicks_pre", "carts_pre", "purchases_pre"]:
        out[col] = out[col].fillna(0)
        out[f"user_{col}_log"] = np.log1p(out[col])
    return out


def add_pre_exposure_semantic_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute user's historical click share for the focal category before t."""
    out = df.copy()
    user_time = (
        out.groupby(["user_id", "timestamp"], as_index=False)["y_click"]
        .sum()
        .rename(columns={"y_click": "user_clicks_at_t"})
        .sort_values(["user_id", "timestamp"])
    )
    user_time["user_clicks_before_t"] = (
        user_time.groupby("user_id")["user_clicks_at_t"]
        .cumsum()
        .groupby(user_time["user_id"])
        .shift(fill_value=0)
    )
    user_cat_time = (
        out.groupby(["user_id", "category_20260712", "timestamp"], as_index=False)["y_click"]
        .sum()
        .rename(columns={"y_click": "user_cat_clicks_at_t"})
        .sort_values(["user_id", "category_20260712", "timestamp"])
    )
    user_cat_time["user_cat_clicks_before_t"] = (
        user_cat_time.groupby(["user_id", "category_20260712"])["user_cat_clicks_at_t"]
        .cumsum()
        .groupby([user_cat_time["user_id"], user_cat_time["category_20260712"]])
        .shift(fill_value=0)
    )
    out = out.merge(
        user_time[["user_id", "timestamp", "user_clicks_before_t"]],
        on=["user_id", "timestamp"],
        how="left",
    )
    out = out.merge(
        user_cat_time[["user_id", "category_20260712", "timestamp", "user_cat_clicks_before_t"]],
        on=["user_id", "category_20260712", "timestamp"],
        how="left",
    )
    denom = out["user_clicks_before_t"].replace(0, np.nan)
    out["T_int_sem"] = (out["user_cat_clicks_before_t"] / denom).fillna(0.01).clip(0, 1)
    return out


def add_pre_exposure_item_rating_features(events: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Use only ratings/reviews strictly before each event timestamp."""
    out = events.copy()
    ratings = ratings.copy()
    if ratings.empty:
        out["item_review_count_pre"] = 0
        out["item_rating_mean_pre"] = np.nan
        return out
    ratings["timestamp"] = numeric_timestamp(ratings["timestamp"])
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["item_id", "timestamp"]).sort_values(["item_id", "timestamp"])
    ratings["rating_count_cum"] = ratings.groupby("item_id").cumcount() + 1
    ratings["rating_sum_cum"] = ratings.groupby("item_id")["rating"].cumsum()
    ratings["item_rating_mean_cum"] = ratings["rating_sum_cum"] / ratings["rating_count_cum"]
    right = ratings[["item_id", "timestamp", "rating_count_cum", "item_rating_mean_cum"]].sort_values("timestamp")
    left = out.reset_index(names="_row_id").sort_values("timestamp")
    merged = pd.merge_asof(
        left,
        right,
        on="timestamp",
        by="item_id",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row_id")
    out["item_review_count_pre"] = merged["rating_count_cum"].fillna(0).to_numpy()
    out["item_rating_mean_pre"] = merged["item_rating_mean_cum"].to_numpy()
    return out


def main() -> None:
    start = time.time()
    logger.info("=" * 80)
    logger.info("Building Suning temporal/pre-exposure DCML dataset, 20260712")

    df_y = read_table(Y_PATH)
    df_user = read_table(USER_PATH)
    df_item = read_table(ITEM_PATH)
    df_rating = read_table(RATING_PATH)
    df_t = read_table(TREATMENT_PATH)

    for df in [df_y, df_user, df_item, df_rating, df_t]:
        normalize_ids(df, ["user_id", "item_id"])
    df_y["timestamp"] = numeric_timestamp(df_y["timestamp"])
    df_rating["timestamp"] = numeric_timestamp(df_rating["timestamp"])
    for col in ["y_click", "y_cart", "y_purchase"]:
        df_y[col] = pd.to_numeric(df_y[col], errors="coerce").fillna(0).astype(int)

    category = extract_category(df_item)
    if "x_pri_log" not in df_item.columns:
        df_item["x_pri_log"] = 0.0

    df_t_calib, calibrated_vars = calibrate_canonical_mllm_treatments(df_t)
    all_treatments = calibrated_vars + CANONICAL_STRUCTURED_TREATMENTS
    if len(all_treatments) != 12:
        logger.warning("Expected 12 treatments, got %d: %s", len(all_treatments), all_treatments)

    master = df_y.merge(df_user, on="user_id", how="inner")
    master = master.merge(df_item[["item_id", "x_pri_log"]].drop_duplicates("item_id"), on="item_id", how="inner")
    master = master.merge(category, on="item_id", how="left")
    master["category_20260712"] = master["category_20260712"].fillna("Unknown")
    master = master.merge(df_t_calib, on="item_id", how="inner")

    master = add_pre_exposure_user_features(master)
    master = add_pre_exposure_semantic_alignment(master)
    master = add_pre_exposure_item_rating_features(master, df_rating)

    master["T_con_soc_raw"] = np.log1p(master["item_review_count_pre"].fillna(0))
    master["T_con_soc"] = zero_preserved_rank_norm(master["T_con_soc_raw"], seed=20260712)
    q1, q3 = np.nanpercentile(master["T_con_soc_raw"], [25, 75])
    iqr = q3 - q1
    master["T_con_soc_iqr"] = 0.0 if not np.isfinite(iqr) or iqr <= 0 else (master["T_con_soc_raw"] - q1) / iqr
    master["T_con_rat"] = ((master["item_rating_mean_pre"].fillna(3.0) - 1.0) / 4.0).clip(0, 1)
    master["M_norm"] = master["T_con_soc_raw"]

    master_sorted, set_a_tmp, set_b_tmp, set_c_tmp = temporal_abc_split_without_time_overlap(master, "timestamp")
    train_tmp = pd.concat([set_a_tmp, set_b_tmp], ignore_index=True)

    pos_map = train_tmp["position"].value_counts(normalize=True).to_dict()
    master_sorted["x_pos_freq"] = master_sorted["position"].map(pos_map).fillna(0.0)
    master_sorted["H_level"] = assign_pre_history_h_levels(
        train_tmp["user_events_pre_log"],
        master_sorted["user_events_pre_log"],
    ).to_numpy()

    user_conf_cols = [c for c in df_user.columns if c != "user_id" and c in master_sorted.columns]
    diagnostic_cols = [
        "user_events_pre_log",
        "user_clicks_pre_log",
        "user_carts_pre_log",
        "user_purchases_pre_log",
        "item_review_count_pre",
        "item_rating_mean_pre",
        "T_con_soc_raw",
        "T_con_soc_iqr",
    ]
    final_cols = (
        ["user_id", "item_id", "timestamp", "y_click", "y_cart", "y_purchase"]
        + all_treatments
        + ["M_norm", "H_level", "x_pri_log", "x_pos_freq"]
        + user_conf_cols
        + diagnostic_cols
    )
    final_cols = [c for c in dict.fromkeys(final_cols) if c in master_sorted.columns]
    master_sorted = master_sorted[final_cols]

    n_a, n_b = len(set_a_tmp), len(set_b_tmp)
    set_a = master_sorted.iloc[:n_a].copy()
    set_b = master_sorted.iloc[n_a:n_a + n_b].copy()
    set_c = master_sorted.iloc[n_a + n_b:].copy()

    for filename, data in {
        "DCML_Table_20260712.parquet": master_sorted,
        "DCML_A_20260712.parquet": set_a,
        "DCML_B_20260712.parquet": set_b,
        "DCML_C_20260712.parquet": set_c,
    }.items():
        write_table(data, SAVE_DIR / filename, logger)

    diagnostics = pd.DataFrame(
        [
            {"metric": "n_master", "value": len(master_sorted)},
            {"metric": "n_set_a", "value": len(set_a)},
            {"metric": "n_set_b", "value": len(set_b)},
            {"metric": "n_set_c", "value": len(set_c)},
            {"metric": "set_a_min_timestamp", "value": set_a["timestamp"].min()},
            {"metric": "set_a_max_timestamp", "value": set_a["timestamp"].max()},
            {"metric": "set_b_min_timestamp", "value": set_b["timestamp"].min()},
            {"metric": "set_b_max_timestamp", "value": set_b["timestamp"].max()},
            {"metric": "set_c_min_timestamp", "value": set_c["timestamp"].min()},
            {"metric": "set_c_max_timestamp", "value": set_c["timestamp"].max()},
            {"metric": "h_level_counts", "value": master_sorted["H_level"].value_counts().sort_index().to_dict()},
            {"metric": "treatment_count", "value": len(all_treatments)},
            {"metric": "treatments", "value": "|".join(all_treatments)},
        ]
    )
    diagnostics.to_csv(SAVE_DIR / "DCML_temporal_split_diagnostics_20260712.csv", index=False)
    write_json(
        {
            "all_treatments": all_treatments,
            "calibrated_mllm_treatments": calibrated_vars,
            "outcomes": ["y_click", "y_cart", "y_purchase"],
            "split": "chronological 20/40/40 without timestamp overlap",
            "H_level": "0=no pre-exposure history; 1/2/3=positive-history tertiles from Set A+B",
            "social_signal": "T_con_soc is ZPRN(log1p(item_review_count_before_t)); raw variants are excluded from DML X.",
        },
        SAVE_DIR / "DCML_dataset_schema_20260712.json",
    )

    logger.info("Completed in %.2f seconds", time.time() - start)
    logger.info("Treatments (%d): %s", len(all_treatments), all_treatments)
    logger.info("H counts: %s", master_sorted["H_level"].value_counts().sort_index().to_dict())
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
