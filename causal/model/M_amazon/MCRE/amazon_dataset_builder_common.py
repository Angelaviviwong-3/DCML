#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Amazon purchase-only DCML dataset builder for 20260717 wrappers."""

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

from dcml_utils import (
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
LOG_DIR = ensure_dir(_dcml_paths.workspace_path(ROOT, 'log'))
VERSION = "20260717"


def canonical_day_timestamp(values: pd.Series) -> pd.Series:
    """Normalize YYYYMMDD, epoch-second, and epoch-millisecond values."""
    raw = pd.Series(values, index=values.index)
    numeric = pd.to_numeric(raw, errors="coerce")
    out = pd.Series(np.nan, index=raw.index, dtype=float)
    ymd = numeric.between(19000101, 21001231)
    millis = numeric >= 1e11
    seconds = numeric.between(1e8, 1e11, inclusive="left")
    out.loc[ymd] = numeric.loc[ymd]
    if millis.any():
        parsed = pd.to_datetime(numeric.loc[millis], unit="ms", errors="coerce", utc=True)
        out.loc[millis] = pd.to_numeric(parsed.dt.strftime("%Y%m%d"), errors="coerce")
    if seconds.any():
        parsed = pd.to_datetime(numeric.loc[seconds], unit="s", errors="coerce", utc=True)
        out.loc[seconds] = pd.to_numeric(parsed.dt.strftime("%Y%m%d"), errors="coerce")
    remaining = out.isna()
    if remaining.any():
        parsed = pd.to_datetime(raw.loc[remaining], errors="coerce", utc=True)
        out.loc[remaining] = pd.to_numeric(parsed.dt.strftime("%Y%m%d"), errors="coerce")
    return out


def treatment_path(base_dir: Path, dataset: str) -> Path:
    filename = {
        "amazon_appliances": "amazon_appliances_item_multimodal_scalars_merged_full_2.parquet",
        "amazon_beauty": "amazon_beauty_item_multimodal_scalars_merged_full.parquet",
    }[dataset]
    return base_dir / "item_multimodal_scalars" / filename


def prefer_versioned(
    base_dir: Path,
    folder: str,
    stem: str,
    versions: tuple[str | None, ...],
) -> Path:
    for version in versions:
        filename = f"{stem}_{version}.parquet" if version else f"{stem}.parquet"
        candidate = base_dir / folder / filename
        if candidate.exists() or candidate.with_suffix(".csv").exists():
            return candidate
    tried = ", ".join(v or "original" for v in versions)
    raise FileNotFoundError(f"Cannot find {stem} in {base_dir / folder}; tried {tried}")


def add_pre_exposure_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    time_counts = (
        out.groupby(["user_id", "timestamp"], as_index=False)
        .agg(events_at_t=("item_id", "size"), purchases_at_t=("y_purchase", "sum"))
        .sort_values(["user_id", "timestamp"])
    )
    for source in ["events_at_t", "purchases_at_t"]:
        target = source.replace("_at_t", "_pre")
        time_counts[target] = (
            time_counts.groupby("user_id")[source]
            .cumsum()
            .groupby(time_counts["user_id"])
            .shift(fill_value=0)
        )
    time_counts["user_history_max_timestamp_pre"] = time_counts.groupby("user_id")["timestamp"].shift()
    out = out.merge(
        time_counts[
            [
                "user_id",
                "timestamp",
                "events_pre",
                "purchases_pre",
                "user_history_max_timestamp_pre",
            ]
        ],
        on=["user_id", "timestamp"],
        how="left",
    )
    out["events_pre"] = out["events_pre"].fillna(0)
    out["purchases_pre"] = out["purchases_pre"].fillna(0)
    out["user_events_pre_log"] = np.log1p(out["events_pre"])
    out["user_purchases_pre_log"] = np.log1p(out["purchases_pre"])
    return out


def add_category_semantic_affinity(df: pd.DataFrame) -> pd.DataFrame:
    """Historical purchase share for the candidate item's taxonomy category."""
    out = df.copy()
    user_time = (
        out.groupby(["user_id", "timestamp"], as_index=False)["y_purchase"]
        .sum()
        .rename(columns={"y_purchase": "user_purchase_at_t"})
        .sort_values(["user_id", "timestamp"])
    )
    user_time["user_purchase_before_t"] = (
        user_time.groupby("user_id")["user_purchase_at_t"]
        .cumsum()
        .groupby(user_time["user_id"])
        .shift(fill_value=0)
    )
    category_time = (
        out.groupby(["user_id", f"category_{VERSION}", "timestamp"], as_index=False)["y_purchase"]
        .sum()
        .rename(columns={"y_purchase": "user_category_purchase_at_t"})
        .sort_values(["user_id", f"category_{VERSION}", "timestamp"])
    )
    category_time["user_category_purchase_before_t"] = (
        category_time.groupby(["user_id", f"category_{VERSION}"])["user_category_purchase_at_t"]
        .cumsum()
        .groupby([category_time["user_id"], category_time[f"category_{VERSION}"]])
        .shift(fill_value=0)
    )
    out = out.merge(
        user_time[["user_id", "timestamp", "user_purchase_before_t"]],
        on=["user_id", "timestamp"],
        how="left",
    )
    out = out.merge(
        category_time[
            [
                "user_id",
                f"category_{VERSION}",
                "timestamp",
                "user_category_purchase_before_t",
            ]
        ],
        on=["user_id", f"category_{VERSION}", "timestamp"],
        how="left",
    )
    denominator = pd.to_numeric(out["user_purchase_before_t"], errors="coerce")
    numerator = pd.to_numeric(out["user_category_purchase_before_t"], errors="coerce").fillna(0.0)
    available = denominator > 0
    out["T_int_sem_available"] = available.astype(int)
    out["T_int_sem_history_denominator"] = denominator.fillna(0.0)
    out["T_int_sem_history_numerator"] = numerator
    out["T_int_sem"] = np.where(available, numerator / denominator, np.nan)
    out["T_int_sem"] = pd.to_numeric(out["T_int_sem"], errors="coerce").clip(0, 1)
    return out


def add_item_rating_history(events: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
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
    right = ratings[
        ["item_id", "timestamp", "rating_count_cum", "item_rating_mean_cum"]
    ].sort_values("timestamp")
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


def resolve_treatment_specs(t_cols: list[str]) -> tuple[list[str], list[str]]:
    def find(base: str) -> str:
        candidates = [c for c in t_cols if c == base or c.startswith(f"{base}_")]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one calibrated column for {base}, found {candidates}")
        return candidates[0]

    aggregate = [
        find("T_con_mkt"),
        "T_con_soc",
        "T_con_rat",
        find("T_int_fac"),
        find("T_int_vis"),
        "T_int_sem",
    ]
    disaggregate = [
        find("atomic_a_pri"),
        find("atomic_a_gft"),
        find("atomic_a_sub"),
        find("atomic_a_urg"),
        "T_con_soc",
        "T_con_rat",
        find("atomic_a_spec"),
        find("atomic_a_str"),
        find("T_int_fac"),
        "T_int_sem",
    ]
    return aggregate, disaggregate


def x_temporal_audit(set_c: pd.DataFrame) -> pd.DataFrame:
    history_required = set_c["purchases_pre"] > 0
    history_source = pd.to_numeric(set_c["user_history_max_timestamp_pre"], errors="coerce")
    event_time = pd.to_numeric(set_c["timestamp"], errors="coerce")
    history_valid = (~history_required) | (
        history_source < event_time
    )
    return pd.DataFrame(
        [
            {
                "x_column": "user_purchases_pre_log",
                "source_type": "strictly_pre_event_user_history",
                "temporal_test": "source_timestamp_lt_event_timestamp",
                "audited_rows": int(len(set_c)),
                "max_source_timestamp": float(history_source.max()),
                "min_event_timestamp": float(event_time.min()),
                "violations": int((~history_valid).sum()),
                "pass": bool(history_valid.all()),
            },
            {
                "x_column": "x_pri_log",
                "source_type": "time_invariant_item_metadata",
                "temporal_test": "not_applicable_static_metadata",
                "audited_rows": int(len(set_c)),
                "violations": 0,
                "pass": True,
                "note": "Static metadata exemption; no behavioral future aggregation is used.",
            },
        ]
    )


def build_amazon_dataset(dataset: str) -> None:
    if dataset not in {"amazon_appliances", "amazon_beauty"}:
        raise ValueError(dataset)
    logger = setup_logging(LOG_DIR, f"dataset_builder_{dataset}_{VERSION}")
    start = time.time()
    base_dir = _dcml_paths.workspace_path(ROOT, 'processed_data') / dataset
    save_dir = ensure_dir(base_dir / "build_dataset")
    y_path = prefer_versioned(base_dir, "Y", "y_behavior", ("20260715", "20260713", None))
    item_path = prefer_versioned(base_dir, "item_feature&Confounder_price", "item_text_price_matrix", ("20260713", None))
    rating_path = prefer_versioned(base_dir, "rating", "user_item_rating", ("20260713", None))
    t_path = treatment_path(base_dir, dataset)
    logger.info("Building %s purchase-only dataset, version=%s", dataset, VERSION)
    logger.info("No full-period Amazon user aggregates are loaded.")

    df_y = read_table(y_path)
    df_item = read_table(item_path)
    df_rating = read_table(rating_path)
    df_t = read_table(t_path)
    for frame in [df_y, df_item, df_rating, df_t]:
        normalize_ids(frame, ["user_id", "item_id"])
    df_y["timestamp"] = canonical_day_timestamp(df_y["timestamp"])
    df_rating["timestamp"] = canonical_day_timestamp(df_rating["timestamp"])
    if df_y["timestamp"].isna().any() or df_rating["timestamp"].isna().any():
        raise RuntimeError(f"{dataset}: timestamp normalization produced missing values")
    df_y["y_purchase"] = pd.to_numeric(df_y["y_purchase"], errors="coerce").fillna(0).astype(int)

    old_category_cols = [c for c in df_y.columns if c.startswith("category_")]
    if old_category_cols:
        df_y = df_y.drop(columns=old_category_cols)
    category = extract_category(df_item, out_col=f"category_{VERSION}")
    if "x_pri_log" not in df_item.columns:
        df_item["x_pri_log"] = 0.0
    df_t_clean, calibrated_cols = calibrate_canonical_mllm_treatments(df_t, seed=20260717)

    master = df_y.merge(
        df_item[["item_id", "x_pri_log"]].drop_duplicates("item_id"),
        on="item_id",
        how="inner",
    )
    master = master.merge(category, on="item_id", how="left")
    master[f"category_{VERSION}"] = master[f"category_{VERSION}"].fillna("Unknown")
    master = master.merge(df_t_clean, on="item_id", how="inner")
    master = add_pre_exposure_history(master)
    master = add_category_semantic_affinity(master)
    master = add_item_rating_history(master, df_rating)
    master["T_con_soc_raw"] = np.log1p(master["item_review_count_pre"].fillna(0))
    master["T_con_soc"] = zero_preserved_rank_norm(master["T_con_soc_raw"], seed=20260717)
    master["T_con_rat"] = ((master["item_rating_mean_pre"].fillna(3.0) - 1.0) / 4.0).clip(0, 1)
    master["M_norm"] = master["T_con_soc_raw"]

    master_sorted, set_a_tmp, set_b_tmp, set_c_tmp = temporal_abc_split_without_time_overlap(master, "timestamp")
    train_tmp = pd.concat([set_a_tmp, set_b_tmp], ignore_index=True)
    master_sorted["H_level"] = assign_pre_history_h_levels(
        train_tmp["user_purchases_pre_log"],
        master_sorted["user_purchases_pre_log"],
    ).to_numpy()

    all_treatments = calibrated_cols + ["T_int_sem", "T_con_soc", "T_con_rat"]
    if len(all_treatments) != 12:
        raise RuntimeError(f"Expected 12 measurement treatments, found {all_treatments}")
    aggregate, disaggregate = resolve_treatment_specs(calibrated_cols)
    x_whitelist = ["x_pri_log", "user_purchases_pre_log"]
    gs_x_whitelist = list(x_whitelist)
    metadata_cols = [
        "position",
        "outcome_source",
        "is_sampled_nonpurchase",
        "verified_purchase",
        "amazon_click_available",
        "amazon_cart_available",
    ]
    diagnostics = [
        "events_pre",
        "purchases_pre",
        "user_history_max_timestamp_pre",
        "item_review_count_pre",
        "item_rating_mean_pre",
        "T_con_soc_raw",
        "T_int_sem_available",
        "T_int_sem_history_denominator",
        "T_int_sem_history_numerator",
    ]
    final_cols = (
        ["user_id", "item_id", "timestamp", "y_purchase"]
        + all_treatments
        + ["M_norm", "H_level"]
        + x_whitelist
        + gs_x_whitelist
        + diagnostics
        + metadata_cols
    )
    final_cols = [c for c in dict.fromkeys(final_cols) if c in master_sorted.columns]
    master_sorted = master_sorted[final_cols]
    n_a, n_b = len(set_a_tmp), len(set_b_tmp)
    set_a = master_sorted.iloc[:n_a].copy()
    set_b = master_sorted.iloc[n_a : n_a + n_b].copy()
    set_c = master_sorted.iloc[n_a + n_b :].copy()

    prefix = f"{dataset}_DCML"
    for filename, frame in {
        f"{prefix}_Table_Master_{VERSION}.parquet": master_sorted,
        f"{prefix}_Set_A_{VERSION}.parquet": set_a,
        f"{prefix}_Set_B_{VERSION}.parquet": set_b,
        f"{prefix}_Set_C_{VERSION}.parquet": set_c,
    }.items():
        write_table(frame, save_dir / filename, logger)

    split_diagnostics = pd.DataFrame(
        [
            {
                "split": name,
                "rows": len(frame),
                "purchase_rate": float(frame["y_purchase"].mean()),
                "min_timestamp": frame["timestamp"].min(),
                "max_timestamp": frame["timestamp"].max(),
                "semantic_history_available_rate": float(frame["T_int_sem_available"].mean()),
            }
            for name, frame in [("All", master_sorted), ("Set_A", set_a), ("Set_B", set_b), ("Set_C", set_c)]
        ]
    )
    split_diagnostics.to_csv(save_dir / f"{prefix}_split_diagnostics_{VERSION}.csv", index=False)
    positive = master_sorted["y_purchase"] == 1
    if "verified_purchase" in master_sorted.columns:
        verified_text = master_sorted["verified_purchase"].astype(str).str.strip().str.lower()
        verified_true = positive & verified_text.isin({"true", "1", "1.0"})
        verified_false = positive & verified_text.isin({"false", "0", "0.0"})
        verified_missing = positive & ~(verified_true | verified_false)
    else:
        verified_true = pd.Series(False, index=master_sorted.index)
        verified_false = pd.Series(False, index=master_sorted.index)
        verified_missing = positive.copy()
    known_verified = int(verified_true.sum() + verified_false.sum())
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "observed_review_rating_positive_rows": int(positive.sum()),
                "verified_purchase_true_rows": int(verified_true.sum()),
                "verified_purchase_false_rows": int(verified_false.sum()),
                "verified_purchase_missing_rows": int(verified_missing.sum()),
                "verified_share_among_known": (
                    float(verified_true.sum() / known_verified) if known_verified else np.nan
                ),
                "sampled_nonpurchase_rows": int((master_sorted["y_purchase"] == 0).sum()),
                "primary_outcome_changed_by_verified_field": False,
                "interpretation": "Label-composition audit only; the primary sampled-choice Purchase outcome is unchanged.",
            }
        ]
    ).to_csv(save_dir / f"{prefix}_purchase_label_audit_{VERSION}.csv", index=False)
    audit = x_temporal_audit(set_c)
    audit.to_csv(save_dir / f"{prefix}_X_temporal_audit_{VERSION}.csv", index=False)
    if not audit["pass"].all():
        raise RuntimeError(f"{dataset} X temporal audit failed:\n{audit.to_string(index=False)}")

    write_json(
        {
            "dataset": dataset,
            "script_version": VERSION,
            "outcomes": ["y_purchase"],
            "unavailable_outcomes": ["y_click", "y_cart"],
            "all_treatments": all_treatments,
            "aggregate_treatments": aggregate,
            "disaggregate_treatments": disaggregate,
            "confounder_whitelist": x_whitelist,
            "gs_confounder_whitelist": gs_x_whitelist,
            "excluded_amazon_position_proxy": "The legacy popularity-derived position bucket is not a native exposure position and is excluded from X.",
            "semantic_construct": "Behaviorally Revealed Category-Semantic Affinity",
            "semantic_formula": "prior purchases in candidate category / all prior purchases",
            "semantic_missing_policy": "T_int_sem is NaN and non-estimable when no prior purchase history exists",
            "mcre_source": str(t_path),
            "mcre_temporal_audit_required": f"processed_data/{dataset}/audit/mcre_temporal_snapshot_summary_{VERSION}.json",
            "input_y_source": str(y_path),
            "split": "chronological 20/40/40 without timestamp-boundary overlap",
            "H_level": "0=no prior observed Purchase; 1/2/3=positive prior-Purchase tertiles fitted on Set A+B; sampled alternatives are excluded from the moderator definition",
            "outcome_design": "Observed review/rating records are Purchase positives; sampled unseen-item alternatives are comparison rows.",
            "estimand_scope": "Purchase-stage sampled-choice external validation within the constructed candidate set.",
            "claim_boundary": "No native Click/Cart, full-funnel reversal, population checkout conversion, or policy/OPE claim.",
        },
        save_dir / f"{prefix}_schema_{VERSION}.json",
    )
    logger.info("Completed %s in %.2f seconds", dataset, time.time() - start)
