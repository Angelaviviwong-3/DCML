#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit whether historical MCRE review inputs predate item entry into Set C.

This script does not load an MLLM and does not regenerate treatment scores. It
reproduces the historical extractor's item-level review selection rule (the
first five unique non-null review texts in stored row order), restricts the
audit to items appearing in Set C, and verifies that every selected review for
an item is strictly earlier than that item's first Set C event timestamp.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CAUSAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_utils import ensure_dir, find_causal_root, normalize_ids, numeric_timestamp, read_table


ROOT = find_causal_root(__file__)
VERSION = "20260717"


def canonical_day_timestamp(values: pd.Series) -> pd.Series:
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


def first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists() or candidate.with_suffix(".csv").exists():
            return candidate
    raise FileNotFoundError("No candidate file exists:\n" + "\n".join(map(str, candidates)))


def dataset_paths(dataset: str) -> tuple[Path, Path]:
    base = _dcml_paths.workspace_path(ROOT, 'processed_data') / dataset
    review_path = first_existing(
        [
            base / "pure_reviews" / "user_item_pure_reviews.parquet",
            base / "pure_reviews" / "user_item_pure_reviews_20260713.parquet",
        ]
    )
    if dataset == "suning":
        set_c_path = first_existing(
            [
                base / "build_dataset" / "DCML_C_20260712.parquet",
                base / "build_dataset" / "DCML_C_5.parquet",
            ]
        )
    else:
        prefix = f"{dataset}_DCML"
        set_c_path = first_existing(
            [
                base / "build_dataset" / f"{prefix}_Set_C_20260717.parquet",
                base / "build_dataset" / f"{prefix}_Set_C_20260715.parquet",
                base / "build_dataset" / f"{prefix}_Set_C_3.parquet",
                base / "build_dataset" / f"{prefix}_Set_C_2.parquet",
                base / "build_dataset" / f"{prefix}_Set_C_1.parquet",
            ]
        )
    return review_path, set_c_path


def audit_dataset(dataset: str) -> dict:
    review_path, set_c_path = dataset_paths(dataset)
    reviews = read_table(review_path)
    set_c = read_table(set_c_path)
    required_reviews = {"item_id", "review_text", "timestamp"}
    required_set_c = {"item_id", "timestamp"}
    if missing := sorted(required_reviews.difference(reviews.columns)):
        raise RuntimeError(f"{dataset} review input is missing columns: {missing}")
    if missing := sorted(required_set_c.difference(set_c.columns)):
        raise RuntimeError(f"{dataset} Set C is missing columns: {missing}")

    normalize_ids(reviews, ["item_id"])
    normalize_ids(set_c, ["item_id"])
    reviews["review_timestamp"] = canonical_day_timestamp(reviews["timestamp"])
    set_c["event_timestamp"] = canonical_day_timestamp(set_c["timestamp"])
    set_c_start = float(set_c["event_timestamp"].min())
    item_first_set_c = (
        set_c.groupby("item_id", as_index=False)["event_timestamp"]
        .min()
        .rename(columns={"event_timestamp": "item_first_set_c_timestamp"})
    )
    evaluation_items = set(set_c["item_id"].dropna().astype(str))

    # This intentionally matches the old extractor: unique texts, stored order,
    # then the first five rows per item. Do not sort by time in this audit.
    selected = reviews.dropna(subset=["item_id", "review_text"]).copy()
    selected["review_text"] = selected["review_text"].astype(str)
    selected = selected[selected["review_text"].str.strip().ne("")]
    selected = selected.drop_duplicates(["item_id", "review_text"], keep="first")
    selected = selected.groupby("item_id", sort=False, group_keys=False).head(5)
    selected = selected[selected["item_id"].astype(str).isin(evaluation_items)].copy()
    selected = selected.merge(item_first_set_c, on="item_id", how="left")
    selected["global_set_c_start_timestamp"] = set_c_start
    selected["timestamp_parse_ok"] = np.isfinite(selected["review_timestamp"])
    selected["strictly_pre_set_c"] = selected["timestamp_parse_ok"] & (
        selected["review_timestamp"] < selected["item_first_set_c_timestamp"]
    )

    violations = selected[~selected["strictly_pre_set_c"]].copy()
    selected_items = set(selected["item_id"].astype(str))
    output_dir = ensure_dir(_dcml_paths.workspace_path(ROOT, 'processed_data') / dataset / "audit")
    violation_path = output_dir / f"mcre_temporal_snapshot_violations_{VERSION}.csv"
    violation_cols = [
        "item_id",
        "user_id",
        "review_timestamp",
        "item_first_set_c_timestamp",
        "global_set_c_start_timestamp",
        "timestamp_parse_ok",
        "strictly_pre_set_c",
    ]
    violation_cols = [c for c in violation_cols if c in violations.columns]
    violations[violation_cols].to_csv(violation_path, index=False)

    selected_item_audit = (
        selected.groupby("item_id", as_index=False)
        .agg(
            selected_review_rows=("review_timestamp", "size"),
            min_selected_review_timestamp=("review_timestamp", "min"),
            max_selected_review_timestamp=("review_timestamp", "max"),
            timestamp_parse_ok=("timestamp_parse_ok", "all"),
            strictly_pre_set_c=("strictly_pre_set_c", "all"),
        )
    )
    item_audit = item_first_set_c.copy()
    item_audit = item_audit.merge(selected_item_audit, on="item_id", how="left")
    item_audit["selected_review_rows"] = item_audit["selected_review_rows"].fillna(0).astype(int)
    no_review = item_audit["selected_review_rows"].eq(0)
    item_audit["input_mode"] = np.where(no_review, "metadata_image_only", "metadata_image_plus_reviews")
    item_audit["timestamp_parse_ok"] = item_audit["timestamp_parse_ok"].fillna(True).astype(bool)
    item_audit["strictly_pre_set_c"] = item_audit["strictly_pre_set_c"].fillna(True).astype(bool)
    item_audit["certification_status"] = np.where(
        item_audit["strictly_pre_set_c"],
        "CERTIFIED_PRE_SET_C",
        "FAIL_POST_OR_AT_SET_C_REVIEW",
    )
    item_audit_path = output_dir / f"mcre_temporal_snapshot_item_certification_{VERSION}.csv"
    item_audit.to_csv(item_audit_path, index=False)
    certified_items = set(
        item_audit.loc[item_audit["strictly_pre_set_c"], "item_id"].astype(str)
    )
    set_c_certified_rows = int(set_c["item_id"].astype(str).isin(certified_items).sum())
    failed_items = int((~item_audit["strictly_pre_set_c"]).sum())

    summary = {
        "dataset": dataset,
        "script_version": VERSION,
        "audit_status": "PASS" if len(violations) == 0 else "FAIL",
        "review_input": str(review_path),
        "set_c_input": str(set_c_path),
        "selection_rule": "first five unique non-null review texts per item in stored row order",
        "set_c_start_timestamp": set_c_start,
        "set_c_rows": int(len(set_c)),
        "set_c_items": int(len(evaluation_items)),
        "set_c_items_with_selected_reviews": int(len(selected_items)),
        "set_c_items_without_review_text": int(len(evaluation_items.difference(selected_items))),
        "set_c_items_temporally_certified": int(len(certified_items)),
        "set_c_items_temporally_failed": failed_items,
        "set_c_item_certification_rate": float(len(certified_items) / len(evaluation_items)) if evaluation_items else np.nan,
        "set_c_rows_on_temporally_certified_items": set_c_certified_rows,
        "set_c_row_certification_rate": float(set_c_certified_rows / len(set_c)) if len(set_c) else np.nan,
        "selected_review_rows": int(len(selected)),
        "selected_review_timestamp_parse_failures": int((~selected["timestamp_parse_ok"]).sum()),
        "selected_reviews_not_strictly_pre_set_c": int(len(violations)),
        "pass_condition": "all selected review timestamps are finite and strictly earlier than the item's first Set C event",
        "interpretation_if_pass": "frozen pre-entry item-level content snapshot; no Set C outcome labels used",
        "interpretation_if_fail": "historical MCRE output is not temporally clean and cannot be certified by this audit",
        "violation_file": str(violation_path),
        "item_certification_file": str(item_audit_path),
    }
    pd.DataFrame([summary]).to_csv(
        output_dir / f"mcre_temporal_snapshot_summary_{VERSION}.csv",
        index=False,
    )
    with open(output_dir / f"mcre_temporal_snapshot_summary_{VERSION}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["all", "suning", "amazon_appliances", "amazon_beauty"],
        default="all",
    )
    parser.add_argument("--fail-on-violation", action="store_true")
    args = parser.parse_args()
    datasets = (
        ["suning", "amazon_appliances", "amazon_beauty"]
        if args.dataset == "all"
        else [args.dataset]
    )
    summaries = [audit_dataset(dataset) for dataset in datasets]
    print(
        pd.DataFrame(summaries)[
            [
                "dataset",
                "audit_status",
                "selected_review_rows",
                "selected_reviews_not_strictly_pre_set_c",
                "set_c_item_certification_rate",
                "set_c_row_certification_rate",
            ]
        ].to_string(index=False)
    )
    if args.fail_on_violation and any(row["audit_status"] != "PASS" for row in summaries):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
