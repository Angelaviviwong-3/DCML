#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared temporally certified analysis-sample contract for 20260717.

The full historical MCRE audit may fail. Downstream causal and predictive
analyses remain admissible only after every split is restricted to items whose
selected review inputs strictly precede that item's first Set C event.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[0]))
import project_paths as _dcml_paths


import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from dcml_utils import ensure_dir, normalize_ids, write_json


VERSION = "20260717"
CERTIFIED_STATUS = "CERTIFIED_PRE_SET_C"
ANALYSIS_STATUS = "PASS_BY_CONSTRUCTION_CERTIFIED_ITEMS_ONLY"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def certification_paths(root: Path, dataset: str) -> tuple[Path, Path]:
    audit_dir = _dcml_paths.workspace_path(root, 'processed_data') / dataset / "audit"
    return (
        audit_dir / f"mcre_temporal_snapshot_summary_{VERSION}.json",
        audit_dir / f"mcre_temporal_snapshot_item_certification_{VERSION}.csv",
    )


def load_certification(root: Path, dataset: str) -> tuple[set[str], dict, pd.DataFrame]:
    summary_path, item_path = certification_paths(root, dataset)
    if not summary_path.is_file() or not item_path.is_file():
        raise FileNotFoundError(
            f"{dataset}: missing temporal certification inputs. Run "
            f"causal/model/audit_mcre_temporal_snapshot.py first. "
            f"Expected {summary_path} and {item_path}."
        )
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    item = pd.read_csv(item_path, dtype={"item_id": str})
    required = {"item_id", "strictly_pre_set_c", "certification_status"}
    if missing := sorted(required.difference(item.columns)):
        raise RuntimeError(f"{dataset}: certification table is missing columns: {missing}")
    normalize_ids(item, ["item_id"])
    if item["item_id"].isna().any() or item["item_id"].duplicated().any():
        raise RuntimeError(f"{dataset}: certification table contains missing or duplicate item IDs")
    item["strictly_pre_set_c"] = _as_bool(item["strictly_pre_set_c"])
    certified_mask = item["strictly_pre_set_c"] & item["certification_status"].eq(CERTIFIED_STATUS)
    inconsistent = item["strictly_pre_set_c"] ^ item["certification_status"].eq(CERTIFIED_STATUS)
    if inconsistent.any():
        raise RuntimeError(f"{dataset}: certification status and boolean disagree for {int(inconsistent.sum())} items")
    expected_items = int(summary.get("set_c_items", -1))
    expected_certified = int(summary.get("set_c_items_temporally_certified", -1))
    if expected_items != len(item) or expected_certified != int(certified_mask.sum()):
        raise RuntimeError(
            f"{dataset}: certification table does not match summary counts "
            f"(items {len(item)}/{expected_items}, certified {int(certified_mask.sum())}/{expected_certified})"
        )
    certified = set(item.loc[certified_mask, "item_id"].astype(str))
    if not certified:
        raise RuntimeError(f"{dataset}: temporal certification produced an empty item set")
    summary = dict(summary)
    summary.update(
        {
            "summary_file": str(summary_path),
            "item_certification_file": str(item_path),
            "item_certification_sha256": _sha256(item_path),
        }
    )
    return certified, summary, item


def filter_certified_frame(frame: pd.DataFrame, certified_items: set[str]) -> pd.DataFrame:
    if "item_id" not in frame.columns:
        raise RuntimeError("Cannot apply temporal item certification without item_id")
    out = frame.copy()
    normalize_ids(out, ["user_id", "item_id"])
    return out[out["item_id"].isin(certified_items)].copy()


def _split_diagnostic(name: str, before: pd.DataFrame, after: pd.DataFrame, outcomes: Sequence[str]) -> dict:
    row = {
        "split": name,
        "rows_before": int(len(before)),
        "rows_after": int(len(after)),
        "rows_removed": int(len(before) - len(after)),
        "row_retention_rate": float(len(after) / len(before)) if len(before) else np.nan,
        "items_before": int(before["item_id"].astype(str).nunique()),
        "items_after": int(after["item_id"].astype(str).nunique()),
    }
    if "timestamp" in after.columns and len(after):
        timestamp = pd.to_numeric(after["timestamp"], errors="coerce")
        row.update({"min_timestamp_after": float(timestamp.min()), "max_timestamp_after": float(timestamp.max())})
    for outcome in outcomes:
        if outcome in before.columns:
            row[f"{outcome}_rate_before"] = float(pd.to_numeric(before[outcome], errors="coerce").mean())
        if outcome in after.columns:
            row[f"{outcome}_rate_after"] = float(pd.to_numeric(after[outcome], errors="coerce").mean())
    return row


def _balance_table(
    set_c: pd.DataFrame,
    certified_items: set[str],
    columns: Sequence[str],
    roles: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    certified_mask = set_c["item_id"].astype(str).isin(certified_items)
    rows: list[dict] = []
    roles = roles or {}
    for column in dict.fromkeys(columns):
        if column not in set_c.columns:
            continue
        values = pd.to_numeric(set_c[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        retained = values[certified_mask].dropna()
        excluded = values[~certified_mask].dropna()
        if retained.empty and excluded.empty:
            continue
        retained_sd = float(retained.std(ddof=1)) if len(retained) > 1 else 0.0
        excluded_sd = float(excluded.std(ddof=1)) if len(excluded) > 1 else 0.0
        pooled_sd = float(np.sqrt((retained_sd**2 + excluded_sd**2) / 2.0))
        difference = float(retained.mean() - excluded.mean()) if len(retained) and len(excluded) else np.nan
        rows.append(
            {
                "column": column,
                "role": roles.get(column, "diagnostic"),
                "certified_n": int(len(retained)),
                "failed_n": int(len(excluded)),
                "certified_mean": float(retained.mean()) if len(retained) else np.nan,
                "failed_mean": float(excluded.mean()) if len(excluded) else np.nan,
                "certified_sd": retained_sd,
                "failed_sd": excluded_sd,
                "standardized_mean_difference": difference / pooled_sd if pooled_sd > 0 else np.nan,
                "interpretation": "descriptive selection diagnostic; not an outcome model or identification adjustment",
            }
        )
    return pd.DataFrame(rows)


def restrict_analysis_splits(
    root: Path,
    dataset: str,
    frames: Mapping[str, pd.DataFrame],
    output_dir: Path,
    output_prefix: str,
    outcomes: Sequence[str] = (),
    balance_columns: Sequence[str] = (),
    balance_roles: Mapping[str, str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Restrict every supplied split to one immutable certified item universe."""
    if "Set_C" not in frames:
        raise ValueError("frames must contain Set_C for certification validation")
    certified_items, audit, _ = load_certification(root, dataset)
    filtered: dict[str, pd.DataFrame] = {}
    diagnostics: list[dict] = []
    set_c_before: pd.DataFrame | None = None
    for name, frame in frames.items():
        original = frame
        normalize_ids(original, ["user_id", "item_id"])
        selected = original.loc[original["item_id"].isin(certified_items)].copy()
        if selected.empty:
            raise RuntimeError(f"{dataset} {name}: no rows remain after temporal certification")
        filtered[name] = selected
        diagnostics.append(_split_diagnostic(name, original, selected, outcomes))
        if name == "Set_C":
            set_c_before = original

    expected_c_rows = int(audit.get("set_c_rows_on_temporally_certified_items", -1))
    actual_c_rows = len(filtered["Set_C"])
    if expected_c_rows != actual_c_rows:
        raise RuntimeError(
            f"{dataset}: certified Set C rows disagree with audit: {actual_c_rows} != {expected_c_rows}"
        )

    output_dir = ensure_dir(output_dir)
    diagnostic_path = output_dir / f"{output_prefix}_temporal_certified_sample_diagnostics_{VERSION}.csv"
    manifest_path = output_dir / f"{output_prefix}_temporal_certified_sample_manifest_{VERSION}.json"
    balance_path = output_dir / f"{output_prefix}_temporal_certification_balance_{VERSION}.csv"
    pd.DataFrame(diagnostics).to_csv(diagnostic_path, index=False)
    if set_c_before is None:
        raise RuntimeError("Set_C disappeared during certification")
    balance = _balance_table(set_c_before, certified_items, balance_columns, balance_roles)
    balance.to_csv(balance_path, index=False)
    manifest = {
        "dataset": dataset,
        "script_version": VERSION,
        "analysis_temporal_status": ANALYSIS_STATUS,
        "full_sample_audit_status": audit.get("audit_status"),
        "analysis_population": "items with all historically selected MCRE reviews strictly before the item's first Set C event",
        "estimand_scope": "conditional effect among temporally certified items",
        "selection_is_not_assumed_random": True,
        "certified_item_count": int(len(certified_items)),
        "full_set_c_item_count": int(audit["set_c_items"]),
        "certified_set_c_rows": int(actual_c_rows),
        "full_set_c_rows": int(audit["set_c_rows"]),
        "set_c_item_certification_rate": float(audit["set_c_item_certification_rate"]),
        "set_c_row_certification_rate": float(audit["set_c_row_certification_rate"]),
        "certification_summary_file": audit["summary_file"],
        "certification_item_file": audit["item_certification_file"],
        "certification_item_sha256": audit["item_certification_sha256"],
        "split_diagnostics_file": str(diagnostic_path),
        "selection_balance_file": str(balance_path),
        "split_diagnostics": diagnostics,
    }
    write_json(manifest, manifest_path)
    manifest["manifest_file"] = str(manifest_path)
    return filtered, manifest
