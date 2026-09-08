#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared DCML helpers for the 20260713 Suning branch and Amazon reruns."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[0]))
import project_paths as _dcml_paths


import json
import logging
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


CANONICAL_MLLM_BASES = [
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

CANONICAL_STRUCTURED_TREATMENTS = ["T_int_sem", "T_con_soc", "T_con_rat"]

TREATMENT_DERIVED_ALIASES = {
    "T_con_soc_raw",
    "T_con_soc_iqr",
    "item_review_count_pre",
    "item_rating_mean_pre",
}


def find_causal_root(anchor_file: str | os.PathLike | None = None) -> Path:
    env_root = os.environ.get("DCML_CAUSAL_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if anchor_file is not None:
        p = Path(anchor_file).resolve()
        for parent in [p.parent] + list(p.parents):
            if parent.name == "causal":
                return parent
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if parent.name == "causal":
            return parent
        candidate = parent / "causal"
        if candidate.exists():
            return candidate.resolve()
    return Path(str(_dcml_paths.project_path('')))


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(log_dir: str | os.PathLike, stem: str) -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
    fh = logging.FileHandler(Path(log_dir) / f"{stem}.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _alternate_table_paths(path: str | os.PathLike) -> list[Path]:
    p = Path(path)
    if p.suffix == ".parquet":
        return [p, p.with_suffix(".csv")]
    if p.suffix == ".csv":
        return [p, p.with_suffix(".parquet")]
    return [p.with_suffix(".parquet"), p.with_suffix(".csv")]


def read_table(path: str | os.PathLike, **kwargs) -> pd.DataFrame:
    errors: list[str] = []
    for candidate in _alternate_table_paths(path):
        if not candidate.exists():
            continue
        if candidate.suffix == ".parquet":
            try:
                return pd.read_parquet(candidate, **kwargs)
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                continue
        if candidate.suffix == ".csv":
            for encoding in ("utf-8", "utf-8-sig", "gb18030"):
                try:
                    return pd.read_csv(candidate, encoding=encoding, **kwargs)
                except Exception as exc:
                    errors.append(f"{candidate} [{encoding}]: {exc}")
    raise FileNotFoundError(f"Could not read table from {path}. Tried errors: {errors[:3]}")


def write_table(df: pd.DataFrame, parquet_path: str | os.PathLike, logger: logging.Logger | None = None) -> None:
    p = Path(parquet_path)
    ensure_dir(p.parent)
    csv_path = p.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(p, index=False)
        if logger:
            logger.info("Saved parquet and csv: %s | %s", p, csv_path)
    except Exception as exc:
        if logger:
            logger.warning("Saved csv only because parquet writing failed: %s", exc)
            logger.info("CSV path: %s", csv_path)


def write_json(obj: Mapping, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_ids(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            if col == "item_id":
                df[col] = df[col].str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    return df


def numeric_timestamp(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().any():
        parsed = pd.to_datetime(series, errors="coerce")
        parsed_num = parsed.astype("int64") // 10**9
        out = out.fillna(parsed_num)
    return out.astype("float64")


def zero_preserved_rank_norm(values: Iterable, seed: int = 20260713) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = np.isfinite(x) & (x > 0)
    n_pos = int(mask.sum())
    if n_pos == 0:
        return out
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0, 1e-9, size=n_pos)
    ranks = pd.Series(x[mask] + jitter).rank(method="first").to_numpy()
    out[mask] = ranks / float(n_pos)
    return out


def extract_category(df_item: pd.DataFrame, out_col: str = "category_20260713") -> pd.DataFrame:
    out = df_item.copy()
    if "category" in out.columns:
        cat = out["category"]
    elif "cat_name" in out.columns:
        cat = out["cat_name"]
    elif "text_W" in out.columns:
        text = out["text_W"].fillna("").astype(str)
        cat = text.str.extract(r"(?:Category|\u7c7b\u76ee)[:\uff1a]\s*([^|]+)")[0]
        if cat.isna().all():
            parts = text.str.split("|", n=2, expand=True)
            cat = parts[1] if parts.shape[1] > 1 else pd.Series("Unknown", index=out.index)
    else:
        cat = pd.Series("Unknown", index=out.index)
    out[out_col] = cat.fillna("Unknown").astype(str).str.strip()
    return out[["item_id", out_col]].drop_duplicates("item_id")


def canonical_source_column(df: pd.DataFrame, base: str) -> str | None:
    aliases = {
        "atomic_a_pri": [
            "atomic_a_pri",
            "atomic_mkt_price",
            "atomic_a_pri_calibrated",
            "atomic_mkt_price_calibrated",
            "atomic_a_pri_calib",
            "atomic_mkt_price_calib",
        ],
    }
    candidates = aliases.get(base, [base, f"{base}_calibrated", f"{base}_calib"])
    for col in candidates:
        if col in df.columns:
            return col
    return None


def calibrate_canonical_mllm_treatments(df_t: pd.DataFrame, seed: int = 20260713) -> tuple[pd.DataFrame, list[str]]:
    out = df_t[["item_id"]].copy()
    calibrated_cols: list[str] = []
    for base in CANONICAL_MLLM_BASES:
        source = canonical_source_column(df_t, base)
        if source is None:
            continue
        target = f"{base}_calib"
        source_values = pd.to_numeric(df_t[source], errors="coerce").fillna(0.0)
        out[target] = zero_preserved_rank_norm(source_values, seed=seed)
        calibrated_cols.append(target)
    return out, calibrated_cols


def temporal_abc_split_without_time_overlap(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    ratios=(0.2, 0.4, 0.4),
):
    ordered = df.sort_values([time_col, "user_id", "item_id"]).reset_index(drop=True)
    n = len(ordered)
    if n == 0:
        return ordered, ordered.copy(), ordered.copy(), ordered.copy()
    a = min(max(int(np.floor(n * ratios[0])), 1), n - 1)
    b = min(max(int(np.floor(n * (ratios[0] + ratios[1]))), a + 1), n - 1)
    boundary = ordered.loc[a - 1, time_col]
    while a < n and ordered.loc[a, time_col] == boundary:
        a += 1
    b = max(b, a + 1)
    boundary = ordered.loc[b - 1, time_col]
    while b < n and ordered.loc[b, time_col] == boundary:
        b += 1
    return ordered, ordered.iloc[:a].copy(), ordered.iloc[a:b].copy(), ordered.iloc[b:].copy()


def assign_pre_history_h_levels(train_values: Iterable, all_values: Iterable) -> pd.Series:
    train = pd.Series(train_values).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    allv = pd.Series(all_values).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    levels = pd.Series(np.zeros(len(allv), dtype=int), index=allv.index)
    train_pos = train[train > 0]
    all_pos = allv > 0
    if len(train_pos) == 0 or all_pos.sum() == 0:
        return levels
    cuts = np.unique(np.quantile(train_pos, [1 / 3, 2 / 3]))
    if len(cuts) == 0:
        levels.loc[all_pos] = 1
    elif len(cuts) == 1:
        levels.loc[all_pos] = 1 + (allv.loc[all_pos] > cuts[0]).astype(int)
    else:
        levels.loc[all_pos] = 1 + np.searchsorted(cuts, allv.loc[all_pos], side="right")
    return levels.clip(0, 3).astype(int)


def treatment_exclusion_columns(columns: Sequence[str], treatments: Sequence[str]) -> set[str]:
    cols = set(columns)
    exclude = set(treatments)
    exclude.update(TREATMENT_DERIVED_ALIASES)
    for t in treatments:
        base = t.replace("_calib", "").replace("_calibrated", "")
        variants = {
            base,
            f"{base}_calib",
            f"{base}_calibrated",
            f"{base}_calibrated_calib",
            f"raw_{base}",
            f"raw_{base}_calib",
            f"raw_{base}_calibrated",
        }
        if base == "atomic_a_pri":
            variants.update(
                {
                    "atomic_mkt_price",
                    "atomic_mkt_price_calib",
                    "atomic_mkt_price_calibrated",
                    "atomic_mkt_price_calibrated_calib",
                }
            )
        exclude.update(v for v in variants if v in cols)
    return exclude


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    idx = np.where(finite)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, original_idx in enumerate(order):
        val = min((m - rank) * p[original_idx], 1.0)
        running = max(running, val)
        out[original_idx] = min(running, 1.0)
    return out


def add_holm_by_family(
    df: pd.DataFrame,
    family_cols: Sequence[str],
    p_col: str = "P_Value",
    out_col: str = "P_Value_Holm",
) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    for _, idx in out.groupby(list(family_cols), dropna=False).groups.items():
        idx_list = list(idx)
        out.loc[idx_list, out_col] = holm_adjust(out.loc[idx_list, p_col].to_numpy())
    out["Significant_Holm_05"] = out[out_col] < 0.05
    return out


def resolve_treatment_column(columns: Sequence[str], component: str) -> str | None:
    cols = set(columns)
    candidates = [
        component,
        f"{component}_calib",
        f"{component}_calibrated",
        f"atomic_{component}",
        f"atomic_{component}_calib",
        f"atomic_{component}_calibrated",
    ]
    aliases = {
        "a_pri": ["atomic_a_pri_calib", "atomic_mkt_price_calib", "atomic_a_pri", "atomic_mkt_price"],
        "a_gft": ["atomic_a_gft_calib", "atomic_a_gft"],
        "a_sub": ["atomic_a_sub_calib", "atomic_a_sub"],
        "a_urg": ["atomic_a_urg_calib", "atomic_a_urg"],
        "a_spec": ["atomic_a_spec_calib", "atomic_a_spec"],
        "a_str": ["atomic_a_str_calib", "atomic_a_str"],
        "T_con_mkt": ["T_con_mkt_calib", "T_con_mkt_calibrated"],
        "T_int_fac": ["T_int_fac_calib", "T_int_fac_calibrated"],
        "T_int_vis": ["T_int_vis_calib", "T_int_vis_calibrated"],
        "T_int_sem": ["T_int_sem"],
        "T_con_soc": ["T_con_soc"],
        "T_con_rat": ["T_con_rat"],
    }
    candidates.extend(aliases.get(component, []))
    for c in candidates:
        if c in cols:
            return c
    return None
