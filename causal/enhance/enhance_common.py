#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the 20260730 enhancement experiments."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


VERSION = "20260730"
ENHANCE_DIR = Path(__file__).resolve().parent
CAUSAL_ROOT = ENHANCE_DIR.parent
PROJECT_ROOT = CAUSAL_ROOT.parent
OUTPUT_DIR = _dcml_paths.project_path("enhance/output")
DATASETS = ("suning", "amazon_appliances", "amazon_beauty")


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                f"Cannot read {path}. Install pyarrow or fastparquet in the experiment "
                "environment, or provide the corresponding CSV file."
            ) from exc
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def first_existing(candidates: Iterable[Path | str]) -> Path | None:
    for candidate in candidates:
        path = Path(candidate)
        if any(ch in str(path) for ch in ["*", "?", "["]):
            matches = sorted(path.parent.glob(path.name))
            if matches:
                return matches[-1]
        elif path.exists():
            return path
    return None


def clean_component_name(column: str) -> str:
    name = str(column)
    for token in [
        "__reverse_gs_pure",
        "__gs_pure",
        "__gs_pure_20260730",
        "_calibrated",
        "_calib",
    ]:
        name = name.replace(token, "")
    name = name.replace("res_", "").replace("atomic_", "")
    if name == "mkt_price":
        return "a_pri"
    if name in {"T_int_sem", "T_align"}:
        return "T_align"
    return name


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm adjustment without requiring statsmodels."""
    p = pd.to_numeric(p_values, errors="coerce")
    adjusted = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    m = len(valid)
    if m == 0:
        return adjusted
    ordered = valid.sort_values()
    running = 0.0
    for rank, (idx, value) in enumerate(ordered.items(), start=1):
        candidate = min(1.0, (m - rank + 1) * float(value))
        running = max(running, candidate)
        adjusted.loc[idx] = running
    return adjusted


def add_holm_by_group(frame: pd.DataFrame, group_cols: list[str], p_col: str = "P_Value") -> pd.DataFrame:
    out = frame.copy()
    out["P_Value_Holm"] = np.nan
    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        index = list(idx)
        out.loc[index, "P_Value_Holm"] = holm_adjust(out.loc[index, p_col])
    out["Significant_Holm_05"] = out["P_Value_Holm"] < 0.05
    return out


def spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 3:
        return np.nan
    if pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return np.nan
    return float(pair["left"].rank(method="average").corr(pair["right"].rank(method="average")))


def pearson(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 3:
        return np.nan
    if pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return np.nan
    return float(pair["left"].corr(pair["right"]))


def quadratic_weighted_kappa(left: pd.Series, right: pd.Series, levels: int = 11) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if pair.empty:
        return np.nan
    lval = np.clip(np.rint(pair["left"].to_numpy(dtype=float)).astype(int), 0, levels - 1)
    rval = np.clip(np.rint(pair["right"].to_numpy(dtype=float)).astype(int), 0, levels - 1)
    observed = np.zeros((levels, levels), dtype=float)
    np.add.at(observed, (lval, rval), 1.0)
    expected = np.outer(np.bincount(lval, minlength=levels), np.bincount(rval, minlength=levels))
    expected = expected / max(1, len(pair))
    grid = np.arange(levels, dtype=float)
    weights = np.square(grid[:, None] - grid[None, :]) / np.square(levels - 1)
    denominator = float(np.sum(weights * expected))
    return float(1.0 - np.sum(weights * observed) / denominator) if denominator > 0 else np.nan


def tertile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=object)
    finite = numeric.dropna()
    if finite.nunique() < 3:
        return out
    ranks = finite.rank(method="average", pct=True)
    out.loc[finite.index] = pd.cut(
        ranks,
        bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        labels=["low", "mid", "high"],
        include_lowest=True,
    ).astype(str)
    return out
