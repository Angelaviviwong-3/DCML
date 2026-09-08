#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained Suning strict Click -> Cart -> Purchase funnel builder, 20260712."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


import sys
import time
from pathlib import Path

import pandas as pd


CAUSAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_utils import ensure_dir, find_causal_root, setup_logging, write_table


ROOT = find_causal_root(__file__)
RAW_DIR = _dcml_paths.workspace_path(ROOT, 'raw_data') / "suning_raw"
OUTPUT_DIR = ensure_dir(_dcml_paths.workspace_path(ROOT, 'processed_data') / "suning" / "Y")
LOG_DIR = ensure_dir(_dcml_paths.workspace_path(ROOT, 'log'))
logger = setup_logging(LOG_DIR, "process_suning_y_strict_20260712")

CLICK_PATH = RAW_DIR / "1_suning_y_click.xlsx"
CART_PURCHASE_PATHS = [
    RAW_DIR / "2_suning_y_cart&y_purchase_M4.xlsx",
    RAW_DIR / "2_suning_y_cart&y_purchase_M5.xlsx",
    RAW_DIR / "2_suning_y_cart&y_purchase_M6.xlsx",
]


def enforce_strict_funnel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["y_click", "y_cart", "y_purchase"]:
        out[col] = out[col].fillna(0).astype(int)
    out.loc[out["y_purchase"] == 1, ["y_cart", "y_click"]] = 1
    out.loc[out["y_cart"] == 1, "y_click"] = 1
    return out


def collapse_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["user_id", "item_id", "timestamp"], as_index=False)
        .agg(
            {
                "y_click": "max",
                "y_cart": "max",
                "y_purchase": "max",
                "position": lambda s: next((x for x in s if pd.notna(x) and x != "trans_direct"), "trans_direct"),
            }
        )
        .sort_values(["user_id", "timestamp", "item_id"])
        .reset_index(drop=True)
    )


def diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    patterns = df[["y_click", "y_cart", "y_purchase"]].value_counts().reset_index(name="n")
    rows = [
        {"metric": "purchase_without_cart", "value": int(((df["y_purchase"] == 1) & (df["y_cart"] == 0)).sum())},
        {"metric": "cart_without_click", "value": int(((df["y_cart"] == 1) & (df["y_click"] == 0)).sum())},
        {"metric": "purchase_without_click", "value": int(((df["y_purchase"] == 1) & (df["y_click"] == 0)).sum())},
        {"metric": "n_rows", "value": int(len(df))},
        {"metric": "n_users", "value": int(df["user_id"].nunique())},
        {"metric": "n_items", "value": int(df["item_id"].nunique())},
        {"metric": "n_click", "value": int(df["y_click"].sum())},
        {"metric": "n_cart", "value": int(df["y_cart"].sum())},
        {"metric": "n_purchase", "value": int(df["y_purchase"].sum())},
    ]
    rows.extend(
        {
            "metric": f"pattern_click{int(r.y_click)}_cart{int(r.y_cart)}_purchase{int(r.y_purchase)}",
            "value": int(r.n),
        }
        for r in patterns.itertuples(index=False)
    )
    return pd.DataFrame(rows)


def main() -> None:
    start = time.time()
    logger.info("=" * 80)
    logger.info("Building strict Suning funnel, 20260712")
    logger.info("Root: %s", ROOT)

    click = pd.read_excel(CLICK_PATH, engine="openpyxl").rename(columns={"module_ID": "module_id", "slot_ID": "slot_id"})
    click["y_click"] = (click["click_quantity"].fillna(0) > 0).astype(int)
    click["position"] = click["module_id"].astype(str).fillna("") + "_" + click["slot_id"].astype(str).fillna("")
    click = click[["user_id", "item_id", "timestamp", "y_click", "position"]]

    cp_frames = [pd.read_excel(path, engine="openpyxl") for path in CART_PURCHASE_PATHS]
    cp = pd.concat(cp_frames, ignore_index=True)
    for col in ["y_cart", "y_purchase", "addcart_quantity", "purchase_quantity"]:
        if col not in cp.columns:
            cp[col] = 0
        cp[col] = cp[col].fillna(0)
    cp["y_cart"] = ((cp["y_cart"] > 0) | (cp["addcart_quantity"] > 0)).astype(int)
    cp["y_purchase"] = ((cp["y_purchase"] > 0) | (cp["purchase_quantity"] > 0)).astype(int)
    cp = cp[["user_id", "item_id", "timestamp", "y_cart", "y_purchase"]]

    y = pd.merge(click, cp, on=["user_id", "item_id", "timestamp"], how="outer")
    y["position"] = y["position"].fillna("trans_direct")
    y["user_id"] = y["user_id"].astype(str).str.strip()
    y["item_id"] = y["item_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    y["timestamp"] = pd.to_numeric(y["timestamp"], errors="coerce")
    y = y.dropna(subset=["user_id", "item_id", "timestamp"])
    y = enforce_strict_funnel(y)
    y = collapse_duplicates(y)
    y = enforce_strict_funnel(y)

    diag = diagnostics(y)
    diag.to_csv(OUTPUT_DIR / "y_behavior_strict_diagnostics_20260712.csv", index=False)
    if int(diag.loc[diag["metric"] == "purchase_without_cart", "value"].iloc[0]) != 0:
        raise RuntimeError("purchase_without_cart remains after strict funnel enforcement.")
    if int(diag.loc[diag["metric"] == "cart_without_click", "value"].iloc[0]) != 0:
        raise RuntimeError("cart_without_click remains after strict funnel enforcement.")

    write_table(y, OUTPUT_DIR / "y_behavior_20260712.parquet", logger)
    logger.info("Completed in %.2f seconds", time.time() - start)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
