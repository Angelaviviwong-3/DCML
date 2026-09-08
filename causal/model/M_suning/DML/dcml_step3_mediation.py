#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Suning pre-exposure social-proof channel analysis, 2026-07-13.

This is an exploratory mechanism/channel analysis, not a strong causal
mediation design. The mediator is pre-exposure social proof, so it should be
reported as channel evidence or moved to an appendix in the paper.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths


import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


CAUSAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_utils import add_holm_by_family, ensure_dir, find_causal_root, read_table, setup_logging


ROOT = find_causal_root(__file__)
BASE_DIR = _dcml_paths.workspace_path(ROOT, 'processed_data') / "suning"
DATA_PATH = BASE_DIR / "build_dataset" / "DCML_C_20260712.parquet"
RES_PATH = BASE_DIR / "DML_Results" / "DCML_Residuals_20260717.parquet"
SCHEMA_PATH = BASE_DIR / "DML_Results" / "DML_feature_schema_20260717.json"
SAVE_DIR = ensure_dir(BASE_DIR / "Final_Causal_Output")
LOG_DIR = ensure_dir(_dcml_paths.workspace_path(ROOT, 'log'))
logger = setup_logging(LOG_DIR, "dcml_step3_mediation_suning_20260717")
Y_COLS = ["res_y_click", "res_y_cart", "res_y_purchase"]


def clean_component(col: str) -> str:
    return (
        col.replace("res_", "")
        .replace("_calibrated", "")
        .replace("_calib", "")
        .replace("atomic_", "")
    )


def residualize_mediator(df_raw: pd.DataFrame, x_cols: list[str]) -> pd.Series:
    x_cols = [c for c in x_cols if c in df_raw.columns]
    if not x_cols:
        return df_raw["M_norm"] - df_raw["M_norm"].mean()
    x = df_raw[x_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = sm.add_constant(x, has_constant="add")
    m = pd.to_numeric(df_raw["M_norm"], errors="coerce").fillna(0.0)
    model = sm.OLS(m, x).fit()
    return m - model.predict(x)


def bootstrap_indirect(df: pd.DataFrame, t_col: str, y_col: str, reps: int, seed: int) -> dict:
    work = df[[t_col, "M_tilde", y_col, "cluster_user_id"]].copy()
    for column in [t_col, "M_tilde", y_col]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 10:
        return {
            "Path_a_Coef": np.nan,
            "Path_b_Coef": np.nan,
            "Indirect_Effect": np.nan,
            "CI_Lower_95": np.nan,
            "CI_Upper_95": np.nan,
            "P_Value": np.nan,
            "Analysis_N": int(len(work)),
        }
    t = work[t_col].to_numpy()
    m = work["M_tilde"].to_numpy()
    y = work[y_col].to_numpy()

    xt = np.column_stack([np.ones(len(t)), t])
    alpha = float((np.linalg.pinv(xt.T @ xt) @ xt.T @ m)[1])
    xym = np.column_stack([np.ones(len(t)), t, m])
    beta = float((np.linalg.pinv(xym.T @ xym) @ xym.T @ y)[2])
    point = alpha * beta

    rng = np.random.default_rng(seed)
    groups = work.groupby("cluster_user_id").indices
    draws = []
    if groups and len(groups) > 1:
        keys = np.array(list(groups.keys()))
        for _ in range(reps):
            sampled = rng.choice(keys, size=len(keys), replace=True)
            idx = np.concatenate([groups[g] for g in sampled])
            tb, mb, yb = t[idx], m[idx], y[idx]
            try:
                xtb = np.column_stack([np.ones(len(tb)), tb])
                ab = np.linalg.pinv(xtb.T @ xtb) @ xtb.T @ mb
                xmb = np.column_stack([np.ones(len(tb)), tb, mb])
                bb = np.linalg.pinv(xmb.T @ xmb) @ xmb.T @ yb
                draws.append(float(ab[1] * bb[2]))
            except Exception:
                continue
    else:
        n = len(df)
        for _ in range(reps):
            idx = rng.integers(0, n, size=n)
            tb, mb, yb = t[idx], m[idx], y[idx]
            xtb = np.column_stack([np.ones(len(tb)), tb])
            ab = np.linalg.pinv(xtb.T @ xtb) @ xtb.T @ mb
            xmb = np.column_stack([np.ones(len(tb)), tb, mb])
            bb = np.linalg.pinv(xmb.T @ xmb) @ xmb.T @ yb
            draws.append(float(ab[1] * bb[2]))

    arr = np.asarray(draws, dtype=float)
    if len(arr) == 0:
        ci_l = ci_u = p = np.nan
    else:
        ci_l, ci_u = np.percentile(arr, [2.5, 97.5])
        p = min(float(np.mean(arr >= 0)), float(np.mean(arr <= 0))) * 2
    return {
        "Path_a_Coef": alpha,
        "Path_b_Coef": beta,
        "Indirect_Effect": point,
        "CI_Lower_95": ci_l,
        "CI_Upper_95": ci_u,
        "P_Value": min(p, 1.0) if np.isfinite(p) else np.nan,
        "Analysis_N": int(len(work)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    args = parser.parse_args()

    start = time.time()
    logger.info("=" * 80)
    logger.info("Suning pre-exposure social-proof channel analysis, 20260717")
    df_raw = read_table(DATA_PATH)
    df_res = read_table(RES_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    x_cols = schema.get("confounder_whitelist", schema.get("confounders", []))

    df = df_res.merge(
        df_raw[["user_id", "item_id", "timestamp", "M_norm"] + [c for c in x_cols if c in df_raw.columns]],
        on=["user_id", "item_id", "timestamp", "M_norm"],
        how="inner",
    )
    df["M_tilde"] = residualize_mediator(df, x_cols)

    treatment_cols = [
        c
        for c in df.columns
        if c.startswith("res_")
        and not c.startswith("res_y_")
        and c != "res_T_con_soc"
        and c.replace("res_", "raw_") in df.columns
    ]

    rows = []
    for t_col in treatment_cols:
        for y_col in Y_COLS:
            res = bootstrap_indirect(df, t_col, y_col, args.bootstrap_reps, seed=20260717 + len(rows))
            rows.append(
                {
                    "Mechanism": "Pre-exposure Social-Proof Channel",
                    "Heuristic_Cue": clean_component(t_col),
                    "Treatment_Column": t_col,
                    "Funnel_Stage": y_col.replace("res_y_", ""),
                    **res,
                    "Significant_CI_95": bool(res["CI_Lower_95"] * res["CI_Upper_95"] > 0) if np.isfinite(res["CI_Lower_95"]) else False,
                    "N": int(res["Analysis_N"]),
                    "Bootstrap": "cluster_by_user",
                    "Interpretation_Level": "exploratory_channel_evidence_not_strong_causal_mediation",
                    "Temporal_Analysis_Population": schema.get(
                        "analysis_population",
                        "items with all selected MCRE reviews strictly before item-specific Set C entry",
                    ),
                    "Note": "T_con_soc is excluded because it is the normalized source of the pre-exposure social-proof channel.",
                }
            )

    out = pd.DataFrame(rows)
    out = add_holm_by_family(out, ["Funnel_Stage"], p_col="P_Value", out_col="P_Value_Holm")
    out.to_csv(SAVE_DIR / "Final_Mediation_Results_20260717.csv", index=False)
    logger.info("Saved channel results: %s", SAVE_DIR / "Final_Mediation_Results_20260717.csv")
    logger.info("Completed in %.2f seconds", time.time() - start)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
