#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Amazon purchase-only social-proof channel analysis, 2026-07-15.

Amazon Reviews'23 has no native click/cart logs. This script therefore reports
bottom-funnel purchase-stage channel evidence only, using the 20260717 DCML
tables and residuals.
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
from temporal_certification import filter_certified_frame, load_certification


ROOT = find_causal_root(__file__)
LOG_DIR = ensure_dir(_dcml_paths.workspace_path(ROOT, 'log'))
Y_COLS = ["res_y_purchase"]
CLAIM_BOUNDARY = (
    "Amazon purchase-stage sampled-choice external validation among temporally certified items; not population "
    "checkout conversion, full-funnel reversal, or policy/OPE evidence."
)


def clean_component(col: str) -> str:
    return (
        col.replace("res_", "")
        .replace("_calibrated", "")
        .replace("_calib", "")
        .replace("atomic_", "")
    )


def residualize_mediator(df: pd.DataFrame, x_cols: list[str]) -> pd.Series:
    x_cols = [c for c in x_cols if c in df.columns]
    m = pd.to_numeric(df["M_norm"], errors="coerce").fillna(0.0)
    if not x_cols:
        return m - m.mean()
    x = df[x_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = sm.add_constant(x, has_constant="add")
    model = sm.OLS(m, x).fit()
    return m - model.predict(x)


def _keys_aligned(left: pd.DataFrame, right: pd.DataFrame, key_cols: list[str]) -> bool:
    for col in key_cols:
        if col == "timestamp":
            left_values = pd.to_numeric(left[col], errors="coerce").to_numpy(dtype=float)
            right_values = pd.to_numeric(right[col], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(left_values, right_values, equal_nan=True):
                return False
        else:
            if not left[col].astype(str).reset_index(drop=True).equals(right[col].astype(str).reset_index(drop=True)):
                return False
    return True


def attach_pre_event_x(
    df_raw: pd.DataFrame,
    df_res: pd.DataFrame,
    x_cols: list[str],
    dataset: str,
    logger,
) -> pd.DataFrame:
    merge_cols = ["user_id", "item_id", "timestamp"]
    certified_items, _, _ = load_certification(ROOT, dataset)
    raw_cert = filter_certified_frame(df_raw, certified_items).reset_index(drop=True)
    res = df_res.reset_index(drop=True).copy()
    available_x = [c for c in x_cols if c in raw_cert.columns]
    raw = raw_cert[merge_cols + available_x].copy()
    for frame in [raw, res]:
        frame["user_id"] = frame["user_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")

    if len(raw) == len(res) and _keys_aligned(res, raw, merge_cols):
        for col in available_x:
            res[col] = raw[col].to_numpy()
        res["Channel_Merge_Method"] = "certified_set_c_row_order"
        logger.info("Aligned certified Set C and residuals by row order: %d rows", len(res))
        return res

    def add_occurrence(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["_key_occurrence"] = out.groupby(merge_cols, sort=False).cumcount()
        return out

    res_keyed = add_occurrence(res)
    res_keyed["_res_row"] = np.arange(len(res_keyed))
    raw_keyed = add_occurrence(raw)
    merged = res_keyed.merge(
        raw_keyed,
        on=merge_cols + ["_key_occurrence"],
        how="inner",
        validate="one_to_one",
    ).sort_values("_res_row")
    if len(merged) != len(res):
        raise RuntimeError(
            f"Certified Set C/residual alignment changed row count: residual={len(res)} merged={len(merged)}. "
            "Refuse to run channel analysis on incomplete or mismatched files."
        )
    merged = merged.drop(columns=["_key_occurrence", "_res_row"])
    merged["Channel_Merge_Method"] = "certified_set_c_key_occurrence"
    logger.info("Aligned certified Set C and residuals by duplicate-aware key occurrence: %d rows", len(merged))
    return merged


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
    draws: list[float] = []
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


def run(bootstrap_reps: int) -> None:
    dataset = "amazon_appliances"
    logger = setup_logging(LOG_DIR, f"dcml_step3_mediation_{dataset}_purchase_20260717")
    start = time.time()
    base_dir = _dcml_paths.workspace_path(ROOT, 'processed_data') / dataset
    data_path = base_dir / "build_dataset" / f"{dataset}_DCML_Set_C_20260717.parquet"
    res_path = base_dir / "DML_Results" / f"{dataset}_DML_Residuals_Final_20260717.parquet"
    schema_path = base_dir / "DML_Results" / f"{dataset}_DML_feature_schema_20260717.json"
    save_dir = ensure_dir(base_dir / "Final_Causal_Output")

    logger.info("=" * 80)
    logger.info("Amazon purchase-only social-proof channel analysis, dataset=%s", dataset)
    df_raw = read_table(data_path)
    df_res = read_table(res_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    x_cols = schema.get("confounder_whitelist", schema.get("confounders", []))

    df = attach_pre_event_x(df_raw, df_res, x_cols, dataset, logger)
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
            if y_col not in df.columns:
                continue
            res = bootstrap_indirect(df, t_col, y_col, bootstrap_reps, seed=20260717 + len(rows))
            rows.append(
                {
                    "Mechanism": "Pre-exposure Social-Proof Channel",
                    "Heuristic_Cue": clean_component(t_col),
                    "Treatment_Column": t_col,
                    "Funnel_Stage": "purchase",
                    **res,
                    "Significant_CI_95": bool(res["CI_Lower_95"] * res["CI_Upper_95"] > 0) if np.isfinite(res["CI_Lower_95"]) else False,
                    "N": int(res["Analysis_N"]),
                    "Bootstrap": "cluster_by_user",
                    "Interpretation_Level": "bottom_funnel_channel_evidence_not_full_funnel_mediation",
                    "Claim_Boundary": CLAIM_BOUNDARY,
                    "Temporal_Analysis_Population": schema.get("analysis_population"),
                    "Note": "T_con_soc excluded because it is the normalized source of the pre-exposure social-proof channel.",
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_holm_by_family(out, ["Funnel_Stage"], p_col="P_Value", out_col="P_Value_Holm")
    out_path = save_dir / f"{dataset}_Final_Mediation_Results_20260717.csv"
    out.to_csv(out_path, index=False)
    logger.info("Saved channel results: %s", out_path)
    logger.info("Completed in %.2f seconds", time.time() - start)
    logger.info("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    args = parser.parse_args()
    run(args.bootstrap_reps)


if __name__ == "__main__":
    main()
