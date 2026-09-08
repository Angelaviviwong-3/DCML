#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Suning exploratory offline policy simulation using 20260713 estimates."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CAUSAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_utils import ensure_dir, find_causal_root, read_table, resolve_treatment_column, setup_logging


try:
    import joblib
except Exception:
    joblib = None


ROOT = find_causal_root(__file__)
SAVE_DIR = ensure_dir(_dcml_paths.workspace_path(ROOT, "Unified_Visualization/output") / "Unified_Visualization_RQ5_20260713")
LOG_DIR = ensure_dir(_dcml_paths.workspace_path(ROOT, 'log'))
logger = setup_logging(LOG_DIR, "OPE_Policy_20260713")

DATA_PATH = _dcml_paths.workspace_path(ROOT, 'processed_data') / "suning" / "build_dataset" / "DCML_C_20260712.parquet"
MODEL_DIR = _dcml_paths.workspace_path(ROOT, 'processed_data') / "suning" / "DML_Results" / "nuisance_models_20260712"
FEATURE_SCHEMA_PATH = _dcml_paths.workspace_path(ROOT, 'processed_data') / "suning" / "DML_Results" / "DML_feature_schema_20260712.json"
ATE_PATH = _dcml_paths.workspace_path(ROOT, 'processed_data') / "suning" / "Final_Causal_Output" / "Final_ATE_Results_20260713.csv"
CATE_PATH = _dcml_paths.workspace_path(ROOT, 'processed_data') / "suning" / "Final_CATE_Results_20260713.csv"
FUNNEL_WEIGHTS = {"click": 0.1, "cart": 0.2, "purchase": 0.7}


def coefficient_map(df: pd.DataFrame, stage: str, h_level=None) -> dict[str, tuple[float, float]]:
    sub = df[df["Stage"] == stage].copy()
    if h_level is not None and "H_level" in sub.columns:
        sub = sub[sub["H_level"].astype(str) == str(h_level)]
    if "Estimable" in sub.columns:
        sub = sub[sub["Estimable"].fillna(False).astype(bool)]
    sub = sub[np.isfinite(pd.to_numeric(sub["Coefficient"], errors="coerce"))].copy()
    if sub.empty:
        return {}
    sort_cols = ["Component"]
    if "P_Value_Holm" in sub.columns:
        sort_cols.append("P_Value_Holm")
    sort_cols.append("Std_Error")
    sub = sub.sort_values(sort_cols).drop_duplicates("Component", keep="first")
    return {
        str(r.Component): (float(r.Coefficient), float(r.Std_Error))
        for r in sub.itertuples(index=False)
        if np.isfinite(r.Coefficient)
    }


def causal_lift(df: pd.DataFrame, theta: dict[str, tuple[float, float]]) -> np.ndarray:
    lift = np.zeros(len(df), dtype=float)
    for component, (coef, _) in theta.items():
        col = resolve_treatment_column(df.columns, component)
        if col is not None:
            lift += pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy() * coef
    return lift


def causal_lift_ci(df: pd.DataFrame, theta: dict[str, tuple[float, float]], reps: int = 1000, seed: int = 20260713):
    rng = np.random.default_rng(seed)
    base = causal_lift(df, theta)
    draws = []
    for _ in range(reps):
        sampled = {
            comp: (rng.normal(coef, se if np.isfinite(se) and se > 0 else 0.0), se)
            for comp, (coef, se) in theta.items()
        }
        draws.append(float(np.mean(causal_lift(df, sampled))))
    return float(np.mean(base)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def load_purchase_prediction(df: pd.DataFrame) -> np.ndarray:
    if joblib is None:
        return np.repeat(df["y_purchase"].mean(), len(df))
    model_path = MODEL_DIR / "model_y_purchase.pkl"
    if not model_path.exists() or not FEATURE_SCHEMA_PATH.exists():
        logger.warning("Missing purchase model/schema; using observed mean fallback.")
        return np.repeat(df["y_purchase"].mean(), len(df))
    with open(FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    x_cols = [c for c in schema.get("confounders", []) if c in df.columns]
    if not x_cols:
        return np.repeat(df["y_purchase"].mean(), len(df))
    x = df[x_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model = joblib.load(model_path)
    if hasattr(model, "predict_proba"):
        return np.clip(model.predict_proba(x)[:, 1], 0, 1)
    return np.clip(model.predict(x), 0, 1)


def select_top1(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return df.sort_values(["user_id", score_col], ascending=[True, False]).groupby("user_id", as_index=False).first()


def apply_debias_marketing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mkt = resolve_treatment_column(out.columns, "T_con_mkt")
    fac = resolve_treatment_column(out.columns, "T_int_fac")
    if mkt is not None:
        vals = pd.to_numeric(out[mkt], errors="coerce").fillna(0.0)
        mask = vals > 0.7
        out.loc[mask, mkt] = vals.loc[mask] * 0.5
    if fac is not None:
        out[fac] = np.clip(pd.to_numeric(out[fac], errors="coerce").fillna(0.0) + 0.1, 0, 1)
    return out


def summarize(selected: pd.DataFrame, candidates: pd.DataFrame, policy: str, theta: dict[str, tuple[float, float]], intervention="None"):
    lift_mean, lift_low, lift_high = causal_lift_ci(selected, theta)
    return {
        "Dataset": "Suning_Main_Real_Funnel",
        "Policy": policy,
        "Intervention": intervention,
        "N_Users": int(selected["user_id"].nunique()),
        "N_Items": int(selected["item_id"].nunique()),
        "Observed_Click_Rate_%": float(selected["y_click"].mean() * 100),
        "Observed_Cart_Rate_%": float(selected["y_cart"].mean() * 100),
        "Observed_Purchase_Rate_%": float(selected["y_purchase"].mean() * 100),
        "Mean_Predicted_Purchase_%": float(selected["pred_purchase"].mean() * 100),
        "Mean_Causal_Lift": lift_mean,
        "Mean_Causal_Lift_CI_Lower_95": lift_low,
        "Mean_Causal_Lift_CI_Upper_95": lift_high,
        "Candidate_N": int(len(candidates)),
        "Simulation_Note": "Exploratory offline ranking simulation; not IPS/DR OPE because logged propensities are unavailable.",
    }


def main() -> None:
    logger.info("=" * 80)
    logger.info("Suning OPE, 20260713")
    df = read_table(DATA_PATH)
    ate = pd.read_csv(ATE_PATH)
    cate = pd.read_csv(CATE_PATH)

    eligible_users = df["user_id"].value_counts()
    df = df[df["user_id"].isin(eligible_users[eligible_users >= 3].index)].copy()
    if df.empty:
        raise RuntimeError("No users with at least three candidate rows for OPE.")
    df["pred_purchase"] = load_purchase_prediction(df)

    theta_purchase = coefficient_map(ate, "purchase")
    theta_by_stage = {stage: coefficient_map(ate, stage) for stage in ["click", "cart", "purchase"]}
    rows = []

    static = df.assign(score_static=df["pred_purchase"] + causal_lift(df, theta_purchase))
    rows.append(summarize(select_top1(static, "score_static"), df, "DCML Causal-Aware Static", theta_purchase))

    dynamic_score = np.zeros(len(df), dtype=float)
    for stage, weight in FUNNEL_WEIGHTS.items():
        dynamic_score += weight * causal_lift(df, theta_by_stage.get(stage, {}))
    dynamic = df.assign(score_dynamic=df["pred_purchase"] + dynamic_score)
    rows.append(summarize(select_top1(dynamic, "score_dynamic"), df, "DCML Funnel-Dynamic", theta_purchase))

    cate_scores = np.zeros(len(df), dtype=float)
    for h in sorted(pd.unique(df["H_level"])):
        mask = df["H_level"].astype(str) == str(h)
        theta_h = coefficient_map(cate, "purchase", h_level=h) or theta_purchase
        cate_scores[mask] = causal_lift(df.loc[mask], theta_h)
    cate_df = df.assign(score_cate=df["pred_purchase"] + cate_scores)
    rows.append(summarize(select_top1(cate_df, "score_cate"), df, "DCML CATE-Personalized", theta_purchase))

    rows.append(summarize(select_top1(df, "pred_purchase"), df, "Naive CVR-Max", theta_purchase))

    interv = apply_debias_marketing(df)
    interv["pred_purchase"] = df["pred_purchase"].to_numpy()
    interv = interv.assign(score_intervention=interv["pred_purchase"] + causal_lift(interv, theta_purchase))
    rows.append(summarize(select_top1(interv, "score_intervention"), df, "DCML Atomic-Intervention", theta_purchase, "debias_marketing"))

    out = pd.DataFrame(rows)
    baseline = out.loc[out["Policy"] == "DCML Causal-Aware Static", "Observed_Purchase_Rate_%"]
    if len(baseline):
        out["Delta_Observed_Purchase_vs_Static_pp"] = out["Observed_Purchase_Rate_%"] - float(baseline.iloc[0])
    out.to_csv(SAVE_DIR / "RQ5_OPE_Suning_20260713.csv", index=False)
    logger.info("Saved OPE results: %s", SAVE_DIR / "RQ5_OPE_Suning_20260713.csv")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
