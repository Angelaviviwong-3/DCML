#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Permutation placebo test on the certified 20260717 Suning DML sample."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths


import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


CAUSAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_inference import clean_component_name, prepare_orthogonalized_data
from dcml_utils import ensure_dir, find_causal_root, setup_logging


VERSION = "20260717"
ROOT = find_causal_root(__file__)
BASE_DIR = _dcml_paths.workspace_path(ROOT, 'processed_data') / "suning"
ATE_PATH = BASE_DIR / "Final_Causal_Output" / f"Final_ATE_Results_{VERSION}.csv"
SAVE_DIR = ensure_dir(BASE_DIR / "Final_Causal_Output")
logger = setup_logging(ensure_dir(_dcml_paths.workspace_path(ROOT, 'log')), f"placebo_test_suning_{VERSION}")
Y_COLS = ["res_y_click", "res_y_cart", "res_y_purchase"]


def fast_coefficients(df: pd.DataFrame, y_col: str, treatment_cols: list[str]) -> dict[str, float]:
    frame = df[[y_col, *treatment_cols]].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) <= len(treatment_cols) + 5:
        return {column: np.nan for column in treatment_cols}
    x = np.column_stack([np.ones(len(frame)), frame[treatment_cols].to_numpy(dtype=float)])
    y = frame[y_col].to_numpy(dtype=float)
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    return {column: float(beta[index + 1]) for index, column in enumerate(treatment_cols)}


def collect_coefficients(
    df: pd.DataFrame,
    specifications: dict[str, list[str]],
) -> dict[tuple[str, str, str], float]:
    output: dict[tuple[str, str, str], float] = {}
    for resolution, treatment_cols in specifications.items():
        for y_col in Y_COLS:
            stage = y_col.replace("res_y_", "")
            for treatment, coefficient in fast_coefficients(df, y_col, treatment_cols).items():
                output[(resolution, stage, clean_component_name(treatment))] = coefficient
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutation-reps", type=int, default=200)
    parser.add_argument("--gs-mode", choices=["linear", "linear_quadratic"], default="linear")
    args = parser.parse_args()
    if args.permutation_reps <= 0:
        parser.error("--permutation-reps must be positive")

    start = time.time()
    logger.info("Suning certified-sample permutation placebo, reps=%d", args.permutation_reps)
    final, specifications, _, manifest = prepare_orthogonalized_data(ROOT, "suning", args.gs_mode)
    true_ate = pd.read_csv(ATE_PATH)
    treatment_cols = list(dict.fromkeys(column for columns in specifications.values() for column in columns))
    base_treatments = final[treatment_cols].to_numpy(dtype=float)
    rng = np.random.default_rng(20260717)
    placebo_rows: list[dict] = []

    for rep in range(args.permutation_reps):
        shuffled = final.copy()
        shuffled[treatment_cols] = base_treatments[rng.permutation(len(final)), :]
        coefficients = collect_coefficients(shuffled, specifications)
        for (resolution, stage, component), coefficient in coefficients.items():
            placebo_rows.append(
                {
                    "Resolution": resolution,
                    "Stage": stage,
                    "Component": component,
                    "Permutation_Rep": rep + 1,
                    "Coef_Placebo": coefficient,
                    "Temporal_Analysis_Population": manifest["analysis_population"],
                }
            )
        if (rep + 1) % 20 == 0 or rep + 1 == args.permutation_reps:
            logger.info("Completed placebo permutation %d/%d", rep + 1, args.permutation_reps)

    placebo = pd.DataFrame(placebo_rows)
    aggregate = (
        placebo.groupby(["Resolution", "Stage", "Component"], as_index=False)["Coef_Placebo"]
        .agg(
            Placebo_Mean="mean",
            Placebo_SD="std",
            Placebo_CI_Lower_95=lambda values: float(np.nanpercentile(values, 2.5)),
            Placebo_CI_Upper_95=lambda values: float(np.nanpercentile(values, 97.5)),
        )
    )
    true_columns = [
        "Resolution",
        "Stage",
        "Component",
        "Treatment_Column",
        "Coefficient",
        "Std_Error",
        "P_Value",
        "P_Value_Holm",
        "Significant_Holm_05",
    ]
    output = true_ate[[column for column in true_columns if column in true_ate.columns]].merge(
        aggregate,
        on=["Resolution", "Stage", "Component"],
        how="left",
    )
    empirical = []
    for row in output.itertuples(index=False):
        null = placebo.loc[
            placebo["Resolution"].eq(row.Resolution)
            & placebo["Stage"].eq(row.Stage)
            & placebo["Component"].eq(row.Component),
            "Coef_Placebo",
        ].dropna().to_numpy(dtype=float)
        empirical.append(
            float((1 + np.sum(np.abs(null) >= abs(row.Coefficient))) / (len(null) + 1))
            if len(null) and np.isfinite(row.Coefficient)
            else np.nan
        )
    output["Placebo_Empirical_P_Value"] = empirical
    output["Placebo_Reps"] = args.permutation_reps
    output["Temporal_Analysis_Population"] = manifest["analysis_population"]
    output["Placebo_Note"] = "Certified-sample orthogonalized treatment rows jointly permuted."
    output.to_csv(SAVE_DIR / f"RQ5_Placebo_Test_Results_Detailed_{VERSION}.csv", index=False)
    placebo.to_csv(SAVE_DIR / f"RQ5_Placebo_Test_Permutations_{VERSION}.csv", index=False)
    logger.info("Completed certified placebo analysis in %.2f seconds", time.time() - start)


if __name__ == "__main__":
    main()
