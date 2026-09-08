#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Certified-sample Amazon purchase-stage placebo diagnostics."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[0]))
import project_paths as _dcml_paths


import time
from pathlib import Path

import numpy as np
import pandas as pd

from dcml_inference import clean_component_name, inference_contract, prepare_orthogonalized_data
from dcml_utils import ensure_dir, setup_logging


VERSION = "20260717"


def _permutation_draws(y: np.ndarray, x: np.ndarray, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, x.shape[1]), dtype=float)
    for rep in range(reps):
        shuffled = x[rng.permutation(len(x))]
        design = np.column_stack([np.ones(len(shuffled)), shuffled])
        draws[rep] = (np.linalg.pinv(design.T @ design) @ design.T @ y)[1:]
    return draws


def run_amazon_placebo(root: Path, dataset: str, permutations: int, gs_mode: str) -> None:
    if dataset not in {"amazon_appliances", "amazon_beauty"}:
        raise ValueError(dataset)
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    logger = setup_logging(ensure_dir(_dcml_paths.workspace_path(root, 'log')), f"placebo_test_{dataset}_purchase_{VERSION}")
    start = time.time()
    base = _dcml_paths.workspace_path(root, 'processed_data') / dataset
    save_dir = ensure_dir(base / "Final_Causal_Output")
    contract = inference_contract(root, dataset)
    final, specifications, _, manifest = prepare_orthogonalized_data(root, dataset, gs_mode)
    true_ate = pd.read_csv(save_dir / contract["ate"])
    true_ate = true_ate[
        true_ate["H_level"].astype(str).eq("All (ATE)") & true_ate["Stage"].eq("purchase")
    ].copy()
    summary_rows: list[dict] = []
    draw_rows: list[dict] = []

    for offset, (resolution, treatments) in enumerate(specifications.items()):
        frame = final[["res_y_purchase", *treatments]].apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) <= len(treatments) + 5:
            raise RuntimeError(f"{dataset} {resolution}: insufficient complete cases for placebo")
        y = frame["res_y_purchase"].to_numpy(dtype=float)
        x = frame[treatments].to_numpy(dtype=float)
        draws = _permutation_draws(y, x, permutations, 20260717 + offset)
        true_by_component = {
            row.Component: row.Coefficient
            for row in true_ate[true_ate["Resolution"].eq(resolution)].itertuples(index=False)
        }
        for treatment_index, treatment in enumerate(treatments):
            component = clean_component_name(treatment)
            values = draws[:, treatment_index]
            true_coefficient = float(true_by_component.get(component, np.nan))
            summary_rows.append(
                {
                    "Resolution": resolution,
                    "Stage": "purchase",
                    "Component": component,
                    "Coefficient": true_coefficient,
                    "Placebo_Mean": float(np.mean(values)),
                    "Placebo_SD": float(np.std(values, ddof=1)),
                    "Placebo_CI_Lower_95": float(np.percentile(values, 2.5)),
                    "Placebo_CI_Upper_95": float(np.percentile(values, 97.5)),
                    "Placebo_Empirical_P_Value": (
                        float((1 + np.sum(np.abs(values) >= abs(true_coefficient))) / (len(values) + 1))
                        if np.isfinite(true_coefficient)
                        else np.nan
                    ),
                    "Placebo_Reps": int(permutations),
                    "N": int(len(frame)),
                    "Claim_Boundary": contract["claim"],
                    "Temporal_Analysis_Population": manifest["analysis_population"],
                }
            )
            for rep, value in enumerate(values, start=1):
                draw_rows.append(
                    {
                        "Resolution": resolution,
                        "Stage": "purchase",
                        "Component": component,
                        "Permutation_Rep": rep,
                        "Coef_Placebo": float(value),
                        "Temporal_Analysis_Population": manifest["analysis_population"],
                    }
                )

    summary = pd.DataFrame(summary_rows)
    true_columns = [
        "Resolution",
        "Stage",
        "Component",
        "Treatment_Column",
        "Std_Error",
        "P_Value",
        "P_Value_Holm",
        "Significant_Holm_05",
        "Estimable",
        "Support_Flag",
    ]
    summary = summary.merge(
        true_ate[[column for column in true_columns if column in true_ate.columns]],
        on=["Resolution", "Stage", "Component"],
        how="left",
        validate="one_to_one",
    )
    summary.to_csv(save_dir / f"{dataset}_Purchase_Placebo_Test_Results_Detailed_{VERSION}.csv", index=False)
    pd.DataFrame(draw_rows).to_csv(
        save_dir / f"{dataset}_Purchase_Placebo_Test_Permutations_{VERSION}.csv",
        index=False,
    )
    logger.info("Completed %s certified placebo in %.2f seconds", dataset, time.time() - start)
