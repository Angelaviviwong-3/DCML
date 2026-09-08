#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-GS and reverse-order GS sensitivity analyses for 20260717."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[0]))
import project_paths as _dcml_paths


import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from dcml_inference import (
    VERSION,
    inference_contract,
    prepare_orthogonalized_data,
    projection_design,
    run_family,
)
from dcml_utils import add_holm_by_family, ensure_dir, read_table, setup_logging


def reverse_projection(
    gs_train: pd.DataFrame,
    final: pd.DataFrame,
    treatment_names: list[str],
    resolution: str,
    mode: str,
) -> tuple[pd.DataFrame, list[str]]:
    split = 3 if resolution == "Aggregate" else 6
    heuristic = [f"res_{name}" for name in treatment_names[:split]]
    analytical = [f"res_{name}" for name in treatment_names[split:]]
    train_x = projection_design(gs_train, analytical, mode)
    final_x = projection_design(final, analytical, mode).reindex(columns=train_x.columns, fill_value=0.0)
    out = final.copy()
    pure_heuristic = []
    for target in heuristic:
        train_y = pd.to_numeric(gs_train[target], errors="coerce")
        observed = train_y.notna() & train_x.notna().all(axis=1)
        pure = f"{target}__reverse_gs_pure"
        if observed.sum() <= train_x.shape[1] + 5:
            out[pure] = np.nan
        else:
            model = sm.OLS(train_y.loc[observed], train_x.loc[observed]).fit()
            raw = pd.to_numeric(final[target], errors="coerce").to_numpy(dtype=float)
            prediction = np.asarray(final_x @ model.params, dtype=float)
            out[pure] = np.where(np.isfinite(raw), raw - prediction, np.nan)
        pure_heuristic.append(pure)
    return out, pure_heuristic + analytical


def run_gs_sensitivity(root: Path, dataset: str, mode: str = "linear") -> None:
    logger = setup_logging(ensure_dir(_dcml_paths.workspace_path(root, 'log')), f"gs_sensitivity_{dataset}_{VERSION}")
    start = time.time()
    contract = inference_contract(root, dataset)
    with open(contract["schema"], "r", encoding="utf-8") as f:
        schema = json.load(f)
    final_raw = read_table(contract["final_residual"])
    gs_train = read_table(contract["gs_residual"])
    standard, standard_specs, _, _ = prepare_orthogonalized_data(root, dataset, mode)
    variants: dict[str, tuple[pd.DataFrame, dict[str, list[str]]]] = {
        "standard_heuristic_to_analytical": (standard, standard_specs),
        "no_gs": (
            final_raw,
            {
                "Aggregate": [f"res_{name}" for name in schema["aggregate_treatments"]],
                "Disaggregate": [f"res_{name}" for name in schema["disaggregate_treatments"]],
            },
        ),
    }
    reverse_specs = {}
    reverse_data = final_raw
    for resolution, key in [
        ("Aggregate", "aggregate_treatments"),
        ("Disaggregate", "disaggregate_treatments"),
    ]:
        reverse_data, reverse_specs[resolution] = reverse_projection(
            gs_train,
            reverse_data,
            list(schema[key]),
            resolution,
            mode,
        )
    variants["reverse_analytical_to_heuristic"] = (reverse_data, reverse_specs)

    rows = []
    for variant, (frame, specs) in variants.items():
        for resolution, treatments in specs.items():
            variant_rows = run_family(
                frame,
                contract["outcomes"],
                treatments,
                resolution,
                "All (ATE)",
                0,
                20260717,
                f"{mode}:{variant}",
                contract["claim"],
                contract["population"],
            )
            rows.extend(dict(row, GS_Variant=variant) for row in variant_rows)
    result = add_holm_by_family(
        pd.DataFrame(rows),
        ["GS_Variant", "Resolution", "Stage"],
    )
    save_dir = ensure_dir(_dcml_paths.workspace_path(root, 'processed_data') / dataset / "Final_Causal_Output")
    result.to_csv(save_dir / f"{dataset}_GS_order_sensitivity_{VERSION}.csv", index=False)
    logger.info("Completed GS sensitivity for %s in %.2f seconds", dataset, time.time() - start)
