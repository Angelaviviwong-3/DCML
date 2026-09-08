#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observed-level H3 interaction tests using frozen 20260717 residuals."""

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

from dcml_formal_tests import (
    clustered_fit,
    holm_column,
    linear_contrast,
    wald_summary,
)
from dcml_inference import (
    clean_component_name,
    inference_contract,
    prepare_orthogonalized_data,
    support_diagnostics,
)
from dcml_utils import ensure_dir, setup_logging


OUTPUT_VERSION = "20260719"
SOURCE_VERSION = "20260717"


def _level_text(levels: list[int]) -> str:
    return "|".join(f"H{level}" for level in levels)


def _empty_detail(
    dataset: str,
    resolution: str,
    stage: str,
    target: str,
    support: dict,
    observed_levels: list[int],
    reason: str,
) -> dict:
    return {
        "Dataset": dataset,
        "Resolution": resolution,
        "Stage": stage,
        "Component": clean_component_name(target),
        "Treatment_Column": target,
        "Contrast": "N/A",
        "Reference": reason,
        "Difference": np.nan,
        "Std_Error": np.nan,
        "Z_Stat": np.nan,
        "P_Value": np.nan,
        "N": 0,
        "Covariance": "not_estimated",
        "Observed_H_Levels": _level_text(observed_levels),
        "Estimable_H_Levels": "",
        "Reference_H_Level": np.nan,
        "H_Level_Collapse_Detected": len(observed_levels) < 4,
        "Contrast_Valid": False,
        "GS_Fit_Split": "Set_B A-to-B treatment residuals",
        "Source_Residual_Version": SOURCE_VERSION,
        **support,
    }


def run_h3_observed_levels(root: Path, dataset: str, gs_mode: str = "linear") -> None:
    """Test moderation without inventing absent H levels or contrasts."""
    logger = setup_logging(ensure_dir(_dcml_paths.workspace_path(root, 'log')), f"h3_observed_levels_{dataset}_{OUTPUT_VERSION}")
    start = time.time()
    final, specs, _, _ = prepare_orthogonalized_data(root, dataset, gs_mode)
    contract = inference_contract(root, dataset)
    h = pd.to_numeric(final["H_level"], errors="coerce")
    observed_levels = sorted(int(value) for value in h.dropna().unique())
    if not observed_levels:
        raise RuntimeError(f"{dataset}: no observed H levels in the final residual table")

    detail_rows: list[dict] = []
    omnibus_rows: list[dict] = []
    for resolution, treatments in specs.items():
        support = {target: support_diagnostics(final, target) for target in treatments}
        estimable = [target for target in treatments if support[target]["Estimable"]]
        semantic = next(
            (target for target in estimable if clean_component_name(target) == "T_int_sem"),
            None,
        )
        base_treatments = pd.DataFrame(index=final.index)
        for target in estimable:
            values = pd.to_numeric(final[target], errors="coerce")
            base_treatments[target] = values.fillna(0.0) if target == semantic else values
        if semantic is not None:
            base_treatments["T_int_sem_history_available"] = pd.to_numeric(
                final[semantic], errors="coerce"
            ).notna().astype(float)

        for outcome_column in contract["outcomes"]:
            stage = outcome_column.replace("res_y_", "")
            outcome = pd.to_numeric(final[outcome_column], errors="coerce")
            for target in treatments:
                component = clean_component_name(target)
                if target not in estimable:
                    detail_rows.append(
                        _empty_detail(
                            dataset,
                            resolution,
                            stage,
                            target,
                            support[target],
                            observed_levels,
                            "N/A: treatment support gate failed",
                        )
                    )
                    omnibus_rows.append(
                        {
                            "Dataset": dataset,
                            "Resolution": resolution,
                            "Stage": stage,
                            "Component": component,
                            "Treatment_Column": target,
                            "Hypothesis": "equal treatment slope across actually observed estimable H levels",
                            "Statistic": np.nan,
                            "DF": np.nan,
                            "P_Value": np.nan,
                            "N": 0,
                            "Covariance": "not_estimated",
                            "Observed_H_Levels": _level_text(observed_levels),
                            "Estimable_H_Levels": "",
                            "Reference_H_Level": np.nan,
                            "H_Level_Collapse_Detected": len(observed_levels) < 4,
                            "Omnibus_Valid": False,
                            "Source_Residual_Version": SOURCE_VERSION,
                            **support[target],
                        }
                    )
                    continue

                target_values = pd.to_numeric(final[target], errors="coerce")
                if target == semantic:
                    target_levels = sorted(
                        int(value) for value in h[target_values.notna()].dropna().unique()
                    )
                else:
                    target_levels = observed_levels.copy()
                if len(target_levels) < 2:
                    detail_rows.append(
                        _empty_detail(
                            dataset,
                            resolution,
                            stage,
                            target,
                            support[target],
                            observed_levels,
                            "N/A: fewer than two estimable H levels",
                        )
                    )
                    omnibus_rows.append(
                        {
                            "Dataset": dataset,
                            "Resolution": resolution,
                            "Stage": stage,
                            "Component": component,
                            "Treatment_Column": target,
                            "Hypothesis": "equal treatment slope across actually observed estimable H levels",
                            "Statistic": np.nan,
                            "DF": np.nan,
                            "P_Value": np.nan,
                            "N": 0,
                            "Covariance": "not_estimated",
                            "Observed_H_Levels": _level_text(observed_levels),
                            "Estimable_H_Levels": _level_text(target_levels),
                            "Reference_H_Level": target_levels[0] if target_levels else np.nan,
                            "H_Level_Collapse_Detected": len(observed_levels) < 4,
                            "Omnibus_Valid": False,
                            "Source_Residual_Version": SOURCE_VERSION,
                            **support[target],
                        }
                    )
                    continue

                reference_level = 0 if target != semantic and 0 in target_levels else target_levels[0]
                comparison_levels = [level for level in target_levels if level != reference_level]
                design = base_treatments.copy()
                for level in observed_levels:
                    if level != observed_levels[0]:
                        design[f"H_level_{level}"] = h.eq(level).astype(float)

                contrast_specs: list[tuple[str, dict[str, float]]] = []
                if target == semantic:
                    design = design.drop(columns=[target])
                    slope_names: dict[int, str] = {}
                    filled = target_values.fillna(0.0)
                    for level in target_levels:
                        name = f"{target}__slope_H{level}"
                        design[name] = filled * h.eq(level).astype(float)
                        slope_names[level] = name
                    for level in comparison_levels:
                        contrast_specs.append(
                            (
                                f"H{level}-H{reference_level}",
                                {slope_names[level]: 1.0, slope_names[reference_level]: -1.0},
                            )
                        )
                else:
                    filled = target_values.fillna(0.0)
                    for level in comparison_levels:
                        name = f"{target}__x__H{level}"
                        design[name] = filled * h.eq(level).astype(float)
                        contrast_specs.append((f"H{level}-H{reference_level}", {name: 1.0}))

                model, model_index, covariance = clustered_fit(final, outcome, design)
                names = list(model.params.index)
                restriction = np.zeros((len(contrast_specs), len(names)), dtype=float)
                for row_index, (_, weights) in enumerate(contrast_specs):
                    for name, weight in weights.items():
                        restriction[row_index, names.index(name)] = weight
                omnibus = model.wald_test(restriction, scalar=True)

                for contrast_name, weights in contrast_specs:
                    estimate, standard_error, statistic, p_value = linear_contrast(model, weights)
                    detail_rows.append(
                        {
                            "Dataset": dataset,
                            "Resolution": resolution,
                            "Stage": stage,
                            "Component": component,
                            "Treatment_Column": target,
                            "Contrast": contrast_name,
                            "Reference": f"H{reference_level}",
                            "Difference": estimate,
                            "Std_Error": standard_error,
                            "Z_Stat": statistic,
                            "P_Value": p_value,
                            "N": int(len(model_index)),
                            "Covariance": covariance,
                            "Observed_H_Levels": _level_text(observed_levels),
                            "Estimable_H_Levels": _level_text(target_levels),
                            "Reference_H_Level": reference_level,
                            "H_Level_Collapse_Detected": len(observed_levels) < 4,
                            "Contrast_Valid": True,
                            "GS_Fit_Split": "Set_B A-to-B treatment residuals",
                            "Source_Residual_Version": SOURCE_VERSION,
                            **support[target],
                        }
                    )
                omnibus_rows.append(
                    {
                        "Dataset": dataset,
                        "Resolution": resolution,
                        "Stage": stage,
                        "Component": component,
                        "Treatment_Column": target,
                        "Hypothesis": "equal treatment slope across actually observed estimable H levels",
                        **wald_summary(model, omnibus, restriction),
                        "N": int(len(model_index)),
                        "Covariance": covariance,
                        "Observed_H_Levels": _level_text(observed_levels),
                        "Estimable_H_Levels": _level_text(target_levels),
                        "Reference_H_Level": reference_level,
                        "H_Level_Collapse_Detected": len(observed_levels) < 4,
                        "Omnibus_Valid": True,
                        "Source_Residual_Version": SOURCE_VERSION,
                        **support[target],
                    }
                )

    detail = holm_column(pd.DataFrame(detail_rows), ["Dataset", "Resolution", "Stage"])
    omnibus = holm_column(pd.DataFrame(omnibus_rows), ["Dataset", "Resolution", "Stage"])
    detail["Temporal_Analysis_Population"] = contract["population"]
    omnibus["Temporal_Analysis_Population"] = contract["population"]
    output = ensure_dir(_dcml_paths.workspace_path(root, 'processed_data') / dataset / "Final_Causal_Output")
    detail.to_csv(output / f"{dataset}_H3_interaction_contrasts_{OUTPUT_VERSION}.csv", index=False)
    omnibus.to_csv(output / f"{dataset}_H3_omnibus_tests_{OUTPUT_VERSION}.csv", index=False)
    with open(output / f"{dataset}_H3_level_contract_{OUTPUT_VERSION}.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset,
                "script_version": OUTPUT_VERSION,
                "source_residual_version": SOURCE_VERSION,
                "observed_h_levels": observed_levels,
                "level_collapse_detected": len(observed_levels) < 4,
                "contrast_policy": (
                    "Only actually observed levels are compared. Nonsemantic treatments use H0 when observed, "
                    "otherwise the lowest observed level; T_int_sem uses the lowest history-eligible level."
                ),
                "absent_level_policy": "no coefficient, contrast, or omnibus restriction is created for an absent H level",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(
        "Completed observed-level H3 tests for %s; levels=%s in %.2f seconds",
        dataset,
        observed_levels,
        time.time() - start,
    )
