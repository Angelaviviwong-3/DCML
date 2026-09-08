#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare history-complete 6/10 models with full-certified 5/9 models."""

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

from dcml_formal_tests import clustered_fit, holm_column
from dcml_inference import (
    clean_component_name,
    inference_contract,
    prepare_orthogonalized_data,
    support_diagnostics,
)
from dcml_utils import ensure_dir, setup_logging


OUTPUT_VERSION = "20260719"
SOURCE_VERSION = "20260717"


def _fit_specification(
    final: pd.DataFrame,
    dataset: str,
    resolution: str,
    label: str,
    requested: list[str],
    outcomes: list[str],
    full_n: int,
) -> list[dict]:
    diagnostics = {column: support_diagnostics(final, column) for column in requested}
    estimable = [column for column in requested if diagnostics[column]["Estimable"]]
    semantic_is_estimable = any(
        clean_component_name(column) == "T_int_sem" for column in estimable
    )
    analysis_sample = (
        "pre-event semantic-history complete cases"
        if semantic_is_estimable
        else "full certified sample after treatment support gating"
    )
    rows: list[dict] = []
    for outcome in outcomes:
        stage = outcome.replace("res_y_", "")
        if not estimable:
            continue
        design = final[estimable].apply(pd.to_numeric, errors="coerce")
        model, model_index, covariance = clustered_fit(
            final,
            pd.to_numeric(final[outcome], errors="coerce"),
            design,
        )
        for treatment in requested:
            component = clean_component_name(treatment)
            if treatment not in estimable:
                rows.append(
                    {
                        "Dataset": dataset,
                        "Resolution": resolution,
                        "Model_Specification": label,
                        "Stage": stage,
                        "Component": component,
                        "Treatment_Column": treatment,
                        "Coefficient": np.nan,
                        "Std_Error": np.nan,
                        "CI_Lower_95": np.nan,
                        "CI_Upper_95": np.nan,
                        "P_Value": np.nan,
                        "N": 0,
                        "Full_Certified_N": full_n,
                        "Complete_Case_Rate": 0.0,
                        "Covariance": "not_estimated",
                        "Requested_Treatment_Count": len(requested),
                        "Estimated_Treatment_Count": len(estimable),
                        "Includes_T_int_sem": any(clean_component_name(c) == "T_int_sem" for c in requested),
                        "Analysis_Sample": analysis_sample,
                        "Source_Residual_Version": SOURCE_VERSION,
                        **diagnostics[treatment],
                    }
                )
                continue
            coefficient = float(model.params[treatment])
            standard_error = float(model.bse[treatment])
            rows.append(
                {
                    "Dataset": dataset,
                    "Resolution": resolution,
                    "Model_Specification": label,
                    "Stage": stage,
                    "Component": component,
                    "Treatment_Column": treatment,
                    "Coefficient": coefficient,
                    "Std_Error": standard_error,
                    "CI_Lower_95": coefficient - 1.96 * standard_error,
                    "CI_Upper_95": coefficient + 1.96 * standard_error,
                    "P_Value": float(model.pvalues[treatment]),
                    "N": int(len(model_index)),
                    "Full_Certified_N": full_n,
                    "Complete_Case_Rate": float(len(model_index) / full_n),
                    "Covariance": covariance,
                    "Requested_Treatment_Count": len(requested),
                    "Estimated_Treatment_Count": len(estimable),
                    "Includes_T_int_sem": any(clean_component_name(c) == "T_int_sem" for c in requested),
                    "Analysis_Sample": analysis_sample,
                    "Source_Residual_Version": SOURCE_VERSION,
                    **diagnostics[treatment],
                }
            )
    return rows


def _comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["Dataset", "Resolution", "Stage", "Component"]
    primary = results[results["Model_Specification"].str.startswith("Primary_support_gated")].copy()
    reduced = results[results["Model_Specification"].str.startswith("Full_certified")].copy()
    columns = keys + ["Coefficient", "P_Value_Holm", "N"]
    merged = primary[columns].merge(reduced[columns], on=keys, suffixes=("_Primary", "_Full"))
    merged["Coefficient_Difference_Full_Minus_Primary"] = (
        merged["Coefficient_Full"] - merged["Coefficient_Primary"]
    )
    denominator = merged["Coefficient_Primary"].abs().replace(0.0, np.nan)
    merged["Relative_Absolute_Change"] = (
        merged["Coefficient_Difference_Full_Minus_Primary"].abs() / denominator
    )
    merged["Sign_Stable"] = np.sign(merged["Coefficient_Primary"]) == np.sign(merged["Coefficient_Full"])
    merged["Holm_Significance_Stable"] = (
        merged["P_Value_Holm_Primary"].lt(0.05) == merged["P_Value_Holm_Full"].lt(0.05)
    )
    merged["Interpretation"] = (
        "descriptive specification sensitivity; coefficients use different analysis populations"
    )
    return merged


def _classify_suning_h4(tests: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (resolution, component), group in tests.groupby(["Resolution", "Component"]):
        indexed = group.set_index("Stage_or_Contrast")
        if not {"click", "purchase", "purchase-click"}.issubset(indexed.index):
            continue
        click = indexed.loc["click"]
        purchase = indexed.loc["purchase"]
        difference = indexed.loc["purchase-click"]
        opposite = float(click["Estimate"]) * float(purchase["Estimate"]) < 0
        endpoints_reliable = bool(click["Significant_Holm_05"] and purchase["Significant_Holm_05"])
        difference_reliable = bool(difference["Significant_Holm_05"])
        if opposite and endpoints_reliable and difference_reliable:
            label = "strict_reversal"
        elif opposite:
            label = "suggestive_reversal"
        elif (not click["Significant_Holm_05"]) and purchase["Significant_Holm_05"] and difference_reliable:
            label = "purchase_stage_emergence"
        elif (
            np.sign(click["Estimate"]) == np.sign(purchase["Estimate"])
            and difference_reliable
            and abs(purchase["Estimate"]) < abs(click["Estimate"])
        ):
            label = "significant_attenuation"
        elif not click["Significant_Holm_05"] and not purchase["Significant_Holm_05"]:
            label = "no_reliable_endpoint_effect"
        else:
            label = "stable_or_other_stage_pattern"
        rows.append(
            {
                "Dataset": "suning",
                "Resolution": resolution,
                "Component": component,
                "Model_Specification": str(click["Model_Specification"]),
                "Click_Estimate": float(click["Estimate"]),
                "Click_P_Holm": float(click["P_Value_Holm"]),
                "Purchase_Estimate": float(purchase["Estimate"]),
                "Purchase_P_Holm": float(purchase["P_Value_Holm"]),
                "Purchase_Minus_Click": float(difference["Estimate"]),
                "Difference_P_Holm": float(difference["P_Value_Holm"]),
                "Pattern_Classification": label,
                "Decision_Rule_Version": OUTPUT_VERSION,
                "Source_Residual_Version": SOURCE_VERSION,
            }
        )
    return pd.DataFrame(rows)


def _suning_reduced_h4(final: pd.DataFrame, specifications: dict[str, list[str]]) -> pd.DataFrame:
    outcomes = {
        "click": pd.to_numeric(final["res_y_click"], errors="coerce"),
        "cart": pd.to_numeric(final["res_y_cart"], errors="coerce"),
        "purchase": pd.to_numeric(final["res_y_purchase"], errors="coerce"),
    }
    tests = {
        **outcomes,
        "cart-click": outcomes["cart"] - outcomes["click"],
        "purchase-cart": outcomes["purchase"] - outcomes["cart"],
        "purchase-click": outcomes["purchase"] - outcomes["click"],
    }
    rows: list[dict] = []
    for resolution, requested in specifications.items():
        treatments = [column for column in requested if clean_component_name(column) != "T_int_sem"]
        design = final[treatments].apply(pd.to_numeric, errors="coerce")
        for test_name, outcome in tests.items():
            model, model_index, covariance = clustered_fit(final, outcome, design)
            for treatment in treatments:
                coefficient = float(model.params[treatment])
                standard_error = float(model.bse[treatment])
                rows.append(
                    {
                        "Dataset": "suning",
                        "Resolution": resolution,
                        "Model_Specification": f"Full_certified_{len(treatments)}_without_T_int_sem",
                        "Test_Type": "stage_effect" if test_name in outcomes else "stage_difference",
                        "Stage_or_Contrast": test_name,
                        "Component": clean_component_name(treatment),
                        "Treatment_Column": treatment,
                        "Estimate": coefficient,
                        "Std_Error": standard_error,
                        "CI_Lower_95": coefficient - 1.96 * standard_error,
                        "CI_Upper_95": coefficient + 1.96 * standard_error,
                        "P_Value": float(model.pvalues[treatment]),
                        "N": int(len(model_index)),
                        "Covariance": covariance,
                        "Source_Residual_Version": SOURCE_VERSION,
                    }
                )
    return holm_column(
        pd.DataFrame(rows),
        ["Dataset", "Resolution", "Test_Type", "Stage_or_Contrast"],
    )


def run_specification_sensitivity(root: Path, dataset: str, gs_mode: str = "linear") -> None:
    logger = setup_logging(ensure_dir(_dcml_paths.workspace_path(root, 'log')), f"specification_sensitivity_{dataset}_{OUTPUT_VERSION}")
    start = time.time()
    contract = inference_contract(root, dataset)
    final, specifications, _, _ = prepare_orthogonalized_data(root, dataset, gs_mode)
    full_n = len(final)
    rows: list[dict] = []
    reduced_specs: dict[str, list[str]] = {}
    for resolution, treatments in specifications.items():
        reduced = [column for column in treatments if clean_component_name(column) != "T_int_sem"]
        reduced_specs[resolution] = reduced
        rows.extend(
            _fit_specification(
                final,
                dataset,
                resolution,
                f"Primary_support_gated_{len(treatments)}_requested_with_T_int_sem",
                treatments,
                contract["outcomes"],
                full_n,
            )
        )
        rows.extend(
            _fit_specification(
                final,
                dataset,
                resolution,
                f"Full_certified_{len(reduced)}_without_T_int_sem",
                reduced,
                contract["outcomes"],
                full_n,
            )
        )
    results = holm_column(
        pd.DataFrame(rows),
        ["Dataset", "Resolution", "Model_Specification", "Stage"],
    )
    results["Temporal_Analysis_Population"] = contract["population"]
    comparison = _comparison_table(results)
    output = ensure_dir(_dcml_paths.workspace_path(root, 'processed_data') / dataset / "Final_Causal_Output")
    results.to_csv(output / f"{dataset}_ATE_specification_sensitivity_{OUTPUT_VERSION}.csv", index=False)
    comparison.to_csv(
        output / f"{dataset}_ATE_specification_sensitivity_comparison_{OUTPUT_VERSION}.csv",
        index=False,
    )
    if dataset == "suning":
        h4 = _suning_reduced_h4(final, reduced_specs)
        h4["Temporal_Analysis_Population"] = contract["population"]
        h4.to_csv(output / f"suning_H4_full_certified_specification_sensitivity_{OUTPUT_VERSION}.csv", index=False)
        classification = _classify_suning_h4(h4)
        classification["Temporal_Analysis_Population"] = contract["population"]
        classification.to_csv(
            output / f"suning_H4_full_certified_reversal_classification_{OUTPUT_VERSION}.csv",
            index=False,
        )
    logger.info("Completed specification sensitivity for %s in %.2f seconds", dataset, time.time() - start)
