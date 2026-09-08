#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formal H3 moderation and Suning-only H4 stage-contrast tests, 20260717."""

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
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from dcml_inference import (
    VERSION,
    clean_component_name,
    inference_contract,
    prepare_orthogonalized_data,
    support_diagnostics,
)
from dcml_utils import ensure_dir, setup_logging


def clustered_fit(df: pd.DataFrame, outcome: pd.Series, design: pd.DataFrame):
    frame = design.copy()
    frame["__outcome"] = pd.to_numeric(outcome, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) <= design.shape[1] + 5:
        raise RuntimeError("Insufficient complete cases for formal test")
    x = sm.add_constant(frame.drop(columns="__outcome"), has_constant="add")
    base = sm.OLS(frame["__outcome"], x)
    groups = df.loc[frame.index, ["cluster_user_id", "cluster_item_id"]]
    user_codes = pd.factorize(groups["cluster_user_id"].astype(str))[0]
    item_codes = pd.factorize(groups["cluster_item_id"].astype(str))[0]
    try:
        model = base.fit(
            cov_type="cluster",
            cov_kwds={"groups": np.column_stack([user_codes, item_codes])},
        )
        covariance = "two_way_cluster_user_item"
    except Exception:
        model = base.fit(cov_type="cluster", cov_kwds={"groups": user_codes})
        covariance = "cluster_user"
    return model, frame.index, covariance


def linear_contrast(model, weights: dict[str, float]) -> tuple[float, float, float, float]:
    names = list(model.params.index)
    vector = np.zeros(len(names), dtype=float)
    for name, weight in weights.items():
        if name not in names:
            raise KeyError(name)
        vector[names.index(name)] = weight
    estimate = float(vector @ np.asarray(model.params))
    variance = float(vector @ np.asarray(model.cov_params()) @ vector)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    statistic = estimate / standard_error if standard_error > 0 else np.nan
    p_value = float(2 * stats.norm.sf(abs(statistic))) if np.isfinite(statistic) else np.nan
    return estimate, standard_error, statistic, p_value


def holm_column(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["P_Value_Holm"] = np.nan
    for _, index in out.groupby(group_columns, dropna=False).groups.items():
        index = list(index)
        valid = out.loc[index, "P_Value"].notna()
        valid_index = list(out.loc[index].index[valid])
        if valid_index:
            out.loc[valid_index, "P_Value_Holm"] = multipletests(
                out.loc[valid_index, "P_Value"].to_numpy(dtype=float),
                method="holm",
            )[1]
    out["Significant_Holm_05"] = out["P_Value_Holm"] < 0.05
    return out


def scalar_float(value) -> float:
    arr = np.asarray(value).squeeze()
    return float(arr) if np.size(arr) else np.nan


def wald_summary(model, omnibus, restriction: np.ndarray) -> dict:
    """Extract Wald-test metadata across statsmodels versions.

    Some statsmodels releases expose df_num, while others expose only df_denom
    or no explicit degrees-of-freedom attribute for chi-square Wald tests. The
    effective test df is the rank of the restricted covariance block when the
    constraint covariance is rank deficient.
    """
    statistic = scalar_float(omnibus.statistic)
    p_value = scalar_float(omnibus.pvalue)
    nominal_df = int(np.linalg.matrix_rank(restriction))
    try:
        cov = np.asarray(model.cov_params(), dtype=float)
        restricted_cov = restriction @ cov @ restriction.T
        effective_df = int(np.linalg.matrix_rank(restricted_cov))
    except Exception:
        effective_df = nominal_df
    df_value = effective_df if effective_df > 0 else nominal_df
    df_value = int(max(1, round(float(df_value))))
    object_df = {}
    for attr in ("df_num", "df_denom", "df_constraint"):
        if hasattr(omnibus, attr):
            candidate = scalar_float(getattr(omnibus, attr))
            if np.isfinite(candidate):
                object_df[f"Statsmodels_{attr}"] = candidate
    if (not np.isfinite(p_value)) and np.isfinite(statistic):
        p_value = float(stats.chi2.sf(statistic, df_value))
        df_source = "constraint_covariance_rank_chi2_fallback"
    else:
        df_source = "constraint_covariance_rank"
    return {
        "Statistic": statistic,
        "DF": df_value,
        "DF_Nominal": nominal_df,
        "DF_Effective": effective_df,
        "DF_Source": df_source,
        "P_Value": p_value,
        **object_df,
    }


def run_h3_moderation(root: Path, dataset: str, gs_mode: str = "linear") -> None:
    logger = setup_logging(ensure_dir(_dcml_paths.workspace_path(root, 'log')), f"h3_moderation_{dataset}_{VERSION}")
    start = time.time()
    final, specs, _, _ = prepare_orthogonalized_data(root, dataset, gs_mode)
    contract = inference_contract(root, dataset)
    h = pd.to_numeric(final["H_level"], errors="coerce").fillna(0).astype(int)
    h_dummies = pd.DataFrame(
        {f"H{k}": (h == k).astype(float) for k in [1, 2, 3]},
        index=final.index,
    )
    detail_rows: list[dict] = []
    omnibus_rows: list[dict] = []

    for resolution, treatments in specs.items():
        support = {treatment: support_diagnostics(final, treatment) for treatment in treatments}
        estimable = [treatment for treatment in treatments if support[treatment]["Estimable"]]
        semantic_columns = [col for col in estimable if clean_component_name(col) == "T_int_sem"]
        semantic = semantic_columns[0] if semantic_columns else None
        base_treatments = pd.DataFrame(index=final.index)
        for treatment in estimable:
            values = pd.to_numeric(final[treatment], errors="coerce")
            base_treatments[treatment] = values.fillna(0.0) if treatment == semantic else values
        if semantic:
            base_treatments["T_int_sem_history_available"] = pd.to_numeric(
                final[semantic], errors="coerce"
            ).notna().astype(float)

        for outcome_column in contract["outcomes"]:
            stage = outcome_column.replace("res_y_", "")
            outcome = pd.to_numeric(final[outcome_column], errors="coerce")
            for target in treatments:
                component = clean_component_name(target)
                if target not in estimable:
                    contrasts = ["H2-H1", "H3-H1"] if component == "T_int_sem" else ["H1-H0", "H2-H0", "H3-H0"]
                    for contrast_name in contrasts:
                        detail_rows.append(
                            {
                                "Dataset": dataset,
                                "Resolution": resolution,
                                "Stage": stage,
                                "Component": component,
                                "Treatment_Column": target,
                                "Contrast": contrast_name,
                                "Reference": "N/A: support gate failed",
                                "Difference": np.nan,
                                "Std_Error": np.nan,
                                "Z_Stat": np.nan,
                                "P_Value": np.nan,
                                "N": 0,
                                "Covariance": "not_estimated",
                                "GS_Fit_Split": "Set_B A-to-B treatment residuals",
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
                            "Hypothesis": "equal treatment slope across estimable H levels",
                            "Statistic": np.nan,
                            "DF": np.nan,
                            "P_Value": np.nan,
                            "N": 0,
                            "Covariance": "not_estimated",
                            "Semantic_H0_Policy": "N/A" if component == "T_int_sem" else "not_applicable",
                            **support[target],
                        }
                    )
                    continue
                design = pd.concat([base_treatments, h_dummies], axis=1)
                target_values = pd.to_numeric(final[target], errors="coerce").fillna(0.0)
                if target == semantic:
                    design = design.drop(columns=[target])
                    slope_names = []
                    for level in [1, 2, 3]:
                        name = f"{target}__slope_H{level}"
                        design[name] = target_values * h_dummies[f"H{level}"]
                        slope_names.append(name)
                    model, model_index, covariance = clustered_fit(final, outcome, design)
                    contrast_specs = [
                        ("H2-H1", {slope_names[1]: 1.0, slope_names[0]: -1.0}),
                        ("H3-H1", {slope_names[2]: 1.0, slope_names[0]: -1.0}),
                    ]
                    restriction = np.zeros((2, len(model.params)))
                    names = list(model.params.index)
                    for row_index, (_, weights) in enumerate(contrast_specs):
                        for name, weight in weights.items():
                            restriction[row_index, names.index(name)] = weight
                    omnibus = model.wald_test(restriction, scalar=True)
                    reference = "H1; H0 has no pre-purchase semantic history and is N/A"
                else:
                    interaction_names = []
                    for level in [1, 2, 3]:
                        name = f"{target}__x__H{level}"
                        design[name] = target_values * h_dummies[f"H{level}"]
                        interaction_names.append(name)
                    model, model_index, covariance = clustered_fit(final, outcome, design)
                    contrast_specs = [
                        (f"H{level}-H0", {name: 1.0})
                        for level, name in zip([1, 2, 3], interaction_names)
                    ]
                    restriction = np.zeros((3, len(model.params)))
                    names = list(model.params.index)
                    for row_index, name in enumerate(interaction_names):
                        restriction[row_index, names.index(name)] = 1.0
                    omnibus = model.wald_test(restriction, scalar=True)
                    reference = "H0"

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
                            "Reference": reference,
                            "Difference": estimate,
                            "Std_Error": standard_error,
                            "Z_Stat": statistic,
                            "P_Value": p_value,
                            "N": int(len(model_index)),
                            "Covariance": covariance,
                            "GS_Fit_Split": "Set_B A-to-B treatment residuals",
                            **support[target],
                        }
                    )
                omnibus_record = wald_summary(model, omnibus, restriction)
                omnibus_rows.append(
                    {
                        "Dataset": dataset,
                        "Resolution": resolution,
                        "Stage": stage,
                        "Component": component,
                        "Treatment_Column": target,
                        "Hypothesis": "equal treatment slope across estimable H levels",
                        **omnibus_record,
                        "N": int(len(model_index)),
                        "Covariance": covariance,
                        "Semantic_H0_Policy": "N/A" if target == semantic else "not_applicable",
                        **support[target],
                    }
                )

    detail = holm_column(pd.DataFrame(detail_rows), ["Dataset", "Resolution", "Stage"])
    omnibus = holm_column(pd.DataFrame(omnibus_rows), ["Dataset", "Resolution", "Stage"])
    detail["Temporal_Analysis_Population"] = contract["population"]
    omnibus["Temporal_Analysis_Population"] = contract["population"]
    save_dir = ensure_dir(_dcml_paths.workspace_path(root, 'processed_data') / dataset / "Final_Causal_Output")
    detail.to_csv(save_dir / f"{dataset}_H3_interaction_contrasts_{VERSION}.csv", index=False)
    omnibus.to_csv(save_dir / f"{dataset}_H3_omnibus_tests_{VERSION}.csv", index=False)
    logger.info("Completed H3 formal tests for %s in %.2f seconds", dataset, time.time() - start)


def run_h4_stage_contrasts(root: Path, gs_mode: str = "linear") -> None:
    dataset = "suning"
    contract = inference_contract(root, dataset)
    logger = setup_logging(ensure_dir(_dcml_paths.workspace_path(root, 'log')), f"h4_stage_contrasts_suning_{VERSION}")
    start = time.time()
    final, specs, _, _ = prepare_orthogonalized_data(root, dataset, gs_mode)
    outcome_map = {
        "click": pd.to_numeric(final["res_y_click"], errors="coerce"),
        "cart": pd.to_numeric(final["res_y_cart"], errors="coerce"),
        "purchase": pd.to_numeric(final["res_y_purchase"], errors="coerce"),
    }
    tests = {
        "click": outcome_map["click"],
        "cart": outcome_map["cart"],
        "purchase": outcome_map["purchase"],
        "cart-click": outcome_map["cart"] - outcome_map["click"],
        "purchase-cart": outcome_map["purchase"] - outcome_map["cart"],
        "purchase-click": outcome_map["purchase"] - outcome_map["click"],
    }
    rows: list[dict] = []
    for resolution, treatments in specs.items():
        support = {treatment: support_diagnostics(final, treatment) for treatment in treatments}
        estimable = [treatment for treatment in treatments if support[treatment]["Estimable"]]
        skipped = [treatment for treatment in treatments if not support[treatment]["Estimable"]]
        design = final[estimable].apply(pd.to_numeric, errors="coerce")
        for test_name, outcome in tests.items():
            test_type = "stage_effect" if test_name in outcome_map else "stage_difference"
            for treatment in skipped:
                rows.append(
                    {
                        "Dataset": dataset,
                        "Resolution": resolution,
                        "Test_Type": test_type,
                        "Stage_or_Contrast": test_name,
                        "Component": clean_component_name(treatment),
                        "Treatment_Column": treatment,
                        "Estimate": np.nan,
                        "Std_Error": np.nan,
                        "CI_Lower_95": np.nan,
                        "CI_Upper_95": np.nan,
                        "P_Value": np.nan,
                        "N": 0,
                        "Covariance": "not_estimated",
                        "GS_Fit_Split": "Set_B A-to-B treatment residuals",
                        **support[treatment],
                    }
                )
            if not estimable:
                continue
            model, model_index, covariance = clustered_fit(final, outcome, design)
            for treatment in estimable:
                coefficient = float(model.params[treatment])
                standard_error = float(model.bse[treatment])
                rows.append(
                    {
                        "Dataset": dataset,
                        "Resolution": resolution,
                        "Test_Type": test_type,
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
                        "GS_Fit_Split": "Set_B A-to-B treatment residuals",
                        **support[treatment],
                    }
                )
    tests_frame = holm_column(
        pd.DataFrame(rows),
        ["Dataset", "Resolution", "Test_Type", "Stage_or_Contrast"],
    )
    tests_frame["Temporal_Analysis_Population"] = contract["population"]
    classification_rows = []
    for (resolution, component), group in tests_frame.groupby(["Resolution", "Component"]):
        indexed = group.set_index("Stage_or_Contrast")
        if not {"click", "purchase", "purchase-click"}.issubset(indexed.index):
            continue
        click = indexed.loc["click"]
        purchase = indexed.loc["purchase"]
        difference = indexed.loc["purchase-click"]
        if not bool(click["Estimable"] and purchase["Estimable"] and difference["Estimable"]):
            classification_rows.append(
                {
                    "Dataset": dataset,
                    "Resolution": resolution,
                    "Component": component,
                    "Click_Estimate": np.nan,
                    "Click_P_Holm": np.nan,
                    "Purchase_Estimate": np.nan,
                    "Purchase_P_Holm": np.nan,
                    "Purchase_Minus_Click": np.nan,
                    "Difference_P_Holm": np.nan,
                    "Pattern_Classification": "not_estimable_support_gate",
                    "Decision_Rule_Version": VERSION,
                }
            )
            continue
        opposite = float(click["Estimate"]) * float(purchase["Estimate"]) < 0
        endpoints_reliable = bool(click["Significant_Holm_05"] and purchase["Significant_Holm_05"])
        difference_reliable = bool(difference["Significant_Holm_05"])
        if opposite and endpoints_reliable and difference_reliable:
            label = "strict_reversal"
        elif opposite:
            label = "suggestive_reversal"
        elif (not click["Significant_Holm_05"]) and purchase["Significant_Holm_05"] and difference_reliable:
            label = "purchase_stage_emergence"
        elif np.sign(click["Estimate"]) == np.sign(purchase["Estimate"]) and difference_reliable and abs(purchase["Estimate"]) < abs(click["Estimate"]):
            label = "significant_attenuation"
        elif not click["Significant_Holm_05"] and not purchase["Significant_Holm_05"]:
            label = "no_reliable_endpoint_effect"
        else:
            label = "stable_or_other_stage_pattern"
        classification_rows.append(
            {
                "Dataset": dataset,
                "Resolution": resolution,
                "Component": component,
                "Click_Estimate": float(click["Estimate"]),
                "Click_P_Holm": float(click["P_Value_Holm"]),
                "Purchase_Estimate": float(purchase["Estimate"]),
                "Purchase_P_Holm": float(purchase["P_Value_Holm"]),
                "Purchase_Minus_Click": float(difference["Estimate"]),
                "Difference_P_Holm": float(difference["P_Value_Holm"]),
                "Pattern_Classification": label,
                "Decision_Rule_Version": VERSION,
            }
        )
    save_dir = ensure_dir(_dcml_paths.workspace_path(root, 'processed_data') / dataset / "Final_Causal_Output")
    for row in classification_rows:
        row["Temporal_Analysis_Population"] = contract["population"]
    tests_frame.to_csv(save_dir / f"suning_H4_stage_contrast_tests_{VERSION}.csv", index=False)
    pd.DataFrame(classification_rows).to_csv(
        save_dir / f"suning_H4_reversal_classification_{VERSION}.csv",
        index=False,
    )
    logger.info("Completed Suning H4 tests in %.2f seconds", time.time() - start)
