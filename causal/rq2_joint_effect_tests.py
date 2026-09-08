#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ2 joint pathway tests, component-family tests, and precision diagnostics."""

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
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

from dcml_formal_tests import clustered_fit, wald_summary
from dcml_inference import (
    clean_component_name,
    inference_contract,
    prepare_orthogonalized_data,
    support_diagnostics,
)
from dcml_utils import ensure_dir, setup_logging


OUTPUT_VERSION = "20260723"
SOURCE_VERSION = "20260717"
ALPHA = 0.05
TARGET_POWER = 0.80

MNC_COMPONENTS = {
    "T_con_mkt",
    "T_con_soc",
    "T_con_rat",
    "a_pri",
    "a_gft",
    "a_sub",
    "a_urg",
}
MAI_COMPONENTS = {
    "T_int_fac",
    "T_int_vis",
    "T_int_sem",
    "a_spec",
    "a_str",
}
COMPONENT_FAMILIES = {
    "MNC_marketing_components": {"a_pri", "a_gft", "a_sub", "a_urg"},
    "MAI_functional_visual_components": {"a_spec", "a_str"},
}


def component_pathway(component: str) -> str:
    if component in MNC_COMPONENTS:
        return "MNC"
    if component in MAI_COMPONENTS:
        return "MAI"
    raise ValueError(f"Unmapped RQ2 component: {component}")


def restriction_matrix(model, columns: list[str]) -> np.ndarray:
    names = list(model.params.index)
    missing = [column for column in columns if column not in names]
    if missing:
        raise KeyError(f"Model does not contain restricted columns: {missing}")
    restriction = np.zeros((len(columns), len(names)), dtype=float)
    for row_index, column in enumerate(columns):
        restriction[row_index, names.index(column)] = 1.0
    return restriction


def joint_test_record(
    model,
    columns: list[str],
    component_names: list[str],
) -> dict:
    restriction = restriction_matrix(model, columns)
    test = model.wald_test(restriction, scalar=True)
    return {
        **wald_summary(model, test, restriction),
        "Cue_Count": len(columns),
        "Components": "|".join(component_names),
    }


def adjusted_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["P_Value_Holm_FWER"] = np.nan
    out["P_Value_BH_FDR"] = np.nan
    valid = out["P_Value"].notna()
    if valid.any():
        p_values = out.loc[valid, "P_Value"].to_numpy(dtype=float)
        out.loc[valid, "P_Value_Holm_FWER"] = multipletests(
            p_values,
            alpha=ALPHA,
            method="holm",
        )[1]
        out.loc[valid, "P_Value_BH_FDR"] = multipletests(
            p_values,
            alpha=ALPHA,
            method="fdr_bh",
        )[1]
    out["Significant_Raw_05"] = out["P_Value"] < ALPHA
    out["Significant_Holm_FWER_05"] = out["P_Value_Holm_FWER"] < ALPHA
    out["Significant_BH_FDR_05"] = out["P_Value_BH_FDR"] < ALPHA
    return out


def pathway_adjustment(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["P_Value_Holm_Pathways"] = np.nan
    out["Significant_Holm_Pathways_05"] = False
    pathway_mask = out["Test_Scope"].eq("pathway") & out["P_Value"].notna()
    if pathway_mask.any():
        adjusted = multipletests(
            out.loc[pathway_mask, "P_Value"].to_numpy(dtype=float),
            alpha=ALPHA,
            method="holm",
        )[1]
        out.loc[pathway_mask, "P_Value_Holm_Pathways"] = adjusted
        out.loc[pathway_mask, "Significant_Holm_Pathways_05"] = adjusted < ALPHA
    return out


def precision_record(
    dataset: str,
    resolution: str,
    stage: str,
    component: str,
    treatment: str,
    coefficient: float,
    standard_error: float,
    diagnostics: dict,
    n: int,
    covariance: str,
) -> dict:
    z_alpha = stats.norm.ppf(1.0 - ALPHA / 2.0)
    z_power = stats.norm.ppf(TARGET_POWER)
    mde = float((z_alpha + z_power) * standard_error)
    residual_iqr = float(diagnostics.get("Residual_IQR_Final", np.nan))
    return {
        "Dataset": dataset,
        "Resolution": resolution,
        "Stage": stage,
        "Pathway": component_pathway(component),
        "Component": component,
        "Treatment_Column": treatment,
        "Coefficient": coefficient,
        "Std_Error": standard_error,
        "Absolute_Effect": abs(coefficient),
        "Approx_MDE_80pct_Power_Alpha05": mde,
        "Effect_to_MDE_Ratio": abs(coefficient) / mde if mde > 0 else np.nan,
        "Residual_IQR": residual_iqr,
        "Observed_IQR_Effect_pp": coefficient * residual_iqr * 100.0,
        "Approx_MDE_IQR_pp": mde * residual_iqr * 100.0,
        "N": n,
        "Covariance": covariance,
        "Interpretation": (
            "precision diagnostic only; non-significance does not establish a zero effect"
        ),
        **diagnostics,
    }


def run_rq2_joint_tests(
    root: Path,
    dataset: str,
    gs_mode: str = "linear",
) -> None:
    contract = inference_contract(root, dataset)
    logger = setup_logging(
        ensure_dir(_dcml_paths.workspace_path(root, 'log')),
        f"rq2_joint_effect_tests_{dataset}_{OUTPUT_VERSION}",
    )
    started = time.time()
    final, specs, _, _ = prepare_orthogonalized_data(root, dataset, gs_mode)

    individual_frames: list[pd.DataFrame] = []
    pathway_rows: list[dict] = []
    family_rows: list[dict] = []
    precision_rows: list[dict] = []

    for resolution, treatments in specs.items():
        diagnostics = {
            treatment: support_diagnostics(final, treatment)
            for treatment in treatments
        }
        estimable = [
            treatment
            for treatment in treatments
            if diagnostics[treatment]["Estimable"]
        ]
        skipped = [
            treatment
            for treatment in treatments
            if not diagnostics[treatment]["Estimable"]
        ]
        design = final[estimable].apply(pd.to_numeric, errors="coerce")

        for outcome_column in contract["outcomes"]:
            stage = outcome_column.replace("res_y_", "")
            outcome = pd.to_numeric(final[outcome_column], errors="coerce")
            model, model_index, covariance = clustered_fit(final, outcome, design)

            individual_rows: list[dict] = []
            for treatment in estimable:
                component = clean_component_name(treatment)
                coefficient = float(model.params[treatment])
                standard_error = float(model.bse[treatment])
                individual_rows.append(
                    {
                        "Dataset": dataset,
                        "Resolution": resolution,
                        "Stage": stage,
                        "Pathway": component_pathway(component),
                        "Component": component,
                        "Treatment_Column": treatment,
                        "Coefficient": coefficient,
                        "Std_Error": standard_error,
                        "CI_Lower_95": coefficient - 1.96 * standard_error,
                        "CI_Upper_95": coefficient + 1.96 * standard_error,
                        "P_Value": float(model.pvalues[treatment]),
                        "N": int(len(model_index)),
                        "Covariance": covariance,
                        "Estimable": True,
                        "Support_Flag": diagnostics[treatment]["Support_Flag"],
                    }
                )
                precision_rows.append(
                    precision_record(
                        dataset,
                        resolution,
                        stage,
                        component,
                        treatment,
                        coefficient,
                        standard_error,
                        diagnostics[treatment],
                        int(len(model_index)),
                        covariance,
                    )
                )
            for treatment in skipped:
                component = clean_component_name(treatment)
                individual_rows.append(
                    {
                        "Dataset": dataset,
                        "Resolution": resolution,
                        "Stage": stage,
                        "Pathway": component_pathway(component),
                        "Component": component,
                        "Treatment_Column": treatment,
                        "Coefficient": np.nan,
                        "Std_Error": np.nan,
                        "CI_Lower_95": np.nan,
                        "CI_Upper_95": np.nan,
                        "P_Value": np.nan,
                        "N": 0,
                        "Covariance": "not_estimated",
                        "Estimable": False,
                        "Support_Flag": diagnostics[treatment]["Support_Flag"],
                    }
                )
            individual_frames.append(adjusted_columns(pd.DataFrame(individual_rows)))

            all_components = [clean_component_name(column) for column in estimable]
            global_record = joint_test_record(model, estimable, all_components)
            pathway_rows.append(
                {
                    "Dataset": dataset,
                    "Resolution": resolution,
                    "Stage": stage,
                    "Test_Scope": "global",
                    "Pathway": "All",
                    **global_record,
                    "N": int(len(model_index)),
                    "Covariance": covariance,
                }
            )
            for pathway in ("MNC", "MAI"):
                columns = [
                    column
                    for column in estimable
                    if component_pathway(clean_component_name(column)) == pathway
                ]
                if not columns:
                    continue
                components = [clean_component_name(column) for column in columns]
                pathway_rows.append(
                    {
                        "Dataset": dataset,
                        "Resolution": resolution,
                        "Stage": stage,
                        "Test_Scope": "pathway",
                        "Pathway": pathway,
                        **joint_test_record(model, columns, components),
                        "N": int(len(model_index)),
                        "Covariance": covariance,
                    }
                )

            if resolution == "Disaggregate":
                for family_name, family_components in COMPONENT_FAMILIES.items():
                    columns = [
                        column
                        for column in estimable
                        if clean_component_name(column) in family_components
                    ]
                    if not columns:
                        continue
                    components = [clean_component_name(column) for column in columns]
                    family_rows.append(
                        {
                            "Dataset": dataset,
                            "Resolution": resolution,
                            "Stage": stage,
                            "Component_Family": family_name,
                            **joint_test_record(model, columns, components),
                            "N": int(len(model_index)),
                            "Covariance": covariance,
                        }
                    )

    individual = pd.concat(individual_frames, ignore_index=True)
    pathway = (
        pd.DataFrame(pathway_rows)
        .groupby(["Dataset", "Resolution", "Stage"], group_keys=False)
        .apply(pathway_adjustment)
        .reset_index(drop=True)
    )
    component_family = pd.DataFrame(family_rows)
    if not component_family.empty:
        component_family = (
            component_family.groupby(
                ["Dataset", "Resolution", "Stage"],
                group_keys=False,
            )
            .apply(adjusted_columns)
            .reset_index(drop=True)
        )
    precision = pd.DataFrame(precision_rows)

    output_dir = ensure_dir(
        _dcml_paths.workspace_path(root, 'processed_data') / dataset / "Final_Causal_Output"
    )
    prefix = f"{dataset}_RQ2"
    individual.to_csv(
        output_dir
        / f"{prefix}_individual_effects_multiplicity_{OUTPUT_VERSION}.csv",
        index=False,
    )
    pathway.to_csv(
        output_dir / f"{prefix}_pathway_omnibus_tests_{OUTPUT_VERSION}.csv",
        index=False,
    )
    component_family.to_csv(
        output_dir / f"{prefix}_component_family_tests_{OUTPUT_VERSION}.csv",
        index=False,
    )
    precision.to_csv(
        output_dir / f"{prefix}_precision_diagnostics_{OUTPUT_VERSION}.csv",
        index=False,
    )
    manifest = {
        "dataset": dataset,
        "output_version": OUTPUT_VERSION,
        "source_residual_version": SOURCE_VERSION,
        "orthogonalization_mode": gs_mode,
        "primary_individual_error_control": (
            "Holm family-wise error rate within dataset, resolution, and stage"
        ),
        "supplementary_individual_error_control": (
            "Benjamini-Hochberg false discovery rate within the same family"
        ),
        "pathway_tests": (
            "prespecified joint Wald tests of all supported MNC coefficients and "
            "all supported MAI coefficients"
        ),
        "component_family_tests": (
            "joint Wald tests for the four marketing components and two "
            "functional-visual components in the component-level model"
        ),
        "precision_diagnostic": {
            "target_power": TARGET_POWER,
            "two_sided_alpha": ALPHA,
            "formula": "(z_(1-alpha/2) + z_power) * clustered standard error",
            "claim_boundary": (
                "approximate minimum detectable effect; not post hoc observed power"
            ),
        },
        "reporting_rule": (
            "BH-FDR is supplementary and must not replace the prespecified Holm "
            "result based on which method yields more discoveries."
        ),
        "claim_boundary": contract["claim"],
    }
    with open(
        output_dir / f"{prefix}_testing_manifest_{OUTPUT_VERSION}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    logger.info(
        "Completed RQ2 joint tests for %s in %.2f seconds",
        dataset,
        time.time() - started,
    )
