#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimand-aligned causal baselines for all three frozen DCML datasets."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[0]))
import project_paths as _dcml_paths


import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

from dcml_formal_tests import holm_column
from dcml_inference import clean_component_name, inference_contract
from dcml_nuisance import dataset_contract
from dcml_utils import ensure_dir, find_causal_root, read_table, setup_logging
from temporal_certification import filter_certified_frame, load_certification


OUTPUT_VERSION = "20260719"
SOURCE_VERSION = "20260717"
DATASETS = ("suning", "amazon_appliances", "amazon_beauty")


def _clustered_linear(
    frame: pd.DataFrame,
    outcome: pd.Series,
    design: pd.DataFrame,
    weights: np.ndarray | None = None,
):
    work = design.copy()
    work["__outcome"] = pd.to_numeric(outcome, errors="coerce")
    if weights is not None:
        work["__weight"] = weights
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    y = work.pop("__outcome")
    w = work.pop("__weight") if "__weight" in work else None
    x = sm.add_constant(work, has_constant="add")
    base = sm.WLS(y, x, weights=w) if w is not None else sm.OLS(y, x)
    groups = frame.loc[work.index, ["cluster_user_id", "cluster_item_id"]]
    user = pd.factorize(groups["cluster_user_id"].astype(str))[0]
    item = pd.factorize(groups["cluster_item_id"].astype(str))[0]
    try:
        model = base.fit(cov_type="cluster", cov_kwds={"groups": np.column_stack([user, item])})
        covariance = "two_way_cluster_user_item"
    except Exception:
        model = base.fit(cov_type="cluster", cov_kwds={"groups": user})
        covariance = "cluster_user"
    return model, work.index, covariance


def _record_rows(
    dataset: str,
    estimator: str,
    resolution: str,
    specification: str,
    stage: str,
    treatments: list[str],
    model,
    model_index: pd.Index,
    covariance: str,
    full_n: int,
    extra: dict | None = None,
    parameter_names: dict[str, str] | None = None,
) -> list[dict]:
    rows = []
    for treatment in treatments:
        parameter = (parameter_names or {}).get(treatment, treatment)
        coefficient = float(model.params[parameter])
        standard_error = float(model.bse[parameter])
        rows.append(
            {
                "Dataset": dataset,
                "Estimator": estimator,
                "Resolution": resolution,
                "Model_Specification": specification,
                "Stage": stage,
                "Component": clean_component_name(treatment),
                "Treatment_Column": treatment,
                "Coefficient": coefficient,
                "Std_Error": standard_error,
                "CI_Lower_95": coefficient - 1.96 * standard_error,
                "CI_Upper_95": coefficient + 1.96 * standard_error,
                "P_Value": float(model.pvalues[parameter]),
                "N": int(len(model_index)),
                "Full_Certified_N": int(full_n),
                "Complete_Case_Rate": float(len(model_index) / full_n),
                "Covariance": covariance,
                "Specification_Size": len(treatments),
                "Includes_T_int_sem": any(clean_component_name(column) == "T_int_sem" for column in treatments),
                "Source_Data_Version": SOURCE_VERSION,
                **(extra or {}),
            }
        )
    return rows


def _stabilized_gps_weights(
    raw_t: np.ndarray,
    treatment_hat: np.ndarray,
    training_residual_sd: float,
    trim_quantile: float,
) -> tuple[np.ndarray, dict]:
    marginal_sd = float(np.std(raw_t, ddof=1))
    if marginal_sd <= 1e-8 or training_residual_sd <= 1e-8:
        raise ValueError("insufficient treatment variance for GPS")
    log_num = stats.norm.logpdf(raw_t, loc=float(np.mean(raw_t)), scale=marginal_sd)
    log_den = stats.norm.logpdf(raw_t, loc=treatment_hat, scale=training_residual_sd)
    raw_weight = np.exp(np.clip(log_num - log_den, -20.0, 20.0))
    cap = float(np.quantile(raw_weight[np.isfinite(raw_weight)], trim_quantile))
    weights = np.minimum(np.nan_to_num(raw_weight, nan=cap, posinf=cap, neginf=0.0), cap)
    weights /= max(float(weights.mean()), 1e-12)
    return weights, {
        "GPS_Training_Residual_SD": training_residual_sd,
        "GPS_Weight_Trim_Quantile": trim_quantile,
        "GPS_Weight_P99": float(np.quantile(weights, 0.99)),
        "GPS_Weight_Max": float(weights.max()),
        "GPS_Effective_Sample_Size": float(weights.sum() ** 2 / np.square(weights).sum()),
    }


def _raw_frame(root: Path, dataset: str, certified_items: set[str]) -> pd.DataFrame:
    frame = filter_certified_frame(read_table(dataset_contract(root, dataset)["c"]), certified_items)
    frame = frame.reset_index(drop=True)
    if "T_int_sem" in frame.columns and "H_level" in frame.columns:
        no_history = pd.to_numeric(frame["H_level"], errors="coerce").fillna(0).eq(0)
        frame.loc[no_history, "T_int_sem"] = np.nan
    frame["cluster_user_id"] = frame["user_id"].astype(str)
    frame["cluster_item_id"] = frame["item_id"].astype(str)
    return frame


def run_dataset(root: Path, dataset: str, trim_quantile: float) -> None:
    logger = setup_logging(ensure_dir(_dcml_paths.workspace_path(root, 'log')), f"causal_baselines_{dataset}_{OUTPUT_VERSION}")
    start = time.time()
    contract = inference_contract(root, dataset)
    with open(contract["schema"], "r", encoding="utf-8") as handle:
        schema = json.load(handle)
    certified_items, _, _ = load_certification(root, dataset)
    raw = _raw_frame(root, dataset, certified_items)
    residual = read_table(contract["final_residual"]).reset_index(drop=True)
    gs_residual = read_table(contract["gs_residual"]).reset_index(drop=True)
    specifications = {
        "Aggregate": list(schema["aggregate_treatments"]),
        "Disaggregate": list(schema["disaggregate_treatments"]),
    }
    rows: list[dict] = []

    for resolution, full_treatments in specifications.items():
        reduced_treatments = [t for t in full_treatments if clean_component_name(t) != "T_int_sem"]
        for specification, treatments in (
            (f"History_complete_{len(full_treatments)}", full_treatments),
            (f"Full_certified_{len(reduced_treatments)}_without_T_int_sem", reduced_treatments),
        ):
            for outcome_column in contract["outcomes"]:
                raw_outcome = outcome_column.replace("res_", "")
                stage = raw_outcome.replace("y_", "")
                raw_design = raw[treatments].apply(pd.to_numeric, errors="coerce")
                model, index, covariance = _clustered_linear(
                    raw,
                    pd.to_numeric(raw[raw_outcome], errors="coerce"),
                    raw_design,
                )
                rows.extend(
                    _record_rows(
                        dataset,
                        "Naive OLS",
                        resolution,
                        specification,
                        stage,
                        treatments,
                        model,
                        index,
                        covariance,
                        len(raw),
                        {"Estimator_Contract": "joint raw-outcome association; not unconfounded"},
                    )
                )

                residual_treatments = [f"res_{treatment}" for treatment in treatments]
                residual_design = residual[residual_treatments].apply(pd.to_numeric, errors="coerce")
                dml_model, dml_index, dml_covariance = _clustered_linear(
                    residual,
                    pd.to_numeric(residual[outcome_column], errors="coerce"),
                    residual_design,
                )
                rows.extend(
                    _record_rows(
                        dataset,
                        "Standard DML (No GS)",
                        resolution,
                        specification,
                        stage,
                        treatments,
                        dml_model,
                        dml_index,
                        dml_covariance,
                        len(residual),
                        {"Estimator_Contract": "Set-B-fitted nuisance residuals; no GS projection"},
                        {treatment: f"res_{treatment}" for treatment in treatments},
                    )
                )

        # GPS is intentionally marginal and listed once per unique treatment.
        for treatment in full_treatments:
            residual_treatment = f"res_{treatment}"
            raw_treatment = f"raw_{treatment}"
            treatment_hat = f"mhat_{treatment}"
            observed = residual[[raw_treatment, treatment_hat]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
            if observed.sum() < 100:
                continue
            training_sd = float(pd.to_numeric(gs_residual[residual_treatment], errors="coerce").std(ddof=1))
            t = pd.to_numeric(residual.loc[observed, raw_treatment], errors="coerce").to_numpy(dtype=float)
            t_hat = pd.to_numeric(residual.loc[observed, treatment_hat], errors="coerce").to_numpy(dtype=float)
            try:
                weights, diagnostics = _stabilized_gps_weights(t, t_hat, training_sd, trim_quantile)
            except ValueError:
                continue
            subset = residual.loc[observed].copy()
            subset[treatment] = t
            for outcome_column in contract["outcomes"]:
                stage = outcome_column.replace("res_y_", "")
                outcome = pd.to_numeric(subset[f"raw_y_{stage}"], errors="coerce")
                model, index, covariance = _clustered_linear(
                    subset,
                    outcome,
                    subset[[treatment]],
                    weights,
                )
                rows.extend(
                    _record_rows(
                        dataset,
                        "Marginal GPS-IPW",
                        resolution,
                        "Marginal_single_treatment",
                        stage,
                        [treatment],
                        model,
                        index,
                        covariance,
                        len(residual),
                        {
                            "Estimator_Contract": "marginal continuous-treatment GPS diagnostic; not a joint 6/10 estimator",
                            **diagnostics,
                        },
                    )
                )

    result = holm_column(
        pd.DataFrame(rows),
        ["Dataset", "Estimator", "Resolution", "Model_Specification", "Stage"],
    )
    result["Temporal_Analysis_Population"] = contract["population"]
    output = ensure_dir(_dcml_paths.workspace_path(root, 'processed_data') / dataset / "Final_Causal_Output")
    result_path = output / f"{dataset}_Causal_Baseline_Comparison_{OUTPUT_VERSION}.csv"
    schema_path = output / f"{dataset}_Causal_Baseline_Schema_{OUTPUT_VERSION}.json"
    result.to_csv(result_path, index=False)
    with open(schema_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset,
                "script_version": OUTPUT_VERSION,
                "source_data_version": SOURCE_VERSION,
                "aggregate_treatments": specifications["Aggregate"],
                "disaggregate_treatments": specifications["Disaggregate"],
                "estimators": {
                    "Naive OLS": "joint association benchmark",
                    "Standard DML (No GS)": "joint residual-on-residual benchmark using frozen nuisance fits",
                    "Marginal GPS-IPW": "single-treatment diagnostic; not directly comparable to a joint coefficient vector",
                },
                "specification_policy": (
                    "6/10 models include T_int_sem and target history-eligible complete cases; "
                    "5/9 models exclude T_int_sem and target the full temporally certified population."
                ),
                "claim_boundary": contract["claim"],
                "result_file": str(result_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Completed causal baselines for %s in %.2f seconds", dataset, time.time() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--trim-quantile", type=float, default=0.995)
    args = parser.parse_args()
    root = find_causal_root(__file__)
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        run_dataset(root, dataset, args.trim_quantile)


if __name__ == "__main__":
    main()
