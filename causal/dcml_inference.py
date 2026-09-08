#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared 20260717 final-stage inference with training-split GS projections."""

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

from dcml_utils import add_holm_by_family, ensure_dir, read_table, setup_logging


VERSION = "20260717"


def clean_component_name(column: str) -> str:
    name = column.replace("res_", "").replace("__reverse_gs_pure", "").replace("__gs_pure", "")
    name = name.replace("_calibrated", "").replace("_calib", "").replace("atomic_", "")
    return "a_pri" if name == "mkt_price" else name


def numeric_frame(df: pd.DataFrame, columns: list[str], fill: bool = False) -> pd.DataFrame:
    out = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0) if fill else out


def projection_design(df: pd.DataFrame, controls: list[str], mode: str) -> pd.DataFrame:
    x = numeric_frame(df, controls, fill=True)
    if mode == "linear_quadratic":
        for col in controls:
            x[f"{col}__sq"] = x[col] ** 2
        for i, left in enumerate(controls):
            for right in controls[i + 1 :]:
                x[f"{left}__x__{right}"] = x[left] * x[right]
    return sm.add_constant(x, has_constant="add")


def fit_apply_gs(
    gs_train: pd.DataFrame,
    final: pd.DataFrame,
    treatment_names: list[str],
    resolution: str,
    mode: str,
) -> tuple[pd.DataFrame, list[str], list[dict], dict]:
    if resolution == "Aggregate":
        heuristic_names = treatment_names[:3]
        analytical_names = treatment_names[3:]
    elif resolution == "Disaggregate":
        heuristic_names = treatment_names[:6]
        analytical_names = treatment_names[6:]
    else:
        raise ValueError(resolution)
    controls = [f"res_{name}" for name in heuristic_names]
    targets = [f"res_{name}" for name in analytical_names]
    for column in controls + targets:
        if column not in gs_train.columns or column not in final.columns:
            raise RuntimeError(f"Missing {resolution} residual column: {column}")

    out = final.copy()
    train_x = projection_design(gs_train, controls, mode)
    final_x = projection_design(final, controls, mode).reindex(columns=train_x.columns, fill_value=0.0)
    pure_columns: list[str] = []
    diagnostics: list[dict] = []
    parameter_manifest: dict = {
        "resolution": resolution,
        "mode": mode,
        "heuristic_controls": controls,
        "analytical_targets": targets,
        "projections": {},
    }
    for target in targets:
        train_y = pd.to_numeric(gs_train[target], errors="coerce")
        observed = train_y.notna()
        if observed.sum() <= train_x.shape[1] + 5:
            pure = f"{target}__gs_pure"
            out[pure] = np.nan
            pure_columns.append(pure)
            diagnostics.append(
                {
                    "Resolution": resolution,
                    "Target": target,
                    "Train_N": int(observed.sum()),
                    "Projection_R2": np.nan,
                    "Status": "insufficient_training_support",
                }
            )
            continue
        model = sm.OLS(train_y.loc[observed], train_x.loc[observed]).fit()
        prediction = np.asarray(final_x @ model.params, dtype=float)
        final_y = pd.to_numeric(final[target], errors="coerce").to_numpy(dtype=float)
        pure = f"{target}__gs_pure"
        out[pure] = np.where(np.isfinite(final_y), final_y - prediction, np.nan)
        pure_columns.append(pure)
        diagnostics.append(
            {
                "Resolution": resolution,
                "Target": target,
                "Train_N": int(observed.sum()),
                "Projection_R2": float(model.rsquared),
                "Status": "fitted_on_A_to_B_residuals",
            }
        )
        parameter_manifest["projections"][target] = {
            "design_columns": list(train_x.columns),
            "parameters": {key: float(value) for key, value in model.params.items()},
            "train_n": int(observed.sum()),
            "r2": float(model.rsquared),
        }
    return out, controls + pure_columns, diagnostics, parameter_manifest


def density_stats(values: np.ndarray, bins: int = 10) -> tuple[int, int, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0, 0, np.nan
    counts, _ = np.histogram(finite, bins=bins)
    nonzero = counts[counts > 0]
    return len(nonzero), int(nonzero.min()) if len(nonzero) else 0, float(counts.max() / len(finite))


def support_diagnostics(
    df: pd.DataFrame,
    treatment: str,
    min_n: int = 500,
    min_sd: float = 0.02,
    min_iqr: float = 0.02,
    min_bins: int = 4,
) -> dict:
    values = pd.to_numeric(df[treatment], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {
            "Estimable": False,
            "Support_Flag": "empty_treatment_residual",
            "Support_N": 0,
            "Residual_SD_Final": np.nan,
            "Residual_IQR_Final": np.nan,
            "Residual_Support_1_99_Final": np.nan,
            "Effective_Bins_10_Final": 0,
            "Min_Bin_Count_10_Final": 0,
            "Max_Bin_Share_10_Final": np.nan,
        }
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
    width = float(np.percentile(values, 99) - np.percentile(values, 1))
    effective_bins, min_bin, max_share = density_stats(values)
    reasons = []
    if len(values) < min_n:
        reasons.append("small_n")
    if sd < min_sd:
        reasons.append("low_residual_sd")
    if iqr < min_iqr:
        reasons.append("low_residual_iqr")
    if effective_bins < min_bins:
        reasons.append("sparse_effective_bins")
    return {
        "Estimable": not reasons,
        "Support_Flag": "ok" if not reasons else "|".join(reasons),
        "Support_N": int(len(values)),
        "Residual_SD_Final": sd,
        "Residual_IQR_Final": iqr,
        "Residual_Support_1_99_Final": width,
        "Effective_Bins_10_Final": int(effective_bins),
        "Min_Bin_Count_10_Final": min_bin,
        "Max_Bin_Share_10_Final": max_share,
    }


def fit_clustered_ols(df: pd.DataFrame, outcome: str, treatments: list[str]):
    columns = [outcome] + treatments
    complete = numeric_frame(df, columns).dropna()
    if len(complete) <= len(treatments) + 5:
        raise RuntimeError("Insufficient complete cases for final regression")
    groups = df.loc[complete.index, ["cluster_user_id", "cluster_item_id"]]
    x = sm.add_constant(complete[treatments], has_constant="add")
    base = sm.OLS(complete[outcome], x)
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
    return model, complete.index, covariance


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    outcome: str,
    treatments: list[str],
    reps: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if reps <= 0:
        return {col: (np.nan, np.nan) for col in treatments}
    columns = [outcome] + treatments
    complete = numeric_frame(df, columns).dropna()
    if complete.empty:
        return {col: (np.nan, np.nan) for col in treatments}
    user = df.loc[complete.index, "cluster_user_id"].astype(str)
    groups = {key: np.asarray(indexes, dtype=int) for key, indexes in user.groupby(user).indices.items()}
    keys = np.asarray(list(groups))
    if len(keys) < 2:
        return {col: (np.nan, np.nan) for col in treatments}
    x = np.column_stack([np.ones(len(complete)), complete[treatments].to_numpy(dtype=float)])
    y = complete[outcome].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(reps):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[key] for key in sampled])
        try:
            estimates.append((np.linalg.pinv(x[idx].T @ x[idx]) @ x[idx].T @ y[idx])[1:])
        except np.linalg.LinAlgError:
            continue
    if len(estimates) < max(20, reps // 5):
        return {col: (np.nan, np.nan) for col in treatments}
    values = np.vstack(estimates)
    return {
        treatment: (
            float(np.percentile(values[:, index], 2.5)),
            float(np.percentile(values[:, index], 97.5)),
        )
        for index, treatment in enumerate(treatments)
    }


def run_family(
    df: pd.DataFrame,
    outcome_columns: list[str],
    treatments: list[str],
    resolution: str,
    h_level,
    bootstrap_reps: int,
    seed: int,
    gs_mode: str,
    claim_boundary: str,
    temporal_analysis_population: str = "temporally certified items only",
) -> list[dict]:
    diagnostics = {column: support_diagnostics(df, column) for column in treatments}
    estimable = [column for column in treatments if diagnostics[column]["Estimable"]]
    skipped = [column for column in treatments if not diagnostics[column]["Estimable"]]
    rows: list[dict] = []
    for outcome in outcome_columns:
        stage = outcome.replace("res_y_", "")
        for treatment in skipped:
            rows.append(
                {
                    "Resolution": resolution,
                    "H_level": h_level,
                    "Stage": stage,
                    "Component": clean_component_name(treatment),
                    "Treatment_Column": treatment,
                    "Coefficient": np.nan,
                    "Std_Error": np.nan,
                    "CI_Lower_95": np.nan,
                    "CI_Upper_95": np.nan,
                    "P_Value": np.nan,
                    "T_Stat": np.nan,
                    "N": 0,
                    "Covariance": "not_estimated",
                    "Bootstrap_CI_Lower_95": np.nan,
                    "Bootstrap_CI_Upper_95": np.nan,
                    "Orthogonalization_Mode": gs_mode,
                    "GS_Fit_Split": "Set_B A-to-B treatment residuals",
                    "Analysis_Sample": "support-gated complete cases",
                    "Temporal_Analysis_Population": temporal_analysis_population,
                    "Claim_Boundary": claim_boundary,
                    **diagnostics[treatment],
                }
            )
        if not estimable:
            continue
        model, model_index, covariance = fit_clustered_ols(df, outcome, estimable)
        model_df = df.loc[model_index]
        bootstrap = cluster_bootstrap_ci(model_df, outcome, estimable, bootstrap_reps, seed)
        semantic_in_model = any(clean_component_name(col) == "T_int_sem" for col in estimable)
        analysis_sample = "pre-purchase semantic-history complete cases" if semantic_in_model else "support-gated complete cases"
        for treatment in estimable:
            coefficient = float(model.params[treatment])
            standard_error = float(model.bse[treatment])
            rows.append(
                {
                    "Resolution": resolution,
                    "H_level": h_level,
                    "Stage": stage,
                    "Component": clean_component_name(treatment),
                    "Treatment_Column": treatment,
                    "Coefficient": coefficient,
                    "Std_Error": standard_error,
                    "CI_Lower_95": coefficient - 1.96 * standard_error,
                    "CI_Upper_95": coefficient + 1.96 * standard_error,
                    "P_Value": float(model.pvalues[treatment]),
                    "T_Stat": float(model.tvalues[treatment]),
                    "N": int(len(model_index)),
                    "Covariance": covariance,
                    "Bootstrap_CI_Lower_95": bootstrap[treatment][0],
                    "Bootstrap_CI_Upper_95": bootstrap[treatment][1],
                    "Orthogonalization_Mode": gs_mode,
                    "GS_Fit_Split": "Set_B A-to-B treatment residuals",
                    "Analysis_Sample": analysis_sample,
                    "Temporal_Analysis_Population": temporal_analysis_population,
                    "Claim_Boundary": claim_boundary,
                    **diagnostics[treatment],
                }
            )
    return rows


def inference_contract(root: Path, dataset: str) -> dict:
    base = _dcml_paths.workspace_path(root, 'processed_data') / dataset
    if dataset == "suning":
        return {
            "schema": base / "DML_Results" / f"DML_feature_schema_{VERSION}.json",
            "final_residual": base / "DML_Results" / f"DCML_Residuals_{VERSION}.parquet",
            "gs_residual": base / "DML_Results" / f"DCML_GS_Training_Residuals_{VERSION}.parquet",
            "outcomes": ["res_y_click", "res_y_cart", "res_y_purchase"],
            "ate": f"Final_ATE_Results_{VERSION}.csv",
            "cate": f"Final_CATE_Results_{VERSION}.csv",
            "all": f"Final_ATE_CATE_Results_{VERSION}.csv",
            "claim": "Suning observed three-stage funnel among temporally certified items; stage contrasts require the separate H4 formal test.",
            "population": "items with all selected MCRE reviews strictly before item-specific Set C entry",
        }
    prefix = f"{dataset}_DCML"
    return {
        "schema": base / "DML_Results" / f"{dataset}_DML_feature_schema_{VERSION}.json",
        "final_residual": base / "DML_Results" / f"{dataset}_DML_Residuals_Final_{VERSION}.parquet",
        "gs_residual": base / "DML_Results" / f"{dataset}_DML_GS_Training_Residuals_{VERSION}.parquet",
        "outcomes": ["res_y_purchase"],
        "ate": f"{dataset}_Final_ATE_{VERSION}.csv",
        "cate": f"{dataset}_Final_CATE_{VERSION}.csv",
        "all": f"{dataset}_Final_ATE_CATE_{VERSION}.csv",
        "claim": "Amazon purchase-stage sampled-choice external validation among temporally certified items; no native Click/Cart, full-funnel reversal, population checkout conversion, or policy/OPE claim.",
        "population": "items with all selected MCRE reviews strictly before item-specific Set C entry",
    }


def prepare_orthogonalized_data(
    root: Path,
    dataset: str,
    gs_mode: str,
) -> tuple[pd.DataFrame, dict[str, list[str]], list[dict], dict]:
    contract = inference_contract(root, dataset)
    with open(contract["schema"], "r", encoding="utf-8") as f:
        schema = json.load(f)
    final = read_table(contract["final_residual"])
    gs_train = read_table(contract["gs_residual"])
    aggregate_names = list(schema["aggregate_treatments"])
    disaggregate_names = list(schema["disaggregate_treatments"])
    final, aggregate_columns, aggregate_diagnostics, aggregate_manifest = fit_apply_gs(
        gs_train,
        final,
        aggregate_names,
        "Aggregate",
        gs_mode,
    )
    final, disaggregate_columns, disaggregate_diagnostics, disaggregate_manifest = fit_apply_gs(
        gs_train,
        final,
        disaggregate_names,
        "Disaggregate",
        gs_mode,
    )
    specs = {"Aggregate": aggregate_columns, "Disaggregate": disaggregate_columns}
    diagnostics = aggregate_diagnostics + disaggregate_diagnostics
    manifest = {
        "dataset": dataset,
        "script_version": VERSION,
        "fit_sample": "Set B treatment residuals generated out of sample from Set A nuisance models",
        "apply_sample": "Set C final residuals generated out of sample from Set B nuisance models",
        "analysis_population": schema.get("analysis_population", contract["population"]),
        "analysis_temporal_status": schema.get("analysis_temporal_status"),
        "temporal_certification_manifest": schema.get("temporal_certification_manifest"),
        "Aggregate": aggregate_manifest,
        "Disaggregate": disaggregate_manifest,
    }
    return final, specs, diagnostics, manifest


def run_final_inference(root: Path, dataset: str, bootstrap_reps: int, gs_mode: str) -> None:
    contract = inference_contract(root, dataset)
    logger = setup_logging(ensure_dir(_dcml_paths.workspace_path(root, 'log')), f"dcml_step2_{dataset}_{VERSION}")
    start = time.time()
    final, specs, gs_diagnostics, gs_manifest = prepare_orthogonalized_data(root, dataset, gs_mode)
    rows: list[dict] = []
    for resolution, treatments in specs.items():
        rows.extend(
            run_family(
                final,
                contract["outcomes"],
                treatments,
                resolution,
                "All (ATE)",
                bootstrap_reps,
                20260717,
                gs_mode,
                contract["claim"],
                contract["population"],
            )
        )
        for h_level in sorted(pd.to_numeric(final["H_level"], errors="coerce").dropna().unique()):
            subset = final[pd.to_numeric(final["H_level"], errors="coerce") == h_level]
            rows.extend(
                run_family(
                    subset,
                    contract["outcomes"],
                    treatments,
                    resolution,
                    int(h_level),
                    bootstrap_reps,
                    20260720 + int(h_level),
                    gs_mode,
                    contract["claim"],
                    contract["population"],
                )
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"No final estimates produced for {dataset}")
    result = add_holm_by_family(result, ["Resolution", "H_level", "Stage"])
    save_dir = ensure_dir(_dcml_paths.workspace_path(root, 'processed_data') / dataset / "Final_Causal_Output")
    result[result["H_level"] == "All (ATE)"].to_csv(save_dir / contract["ate"], index=False)
    result[result["H_level"] != "All (ATE)"].to_csv(save_dir / contract["cate"], index=False)
    result.to_csv(save_dir / contract["all"], index=False)
    pd.DataFrame(gs_diagnostics).to_csv(save_dir / f"{dataset}_GS_training_diagnostics_{VERSION}.csv", index=False)
    with open(save_dir / f"{dataset}_GS_projection_manifest_{VERSION}.json", "w", encoding="utf-8") as f:
        json.dump(gs_manifest, f, ensure_ascii=False, indent=2)
    logger.info("Completed %s final inference in %.2f seconds", dataset, time.time() - start)
