#!/usr/bin/env python3
"""Audit MCRE measurement quality and empirical positivity for DML.

The item-level diagnostics describe calibration and marginal variation. The
residual diagnostics address the observable implications of positivity for the
partially linear, multiple-continuous-treatment DML specification: treatments
must retain variation after conditioning on pre-event controls, and the joint
residual second-moment matrix must be nonsingular. These diagnostics can detect
empirical support failures; no finite observational diagnostic can prove the
positivity or unconfoundedness assumptions.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "20260722"

DATASETS = {
    "suning": {
        "item": "suning/item_multimodal_scalars/item_multimodal_scalars_full_3.parquet",
        "residual": "suning/DML_Results/DCML_Residuals_20260717.parquet",
        "schema": "suning/DML_Results/DML_feature_schema_20260717.json",
        "manifest": "suning/Final_Causal_Output/suning_GS_projection_manifest_20260717.json",
    },
    "amazon_appliances": {
        "item": (
            "amazon_appliances/item_multimodal_scalars/"
            "amazon_appliances_item_multimodal_scalars_merged_full_2.parquet"
        ),
        "residual": (
            "amazon_appliances/DML_Results/"
            "amazon_appliances_DML_Residuals_Final_20260717.parquet"
        ),
        "schema": (
            "amazon_appliances/DML_Results/"
            "amazon_appliances_DML_feature_schema_20260717.json"
        ),
        "manifest": (
            "amazon_appliances/Final_Causal_Output/"
            "amazon_appliances_GS_projection_manifest_20260717.json"
        ),
    },
    "amazon_beauty": {
        "item": (
            "amazon_beauty/item_multimodal_scalars/"
            "amazon_beauty_item_multimodal_scalars_merged_full.parquet"
        ),
        "residual": (
            "amazon_beauty/DML_Results/"
            "amazon_beauty_DML_Residuals_Final_20260717.parquet"
        ),
        "schema": (
            "amazon_beauty/DML_Results/"
            "amazon_beauty_DML_feature_schema_20260717.json"
        ),
        "manifest": (
            "amazon_beauty/Final_Causal_Output/"
            "amazon_beauty_GS_projection_manifest_20260717.json"
        ),
    },
}

ITEM_COMPONENTS = {
    "T_con_mkt": ("MNC", "parent"),
    "T_int_vis": ("MAI", "parent"),
    "T_int_fac": ("MAI", "standalone"),
    "atomic_a_pri": ("MNC", "atomic"),
    "atomic_a_gft": ("MNC", "atomic"),
    "atomic_a_sub": ("MNC", "atomic"),
    "atomic_a_urg": ("MNC", "atomic"),
    "atomic_a_spec": ("MAI", "atomic"),
    "atomic_a_str": ("MAI", "atomic"),
}

GLOBAL_SUPPORT = {
    "min_n": 500,
    "min_sd": 0.02,
    "min_iqr": 0.02,
    "min_effective_bins": 4,
}

STRATUM_SUPPORT = {
    "min_n": 100,
    "min_iqr": 0.01,
}


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def iqr(series: pd.Series) -> float:
    values = numeric(series).dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(0.75) - values.quantile(0.25))


def percentile_width(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> float:
    values = numeric(series).dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(upper) - values.quantile(lower))


def effective_bins(series: pd.Series, bins: int = 10) -> tuple[int, int, float]:
    values = numeric(series).dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return 0, 0, np.nan
    if np.nanmin(values) == np.nanmax(values):
        return 1, len(values), 1.0
    counts, _ = np.histogram(values, bins=bins)
    occupied = counts[counts > 0]
    return (
        int(len(occupied)),
        int(occupied.min()) if len(occupied) else 0,
        float(counts.max() / len(values)),
    )


def calibrated_column(columns: set[str], raw: str) -> str:
    for candidate in (f"{raw}_calib", f"{raw}_calibrated"):
        if candidate in columns:
            return candidate
    raise KeyError(f"No calibrated column found for {raw}")


def item_measurement_diagnostics(dataset: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    columns = set(frame.columns)
    rows: list[dict] = []
    for component, (pathway, level) in ITEM_COMPONENTS.items():
        if component not in columns:
            raise KeyError(f"{path}: missing raw MCRE component {component}")
        calibrated = calibrated_column(columns, component)
        raw = numeric(frame[component])
        calibrated_values = numeric(frame[calibrated])
        observed = raw.notna() & calibrated_values.notna()
        raw = raw.loc[observed]
        calibrated_values = calibrated_values.loc[observed]
        positive = raw.gt(0)
        positive_raw = raw.loc[positive]
        positive_calibrated = calibrated_values.loc[positive]
        marginal_flag = (
            len(raw) >= GLOBAL_SUPPORT["min_n"]
            and positive_raw.nunique() >= GLOBAL_SUPPORT["min_effective_bins"]
            and iqr(positive_calibrated) > 0
        )
        rows.append(
            {
                "Dataset": dataset,
                "Component": component,
                "Pathway": pathway,
                "Measurement_Level": level,
                "Observed_Items": int(len(raw)),
                "Zero_Items": int(raw.eq(0).sum()),
                "Zero_Share": float(raw.eq(0).mean()) if len(raw) else np.nan,
                "Positive_Items": int(positive.sum()),
                "Positive_Unique_Raw_Values": int(positive_raw.nunique()),
                "Positive_Raw_IQR": iqr(positive_raw),
                "Positive_Raw_P01_P99_Width": percentile_width(positive_raw),
                "Positive_Calibrated_IQR": iqr(positive_calibrated),
                "Positive_Calibrated_P01_P99_Width": percentile_width(positive_calibrated),
                "Marginal_Measurement_Support_Flag": "PASS" if marginal_flag else "REVIEW",
                "Interpretation": "calibration_and_marginal_support_not_causal_positivity",
            }
        )
    return pd.DataFrame(rows)


def global_residual_diagnostic(
    dataset: str,
    treatment: str,
    raw: pd.Series,
    mhat: pd.Series,
    residual: pd.Series,
) -> dict:
    observed = raw.notna() & mhat.notna() & residual.notna()
    raw = raw.loc[observed]
    mhat = mhat.loc[observed]
    residual = residual.loc[observed]
    bins, min_bin, max_share = effective_bins(residual)
    residual_sd = float(residual.std(ddof=1)) if len(residual) > 1 else 0.0
    residual_iqr = iqr(residual)
    reasons = []
    if len(residual) < GLOBAL_SUPPORT["min_n"]:
        reasons.append("small_n")
    if residual_sd < GLOBAL_SUPPORT["min_sd"]:
        reasons.append("low_residual_sd")
    if residual_iqr < GLOBAL_SUPPORT["min_iqr"]:
        reasons.append("low_residual_iqr")
    if bins < GLOBAL_SUPPORT["min_effective_bins"]:
        reasons.append("sparse_effective_bins")
    return {
        "Dataset": dataset,
        "Treatment": treatment,
        "Observed_N": int(len(residual)),
        "Raw_Zero_Share": float(raw.eq(0).mean()) if len(raw) else np.nan,
        "Mhat_SD": float(mhat.std(ddof=1)) if len(mhat) > 1 else 0.0,
        "Residual_Mean": float(residual.mean()) if len(residual) else np.nan,
        "Residual_SD": residual_sd,
        "Residual_IQR": residual_iqr,
        "Residual_P01_P99_Width": percentile_width(residual),
        "Effective_Bins_10": bins,
        "Minimum_Bin_Count_10": min_bin,
        "Maximum_Bin_Share_10": max_share,
        "Global_Residual_Support_Flag": "PASS" if not reasons else "REVIEW",
        "Empirical_Positivity_Diagnostic": "PASS" if not reasons else "REVIEW",
        "Support_Reason": "ok" if not reasons else "|".join(reasons),
        "Interpretation": "observable_conditional_support_diagnostic_not_assumption_proof",
    }


def residual_support_diagnostics(
    dataset: str,
    residual_path: Path,
    treatments: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = []
    for treatment in treatments:
        required.extend([f"raw_{treatment}", f"mhat_{treatment}", f"res_{treatment}"])
    frame = pd.read_parquet(residual_path, columns=required)
    global_rows: list[dict] = []
    stratum_rows: list[dict] = []
    for treatment in treatments:
        raw = numeric(frame[f"raw_{treatment}"])
        mhat = numeric(frame[f"mhat_{treatment}"])
        residual = numeric(frame[f"res_{treatment}"])
        global_rows.append(global_residual_diagnostic(dataset, treatment, raw, mhat, residual))

        observed = raw.notna() & mhat.notna() & residual.notna()
        treatment_frame = pd.DataFrame(
            {
                "raw": raw.loc[observed],
                "mhat": mhat.loc[observed],
                "residual": residual.loc[observed],
            }
        )
        if treatment_frame.empty:
            continue
        try:
            treatment_frame["mhat_stratum"] = pd.qcut(
                treatment_frame["mhat"], q=10, duplicates="drop"
            )
        except ValueError:
            treatment_frame["mhat_stratum"] = "all"
        grouped = treatment_frame.groupby("mhat_stratum", observed=True, sort=True)
        for index, (_, group) in enumerate(grouped, start=1):
            group_iqr = iqr(group["residual"])
            diagnostic_pass = (
                len(group) >= STRATUM_SUPPORT["min_n"]
                and np.isfinite(group_iqr)
                and group_iqr >= STRATUM_SUPPORT["min_iqr"]
            )
            stratum_rows.append(
                {
                    "Dataset": dataset,
                    "Treatment": treatment,
                    "Mhat_Stratum": index,
                    "N": int(len(group)),
                    "Mhat_Min": float(group["mhat"].min()),
                    "Mhat_Max": float(group["mhat"].max()),
                    "Raw_Zero_Share": float(group["raw"].eq(0).mean()),
                    "Raw_P01": float(group["raw"].quantile(0.01)),
                    "Raw_P99": float(group["raw"].quantile(0.99)),
                    "Residual_Mean": float(group["residual"].mean()),
                    "Residual_SD": float(group["residual"].std(ddof=1)),
                    "Residual_IQR": group_iqr,
                    "Residual_P01_P99_Width": percentile_width(group["residual"]),
                    "Conditional_Variation_Diagnostic": "PASS" if diagnostic_pass else "REVIEW",
                }
            )
    global_frame = pd.DataFrame(global_rows)
    stratum_frame = pd.DataFrame(stratum_rows)
    summaries: list[dict] = []
    for treatment, group in stratum_frame.groupby("Treatment", sort=False):
        summaries.append(
            {
                "Dataset": dataset,
                "Treatment": treatment,
                "Mhat_Strata": int(len(group)),
                "Minimum_Stratum_N": int(group["N"].min()),
                "Minimum_Stratum_Residual_IQR": float(group["Residual_IQR"].min()),
                "Median_Stratum_Residual_IQR": float(group["Residual_IQR"].median()),
                "Supported_Strata": int(group["Conditional_Variation_Diagnostic"].eq("PASS").sum()),
                "Supported_Strata_Share": float(
                    group["Conditional_Variation_Diagnostic"].eq("PASS").mean()
                ),
                "All_Strata_Pass": bool(group["Conditional_Variation_Diagnostic"].eq("PASS").all()),
                "Empirical_Positivity_Diagnostic": (
                    "PASS"
                    if group["Conditional_Variation_Diagnostic"].eq("PASS").all()
                    else "REVIEW"
                ),
            }
        )
    return global_frame, stratum_frame, pd.DataFrame(summaries)


def apply_projection(frame: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    projected = frame.copy()
    for target in manifest.get("analytical_targets", []):
        projection = manifest.get("projections", {}).get(target)
        if not projection:
            continue
        parameters = projection["parameters"]
        fitted = np.full(len(frame), float(parameters.get("const", 0.0)))
        for control in manifest.get("heuristic_controls", []):
            fitted += numeric(frame[control]).fillna(0).to_numpy() * float(parameters.get(control, 0.0))
        projected[target] = numeric(frame[target]).to_numpy() - fitted
    return projected


def matrix_diagnostic(
    dataset: str,
    specification: str,
    projection_status: str,
    frame: pd.DataFrame,
    requested: list[str],
    eligible: list[str],
) -> dict:
    columns = [f"res_{treatment}" for treatment in eligible]
    complete = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if complete.empty or len(columns) < 2:
        return {
            "Dataset": dataset,
            "Specification": specification,
            "Projection_Status": projection_status,
            "Requested_Treatments": len(requested),
            "Included_Treatments": len(columns),
            "Excluded_Treatments": "|".join(t for t in requested if t not in eligible),
            "Complete_Case_N": int(len(complete)),
            "Matrix_Rank": 0,
            "Minimum_Correlation_Eigenvalue": np.nan,
            "Correlation_Condition_Number": np.nan,
            "Rank_Diagnostic": "NOT_ESTIMABLE",
        }
    covariance = complete.cov().to_numpy(dtype=float)
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    covariance_minimum = float(covariance_eigenvalues.min())
    covariance_maximum = float(covariance_eigenvalues.max())
    covariance_condition = (
        float(covariance_maximum / covariance_minimum)
        if covariance_minimum > 0
        else np.inf
    )
    standardized = (complete - complete.mean()) / complete.std(ddof=1).replace(0, np.nan)
    correlation = standardized.corr().to_numpy(dtype=float)
    eigenvalues = np.linalg.eigvalsh(correlation)
    minimum = float(eigenvalues.min())
    maximum = float(eigenvalues.max())
    rank = int(np.linalg.matrix_rank(correlation, tol=1e-10))
    condition = float(maximum / minimum) if minimum > 0 else np.inf
    return {
        "Dataset": dataset,
        "Specification": specification,
        "Projection_Status": projection_status,
        "Requested_Treatments": len(requested),
        "Included_Treatments": len(columns),
        "Excluded_Treatments": "|".join(t for t in requested if t not in eligible),
        "Complete_Case_N": int(len(complete)),
        "Matrix_Rank": rank,
        "Minimum_Residual_Covariance_Eigenvalue": covariance_minimum,
        "Residual_Covariance_Condition_Number": covariance_condition,
        "Minimum_Correlation_Eigenvalue": minimum,
        "Maximum_Correlation_Eigenvalue": maximum,
        "Correlation_Condition_Number": condition,
        "Rank_Diagnostic": (
            "PASS"
            if rank == len(columns) and covariance_minimum > 0 and minimum > 1e-8
            else "REVIEW"
        ),
    }


def joint_rank_diagnostics(
    dataset: str,
    residual_path: Path,
    schema: dict,
    manifest: dict,
    global_support: pd.DataFrame,
) -> pd.DataFrame:
    support_map = global_support.set_index("Treatment")["Global_Residual_Support_Flag"].to_dict()
    specifications = {
        "macro_cue": schema["aggregate_treatments"],
        "fine_grained_cue": schema["disaggregate_treatments"],
    }
    needed = sorted({f"res_{t}" for values in specifications.values() for t in values})
    frame = pd.read_parquet(residual_path, columns=needed)
    rows: list[dict] = []
    manifest_keys = {"macro_cue": "Aggregate", "fine_grained_cue": "Disaggregate"}
    for specification, requested in specifications.items():
        eligible = [t for t in requested if support_map.get(t) == "PASS"]
        rows.append(matrix_diagnostic(dataset, specification, "before_projection", frame, requested, eligible))
        projected = apply_projection(frame, manifest[manifest_keys[specification]])
        rows.append(matrix_diagnostic(dataset, specification, "frozen_projection", projected, requested, eligible))
    return pd.DataFrame(rows)


def write_outputs(
    processed_root: Path,
    dataset: str,
    marginal: pd.DataFrame | None,
    global_support: pd.DataFrame,
    strata: pd.DataFrame,
    stratum_summary: pd.DataFrame,
    rank: pd.DataFrame,
) -> None:
    output = processed_root / dataset / "audit"
    output.mkdir(parents=True, exist_ok=True)
    if marginal is not None:
        marginal.to_csv(output / f"{dataset}_MCRE_marginal_measurement_support_{VERSION}.csv", index=False)
    global_support.to_csv(output / f"{dataset}_MCRE_global_residual_support_{VERSION}.csv", index=False)
    strata.to_csv(output / f"{dataset}_MCRE_conditional_residual_support_{VERSION}.csv", index=False)
    stratum_summary.to_csv(
        output / f"{dataset}_MCRE_conditional_support_summary_{VERSION}.csv", index=False
    )
    rank.to_csv(output / f"{dataset}_MCRE_joint_rank_diagnostics_{VERSION}.csv", index=False)
    stratum_summary.to_csv(
        output / f"{dataset}_DML_empirical_positivity_diagnostics_{VERSION}.csv", index=False
    )
    rank.to_csv(
        output / f"{dataset}_DML_joint_residual_identification_{VERSION}.csv", index=False
    )
    summary = {
        "dataset": dataset,
        "version": VERSION,
        "marginal_measurement_diagnostics_available": marginal is not None,
        "global_residual_support_passed": int(
            global_support["Global_Residual_Support_Flag"].eq("PASS").sum()
        ),
        "global_residual_support_total": int(len(global_support)),
        "all_conditional_strata_pass_by_treatment": {
            row.Treatment: bool(row.All_Strata_Pass)
            for row in stratum_summary.itertuples(index=False)
        },
        "joint_rank_diagnostics": rank.to_dict(orient="records"),
        "interpretation_boundary": (
            "ECDF and zero shares assess calibration and marginal measurement support. "
            "Residual and rank diagnostics assess the observable conditional-variation and "
            "nonsingularity implications of positivity in the certified sample. Passing "
            "diagnostics does not prove positivity or unconfoundedness."
        ),
        "global_support_thresholds": GLOBAL_SUPPORT,
        "conditional_stratum_thresholds": STRATUM_SUPPORT,
    }
    def json_safe(value):
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    (output / f"{dataset}_MCRE_measurement_support_summary_{VERSION}.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=True, allow_nan=False)
    )


def run_dataset(processed_root: Path, dataset: str, skip_item_level: bool) -> None:
    config = DATASETS[dataset]
    schema = json.loads((processed_root / config["schema"]).read_text())
    manifest = json.loads((processed_root / config["manifest"]).read_text())
    treatments = list(schema["treatments"])
    marginal = None
    if not skip_item_level:
        marginal = item_measurement_diagnostics(dataset, processed_root / config["item"])
    global_support, strata, stratum_summary = residual_support_diagnostics(
        dataset, processed_root / config["residual"], treatments
    )
    rank = joint_rank_diagnostics(
        dataset,
        processed_root / config["residual"],
        schema,
        manifest,
        global_support,
    )
    write_outputs(
        processed_root,
        dataset,
        marginal,
        global_support,
        strata,
        stratum_summary,
        rank,
    )
    print(
        f"{dataset}: residual support "
        f"{global_support['Global_Residual_Support_Flag'].eq('PASS').sum()}/{len(global_support)}; "
        f"rank diagnostics {rank['Rank_Diagnostic'].eq('PASS').sum()}/{len(rank)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(str(_dcml_paths.project_path('processed_data'))),
    )
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument(
        "--skip-item-level",
        action="store_true",
        help="Run only residual and rank diagnostics when full item-level MCRE files are unavailable.",
    )
    args = parser.parse_args()
    selected = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for dataset in selected:
        run_dataset(args.processed_root, dataset, args.skip_item_level)


if __name__ == "__main__":
    main()
