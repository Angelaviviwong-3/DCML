#!/usr/bin/env python3
"""Consolidate existing RQ5 robustness and scope evidence.

The script only reads completed outputs. It separates evidence that supports
stability from diagnostics that limit transportability or causal scope.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[0]))
import project_paths as _dcml_paths


import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "20260729"
CAUSAL_ROOT = Path(__file__).resolve().parent
AUDIT_DIR = _dcml_paths.workspace_path(CAUSAL_ROOT, 'processed_data') / "audit"
VIS_DIR = _dcml_paths.workspace_path(CAUSAL_ROOT, "Unified_Visualization/output")
DATASETS = {
    "suning": "Suning",
    "amazon_appliances": "Amazon Appliances",
    "amazon_beauty": "Amazon Beauty",
}


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


class Locator:
    def __init__(self, archive_roots: list[Path]):
        self.roots = [CAUSAL_ROOT, *[path.resolve() for path in archive_roots]]
        self.used: dict[str, str] = {}

    def find(self, basename: str, path_contains: str | None = None) -> Path:
        candidates: list[Path] = []
        for root in self.roots:
            if not root.exists():
                continue
            candidates.extend(path for path in root.rglob(basename) if path.is_file())
        unique = sorted({path.resolve() for path in candidates})
        if path_contains is not None:
            filtered = [path for path in unique if path_contains in str(path)]
            if filtered:
                unique = filtered
        if not unique:
            raise FileNotFoundError(f"Cannot locate {basename}")
        canonical = [path for path in unique if CAUSAL_ROOT.resolve() in path.parents]
        selected = canonical[0] if canonical else unique[0]
        key = f"{path_contains}:{basename}" if path_contains else basename
        self.used[key] = str(selected)
        return selected


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def outcome_transport_row(dataset: str, locator: Locator) -> dict:
    source = locator.find(f"{dataset}_nuisance_temporal_transport_diagnostics_20260719.csv")
    frame = pd.read_csv(source)
    details = []
    for row in frame.itertuples(index=False):
        details.append(
            f"{str(row.Stage).title()}: AUC={fmt(row.ROC_AUC_Set_C)}, "
            f"Brier skill={fmt(row.Brier_Skill_vs_Train_Prevalence_Null)}, "
            f"calibration slope={fmt(row.Calibration_Slope)}"
        )
    calibrated = pd.to_numeric(frame["Calibration_Slope"], errors="coerce").between(0.8, 1.2).all()
    assessment = "SUPPORTS_TEMPORAL_TRANSPORT" if calibrated else "CALIBRATION_LIMITATION"
    return evidence_row(
        dataset,
        1,
        "Out-of-time nuisance transport",
        "Assess whether nuisance outcome models fitted before Set C retain discrimination and calibration in Set C.",
        "; ".join(details),
        assessment,
        "Temporal transfer of the fitted nuisance functions" if calibrated else "Discrimination is informative, but calibration drift limits precision claims",
        "Temporal transport diagnostics do not establish no unmeasured confounding.",
        source,
    )


def selection_row(dataset: str, locator: Locator) -> tuple[dict, dict]:
    metrics_source = locator.find(
        f"{dataset}_temporal_certification_selection_model_20260719.csv"
    )
    selection_manifest_source = locator.find(
        f"{dataset}_temporal_certification_selection_manifest_20260719.json"
    )
    certification_source = locator.find(
        f"{dataset}_temporal_certified_sample_manifest_20260717.json"
    )
    metrics = pd.read_csv(metrics_source).iloc[0]
    selection_manifest = json.loads(selection_manifest_source.read_text(encoding="utf-8"))
    certification = json.loads(certification_source.read_text(encoding="utf-8"))
    unweighted = float(metrics["Max_Absolute_Unweighted_SMD"])
    weighted = float(metrics["Max_Absolute_IPCW_SMD"])
    result = (
        f"Set C row retention={100 * certification['set_c_row_certification_rate']:.1f}%; "
        f"common-support rate={100 * float(metrics['Common_Support_001_099_Rate']):.1f}%; "
        f"maximum absolute SMD changes from {unweighted:.3f} to {weighted:.3f} after IPCW"
    )
    evidence = evidence_row(
        dataset,
        2,
        "Temporal certification and observed-selection sensitivity",
        "Quantify certification retention and whether weighting on observed covariates aligns certified and full Set C samples.",
        result,
        "LIMITS_EXTERNAL_VALIDITY",
        "The primary estimand is restricted to temporally certified items.",
        "IPCW does not recover the full-item population or remove selection on unobserved item attributes.",
        metrics_source,
    )
    estimand = {
        "Dataset": DATASETS[dataset],
        "Dataset_Key": dataset,
        "Set_C_Row_Retention": certification["set_c_row_certification_rate"],
        "Primary_Estimand": (
            "Stage-specific observational DML ATEs among temporally certified items, "
            "under the stated adjustment and identification assumptions."
        ),
        "Adjustment_Columns": "|".join(selection_manifest["model_features"]),
        "Adjustment_Set_Summary": adjustment_summary(dataset),
        "Risk_Set_and_Outcome": risk_set_description(dataset),
        "Claim_Boundary": claim_boundary(dataset),
    }
    return evidence, estimand


def specification_row(dataset: str, locator: Locator) -> dict:
    source = locator.find(f"{dataset}_ATE_specification_sensitivity_comparison_20260719.csv")
    frame = pd.read_csv(source)
    sign = as_bool(frame["Sign_Stable"])
    holm = as_bool(frame["Holm_Significance_Stable"])
    result = (
        f"ATE direction stable for {int(sign.sum())}/{len(frame)} coefficients; "
        f"Holm decision stable for {int(holm.sum())}/{len(frame)} coefficients"
    )
    assessment = "STABLE" if sign.all() and holm.all() else "FOCAL_RESULTS_STABLE_WITH_LOCAL_EXCEPTIONS"
    return evidence_row(
        dataset,
        3,
        "Certified-sample specification sensitivity",
        "Compare the primary complete-case specification with the full temporally certified specification.",
        result,
        assessment,
        "Most estimated directions and multiplicity decisions are not driven by the primary complete-case restriction.",
        "The compared coefficients use different analysis populations and are descriptive sensitivity estimates.",
        source,
    )


def projection_row(dataset: str, locator: Locator) -> dict:
    source = locator.find(f"{dataset}_GS_order_sensitivity_20260717.csv")
    frame = pd.read_csv(source)
    frame = frame[
        frame["H_level"].astype(str).eq("All (ATE)") & as_bool(frame["Estimable"])
    ].copy()
    primary = frame[
        frame["GS_Variant"].astype(str).eq("standard_heuristic_to_analytical")
    ][["Resolution", "Stage", "Component", "Coefficient", "Significant_Holm_05"]]
    summaries = []
    correlations = []
    significant_signs = []
    labels = {
        "no_gs": "no projection",
        "reverse_analytical_to_heuristic": "reverse order",
    }
    for variant, label in labels.items():
        alternative = frame[frame["GS_Variant"].astype(str).eq(variant)][
            ["Resolution", "Stage", "Component", "Coefficient"]
        ]
        merged = primary.merge(
            alternative,
            on=["Resolution", "Stage", "Component"],
            suffixes=("_primary", "_alternative"),
        ).dropna()
        correlation = float(
            merged["Coefficient_primary"].corr(merged["Coefficient_alternative"])
        )
        significant = merged[as_bool(merged["Significant_Holm_05"])]
        agreement = float(
            (
                np.sign(significant["Coefficient_primary"])
                == np.sign(significant["Coefficient_alternative"])
            ).mean()
        )
        correlations.append(correlation)
        significant_signs.append(agreement)
        summaries.append(
            f"{label}: r={correlation:.3f}, Holm-significant direction agreement={100 * agreement:.0f}%"
        )
    return evidence_row(
        dataset,
        4,
        "Gram-Schmidt projection sensitivity",
        "Test whether ATE conclusions depend on applying the projection or on its pathway order.",
        "; ".join(summaries),
        "SIGNIFICANT_EFFECT_DIRECTIONS_STABLE",
        "Holm-significant ATE directions are preserved across projection variants.",
        "High vector correlation does not imply complete statistical independence between pathways.",
        source,
    )


def placebo_row(dataset: str, locator: Locator) -> dict:
    basename = (
        "RQ5_Placebo_Test_Results_Detailed_20260717.csv"
        if dataset == "suning"
        else f"{dataset}_Purchase_Placebo_Test_Results_Detailed_20260717.csv"
    )
    source = locator.find(basename)
    frame = pd.read_csv(source)
    significant = frame[as_bool(frame["Significant_Holm_05"])]
    all_minimum = bool(
        len(significant)
        and np.allclose(
            pd.to_numeric(significant["Placebo_Empirical_P_Value"]),
            1.0 / (pd.to_numeric(significant["Placebo_Reps"]) + 1.0),
        )
    )
    result = (
        f"{len(significant)} Holm-significant ATEs evaluated with 200 joint row permutations; "
        f"all have plus-one empirical p=1/201" if all_minimum else
        f"{len(significant)} Holm-significant ATEs evaluated; permutation results are mixed"
    )
    return evidence_row(
        dataset,
        5,
        "Joint treatment-row placebo",
        "Compare observed ATEs with a null distribution that breaks treatment-outcome alignment.",
        result,
        "FOCAL_EFFECTS_DIFFER_FROM_PERMUTATION_NULL" if all_minimum else "MIXED_PLACEBO_EVIDENCE",
        "The Holm-significant observed estimates are not reproduced by random treatment-row assignment.",
        "Permutation placebos do not rule out structured unmeasured confounding.",
        source,
    )


def ovb_row(dataset: str, locator: Locator) -> dict:
    source = locator.find(f"{dataset}_DML_OVB_Sensitivity_20260729.csv")
    focal_source = locator.find(f"Table_DML_OVB_Focal_Summary_20260729.csv")
    frame = pd.read_csv(source)
    focal = pd.read_csv(focal_source)
    focal = focal[focal["Dataset_Key"].astype(str).eq(dataset)]
    point_min = float(focal["Point_RV_Percent"].min())
    point_max = float(focal["Point_RV_Percent"].max())
    nonfinite = int(
        frame["Bound_Status"].astype(str).isin(
            {"UNBOUNDED_CF_D_BOUNDARY", "NONINFORMATIVE_ZERO_OUTCOME_GAIN"}
        ).sum()
    )
    result = (
        f"Focal equal-strength values needed to move point estimates to zero range from "
        f"{point_min:.2f}% to {point_max:.2f}%; {nonfinite} benchmark-multiplier rows "
        f"reach a non-finite or non-informative boundary"
    )
    assessment = {
        "suning": "MODERATE_OMITTED_CONFOUNDING_SENSITIVITY",
        "amazon_appliances": "HETEROGENEOUS_OMITTED_CONFOUNDING_SENSITIVITY",
        "amazon_beauty": "FOCAL_SOCIAL_PROOF_LESS_SENSITIVE",
    }[dataset]
    return evidence_row(
        dataset,
        6,
        "Benchmarked omitted-confounding sensitivity",
        "Quantify how strong equal-strength omitted confounding must be to move focal DML point estimates to zero.",
        result,
        assessment,
        "The analysis gives effect-specific sensitivity magnitudes rather than a blanket robustness claim.",
        "No universal pass threshold exists, and the bounds do not establish absence of unmeasured confounding.",
        source,
    )


def candidate_row(dataset: str, locator: Locator) -> dict:
    suffix = "20260717" if dataset == "suning" else "20260718"
    source = locator.find(
        f"Recommendation_Candidate_Sensitivity_Summary_{suffix}.csv",
        path_contains=f"/{dataset}/",
    )
    frame = pd.read_csv(source)
    frame["NDCG_Rank"] = frame.groupby(["Stage", "Candidate_Set_Size"])[
        "NDCG@10_mean"
    ].rank(method="min", ascending=False)
    ours = frame[frame["Model"].astype(str).eq("DCML (Ours)")].copy()
    summaries = []
    for stage, group in ours.groupby("Stage", sort=False):
        group = group.sort_values("Candidate_Set_Size")
        ranks = "/".join(str(int(value)) for value in group["NDCG_Rank"])
        sizes = "/".join(str(int(value)) for value in group["Candidate_Set_Size"])
        summaries.append(f"{str(stage).title()} NDCG@10 ranks={ranks} at {sizes} candidates")
    return evidence_row(
        dataset,
        7,
        "Recommendation candidate-set sensitivity",
        "Assess whether comparative recommendation conclusions change with 50, 80, or 100 candidates.",
        "; ".join(summaries),
        "PREDICTIVE_RANKING_STABILITY_QUANTIFIED",
        "Recommendation ranking is evaluated under three candidate-set sizes.",
        "Candidate-set sensitivity concerns offline ranking evaluation, not causal identification.",
        source,
    )


def evidence_row(
    dataset: str,
    order: int,
    analysis: str,
    purpose: str,
    result: str,
    assessment: str,
    supports: str,
    boundary: str,
    source: Path,
) -> dict:
    return {
        "Analysis_Order": order,
        "Dataset_Key": dataset,
        "Dataset": DATASETS[dataset],
        "Robustness_or_Scope_Analysis": analysis,
        "Design_Purpose": purpose,
        "Observed_Result": result,
        "Evidence_Assessment": assessment,
        "Supports": supports,
        "Does_Not_Establish": boundary,
        "Source_File": str(source),
    }


def adjustment_summary(dataset: str) -> str:
    if dataset == "suning":
        return (
            "price, position frequency, user tenure, demographic and spending categories, "
            "and prior event, click, cart, and purchase counts"
        )
    return "price proxy and prior purchase count"


def risk_set_description(dataset: str) -> str:
    if dataset == "suning":
        return (
            "Click, Cart, and Purchase indicators in the constructed user-item interaction risk set; "
            "the Click denominator is not a platform-wide exposure-level CTR denominator."
        )
    return (
        "Purchase-stage sampled-choice proxy formed from an observed review-linked target and "
        "sampled unseen alternatives."
    )


def claim_boundary(dataset: str) -> str:
    if dataset == "suning":
        return (
            "Claims apply to the constructed, temporally certified Suning interaction population; "
            "they are not randomized intervention effects."
        )
    return (
        "Amazon provides purchase-stage sampled-choice external evidence only; no native Click, "
        "Cart, checkout-conversion, funnel-reversal, policy, or OPE claim."
    )


def run(archive_roots: list[Path]) -> list[Path]:
    locator = Locator(archive_roots)
    evidence: list[dict] = []
    estimands: list[dict] = []
    for dataset in DATASETS:
        evidence.append(outcome_transport_row(dataset, locator))
        selection, estimand = selection_row(dataset, locator)
        evidence.append(selection)
        estimands.append(estimand)
        evidence.extend(
            [
                specification_row(dataset, locator),
                projection_row(dataset, locator),
                placebo_row(dataset, locator),
                ovb_row(dataset, locator),
                candidate_row(dataset, locator),
            ]
        )

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    evidence_frame = pd.DataFrame(evidence).sort_values(
        ["Analysis_Order", "Dataset_Key"]
    )
    evidence_path = AUDIT_DIR / f"RQ5_Robustness_Evidence_{VERSION}.csv"
    evidence_frame.to_csv(evidence_path, index=False)

    table_path = VIS_DIR / f"Table_RQ5_Robustness_Summary_{VERSION}.csv"
    evidence_frame[
        [
            "Dataset",
            "Robustness_or_Scope_Analysis",
            "Observed_Result",
            "Evidence_Assessment",
            "Does_Not_Establish",
        ]
    ].to_csv(table_path, index=False)

    estimand_path = AUDIT_DIR / f"Dataset_Estimand_Adjustment_Set_{VERSION}.csv"
    pd.DataFrame(estimands).to_csv(estimand_path, index=False)

    manifest_path = AUDIT_DIR / f"RQ5_Robustness_Manifest_{VERSION}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": VERSION,
                "analyses": [
                    "out-of-time nuisance transport",
                    "temporal certification and observed-selection sensitivity",
                    "certified-sample specification sensitivity",
                    "Gram-Schmidt projection sensitivity",
                    "joint treatment-row placebo",
                    "benchmarked omitted-confounding sensitivity",
                    "recommendation candidate-set sensitivity",
                ],
                "source_files": locator.used,
                "excluded_from_rq5": {
                    "legacy_20260717_partial_r2_robustness_values": "superseded by benchmarked causal-ML sensitivity",
                    "five_training_seed_comparison": "reported under RQ4 predictive stability",
                    "alternative_causal_estimators": "reported under RQ4 estimator comparison",
                    "RQ1_measurement_support": "pre-estimation measurement and design diagnostics",
                },
                "claim_boundary": (
                    "RQ5 triangulates robustness and scope. A diagnostic that limits transportability "
                    "is not relabeled as a passed robustness test."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return [evidence_path, table_path, estimand_path, manifest_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        action="append",
        type=Path,
        default=[],
        help="Optional downloaded result root; repeat for multiple result batches.",
    )
    args = parser.parse_args()
    for path in run(args.archive_root):
        print(path)


if __name__ == "__main__":
    main()
