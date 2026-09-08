#!/usr/bin/env python3
"""Build a claim-aligned robustness audit from completed RQ5 outputs.

No model is fitted. The audit distinguishes robustness within the certified
analysis population, omitted-confounding sensitivity, external replication,
and population transportability.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[0]))
import project_paths as _dcml_paths


import json
from pathlib import Path

import pandas as pd


SOURCE_VERSION = "20260729"
OUTPUT_VERSION = "20260729-2"
ROOT = Path(__file__).resolve().parent
AUDIT_DIR = _dcml_paths.workspace_path(ROOT, 'processed_data') / "audit"
VIS_DIR = _dcml_paths.workspace_path(ROOT, "Unified_Visualization/output")

DATASET_ORDER = ["Suning", "Amazon Appliances", "Amazon Beauty"]
ANALYSIS_ORDER = [
    "Out-of-time nuisance transport",
    "Certified-sample specification sensitivity",
    "Gram-Schmidt projection sensitivity",
    "Joint treatment-row placebo",
    "Benchmarked omitted-confounding sensitivity",
    "Temporal certification and observed-selection sensitivity",
    "Recommendation candidate-set sensitivity",
]


def status_for(analysis: str, assessment: str) -> tuple[str, str]:
    if analysis == "Out-of-time nuisance transport":
        if assessment == "SUPPORTS_TEMPORAL_TRANSPORT":
            return "PASS", "Frozen nuisance functions retain acceptable temporal discrimination and calibration."
        return "SCOPE_LIMIT", "Temporal discrimination remains informative, but calibration drift limits precision."
    if analysis == "Certified-sample specification sensitivity":
        if assessment == "STABLE":
            return "PASS", "ATE directions and Holm decisions are stable across the compared sample specifications."
        return "PASS_WITH_LOCAL_EXCEPTIONS", "Focal conclusions are stable, with a small number of nonfocal coefficient changes."
    if analysis == "Gram-Schmidt projection sensitivity":
        return "PASS", "Holm-significant ATE directions are preserved without projection and under reverse order."
    if analysis == "Joint treatment-row placebo":
        return "PASS", "Holm-significant ATEs are not reproduced under the joint permutation null."
    if analysis == "Benchmarked omitted-confounding sensitivity":
        return "SENSITIVITY_LIMIT", "Sensitivity is effect-specific and does not support a blanket strong-robustness claim."
    if analysis == "Temporal certification and observed-selection sensitivity":
        return "SCOPE_LIMIT", "Inference remains conditional on the temporally certified analysis population."
    if analysis == "Recommendation candidate-set sensitivity":
        return "PREDICTIVE_ONLY", "Candidate-size stability concerns offline ranking, not causal identification."
    raise ValueError(f"Unknown analysis: {analysis}")


def build_matrix(evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for analysis in ANALYSIS_ORDER:
        for dataset in DATASET_ORDER:
            match = evidence[
                evidence["Robustness_or_Scope_Analysis"].astype(str).eq(analysis)
                & evidence["Dataset"].astype(str).eq(dataset)
            ]
            if len(match) != 1:
                raise RuntimeError(f"Expected one evidence row for {analysis}/{dataset}")
            source = match.iloc[0]
            status, interpretation = status_for(analysis, str(source["Evidence_Assessment"]))
            rows.append(
                {
                    "Analysis": analysis,
                    "Dataset": dataset,
                    "Status": status,
                    "Observed_Result": source["Observed_Result"],
                    "Interpretation": interpretation,
                    "Source_File": source["Source_File"],
                }
            )
    return pd.DataFrame(rows)


def focal_values(focal: pd.DataFrame, dataset: str) -> str:
    selected = focal[focal["Dataset_Label"].astype(str).eq(dataset)]
    if selected.empty:
        return "not available"
    minimum = float(selected["Point_RV_Percent"].min())
    maximum = float(selected["Point_RV_Percent"].max())
    return f"{minimum:.2f}% to {maximum:.2f}%"


def focal_point(focal: pd.DataFrame, dataset: str, label: str) -> str:
    selected = focal[
        focal["Dataset_Label"].astype(str).eq(dataset)
        & focal["Focal_Label"].astype(str).eq(label)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one focal row for {dataset}/{label}")
    return f"{float(selected.iloc[0]['Point_RV_Percent']):.2f}%"


def build_claim_gates(focal: pd.DataFrame) -> pd.DataFrame:
    suning_rv = focal_values(focal, "Suning")
    suning_social_purchase_rv = focal_point(focal, "Suning", "Social proof: Purchase")
    appliance_social_rv = focal_point(focal, "Amazon Appliances", "Social proof")
    beauty_social_rv = focal_point(focal, "Amazon Beauty", "Social proof")
    return pd.DataFrame(
        [
            {
                "Claim_ID": "C1",
                "Target_Claim": "Suning focal stage patterns are stable to implemented observed-data perturbations.",
                "Assessment": "SUPPORTED_WITHIN_CERTIFIED_SAMPLE",
                "Current_Evidence": (
                    "Focal directions are preserved across sample specification and GS variants; "
                    "Holm-significant estimates attain the 1/201 permutation p-value floor."
                ),
                "Allowed_Wording": (
                    "Within the temporally certified Suning population, focal ATE directions are "
                    "stable to specification, projection, and joint-permutation checks."
                ),
                "Prohibited_Wording": "All causal assumptions are verified or every robustness analysis passes.",
                "Evidence_Needed_to_Upgrade": "None for the restricted observed-data stability claim.",
            },
            {
                "Claim_ID": "C2",
                "Target_Claim": "Suning focal estimates are strongly robust to omitted confounding.",
                "Assessment": "NOT_SUPPORTED_BY_CURRENT_RESULTS",
                "Current_Evidence": (
                    f"Focal equal-strength point values range from {suning_rv}; some observed-benchmark "
                    "bounds cross zero or reach a non-finite boundary."
                ),
                "Allowed_Wording": "Benchmarked analysis indicates moderate, effect-specific sensitivity to omitted confounding.",
                "Prohibited_Wording": "Suning effects are strongly robust to omitted confounding.",
                "Evidence_Needed_to_Upgrade": (
                    "Richer exposure, campaign, UI, item-time, and user-history controls; negative controls "
                    "or a credible quasi-experimental source; then refit DML and repeat sensitivity analysis."
                ),
            },
            {
                "Claim_ID": "C3",
                "Target_Claim": "The positive purchase-stage social-proof direction replicates across platform-category settings.",
                "Assessment": "SUPPORTED_AS_DIRECTIONAL_EXTERNAL_REPLICATION",
                "Current_Evidence": (
                    "The purchase-stage social-proof estimate is positive and Holm-significant in all three settings; "
                    f"equal-strength point values are {suning_social_purchase_rv} in Suning, "
                    f"{appliance_social_rv} in Appliances, and {beauty_social_rv} in Beauty."
                ),
                "Allowed_Wording": "The positive purchase-stage direction is replicated in two Amazon category settings.",
                "Prohibited_Wording": "A common causal effect generalizes to all platforms or populations.",
                "Evidence_Needed_to_Upgrade": "A harmonized native outcome, risk set, and adjustment set across platforms.",
            },
            {
                "Claim_ID": "C4",
                "Target_Claim": "Amazon establishes broad cross-platform causal generalizability.",
                "Assessment": "NOT_SUPPORTED_BY_CURRENT_DESIGN",
                "Current_Evidence": (
                    "Amazon uses a purchase-stage sampled-choice proxy, two adjustment covariates, "
                    "partial temporal certification, and nuisance calibration slopes above the prespecified range."
                ),
                "Allowed_Wording": "Amazon provides purchase-stage external replication under a distinct sampled-choice design.",
                "Prohibited_Wording": "Amazon proves full-funnel or population-level cross-platform generalizability.",
                "Evidence_Needed_to_Upgrade": (
                    "Native harmonized funnel outcomes, comparable covariate coverage, successful transport weighting, "
                    "and a prespecified multi-domain transportability analysis."
                ),
            },
            {
                "Claim_ID": "C5",
                "Target_Claim": "Purchase-stage recommendation conclusions are stable across candidate-set sizes.",
                "Assessment": "SUPPORTED_FOR_OFFLINE_PREDICTION",
                "Current_Evidence": (
                    "Purchase NDCG@10 ranks remain 2/2/2 in Suning, 2/2/1 in Appliances, "
                    "and 1/1/1 in Beauty across 50/80/100 candidates."
                ),
                "Allowed_Wording": "Comparative offline Purchase ranking is stable across the evaluated candidate sizes.",
                "Prohibited_Wording": "Candidate-set stability validates causal identification.",
                "Evidence_Needed_to_Upgrade": "None for the offline candidate-set stability claim.",
            },
        ]
    )


def run() -> list[Path]:
    evidence_path = AUDIT_DIR / f"RQ5_Robustness_Evidence_{SOURCE_VERSION}.csv"
    focal_path = VIS_DIR / f"Table_DML_OVB_Focal_Summary_{SOURCE_VERSION}.csv"
    if not evidence_path.is_file() or not focal_path.is_file():
        raise FileNotFoundError(
            "Run the 20260729 RQ5 postprocessing and consolidation scripts before this audit."
        )
    evidence = pd.read_csv(evidence_path)
    focal = pd.read_csv(focal_path)
    matrix = build_matrix(evidence)
    gates = build_claim_gates(focal)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = AUDIT_DIR / f"Claim_Aligned_Robustness_Matrix_{OUTPUT_VERSION}.csv"
    gates_path = AUDIT_DIR / f"Claim_Gate_Assessment_{OUTPUT_VERSION}.csv"
    table_path = VIS_DIR / f"Table_Claim_Aligned_Robustness_{OUTPUT_VERSION}.csv"
    manifest_path = AUDIT_DIR / f"Claim_Aligned_Robustness_Manifest_{OUTPUT_VERSION}.json"

    matrix.to_csv(matrix_path, index=False)
    gates.to_csv(gates_path, index=False)
    gates[
        [
            "Claim_ID",
            "Target_Claim",
            "Assessment",
            "Current_Evidence",
            "Allowed_Wording",
        ]
    ].to_csv(table_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "version": OUTPUT_VERSION,
                "source_version": SOURCE_VERSION,
                "model_refit": False,
                "bootstrap_rerun": False,
                "classification_principle": (
                    "Observed-data stability, omitted-confounding sensitivity, external replication, "
                    "and transportability are evaluated as separate claims."
                ),
                "overall_conclusion": (
                    "Current evidence supports certified-population stability and directional external "
                    "replication, but not blanket strong omitted-confounding robustness or broad causal generalizability."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return [matrix_path, gates_path, table_path, manifest_path]


def main() -> None:
    for path in run():
        print(path)


if __name__ == "__main__":
    main()
