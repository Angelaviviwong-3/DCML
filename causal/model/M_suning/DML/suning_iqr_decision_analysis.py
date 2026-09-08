#!/usr/bin/env python3
"""Suning-only IQR-scaled, decision-relevant effect profiles.

This is a deterministic transformation of frozen DCML estimates. It is not
off-policy evaluation, policy learning, or an estimate of realized policy
value because logged action propensities are unavailable.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths


import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CAUSAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_utils import ensure_dir, setup_logging
from dcml_inference import (
    clean_component_name,
    numeric_frame,
    prepare_orthogonalized_data,
)


OUTPUT_VERSION = "20260720"
SOURCE_PRIMARY_VERSION = "20260717"
SOURCE_SENSITIVITY_VERSION = "20260719"


def _classification_map(path: Path, specification_role: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["Resolution"].astype(str).eq("Aggregate")].copy()
    frame["Specification_Role"] = specification_role
    columns = [
        "Specification_Role",
        "Component",
        "Pattern_Classification",
        "Purchase_Minus_Click",
        "Difference_P_Holm",
    ]
    return frame[columns].drop_duplicates(["Specification_Role", "Component"])


def _analysis_sample_iqr() -> pd.DataFrame:
    final, specifications, _, _ = prepare_orthogonalized_data(
        CAUSAL_ROOT,
        "suning",
        "linear",
    )
    aggregate = list(specifications["Aggregate"])
    columns_by_role = {
        "primary_history_eligible_aggregate_6": aggregate,
        "full_certified_aggregate_5_without_T_int_sem": [
            column for column in aggregate if clean_component_name(column) != "T_int_sem"
        ],
    }
    rows: list[dict] = []
    for role, treatments in columns_by_role.items():
        for stage in ("click", "cart", "purchase"):
            outcome = f"res_y_{stage}"
            complete = numeric_frame(final, [outcome, *treatments]).dropna()
            for treatment in treatments:
                values = complete[treatment].to_numpy(dtype=float)
                q25, q75 = np.percentile(values, [25, 75])
                rows.append(
                    {
                        "Specification_Role": role,
                        "Stage": stage,
                        "Component": clean_component_name(treatment),
                        "Analysis_Sample_N_Audit": int(len(complete)),
                        "Analysis_Sample_Treatment_Column": treatment,
                        "Analysis_Sample_Q25": float(q25),
                        "Analysis_Sample_Q75": float(q75),
                        "Analysis_Sample_Residual_IQR": float(q75 - q25),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    logger = setup_logging(
        ensure_dir(_dcml_paths.workspace_path(CAUSAL_ROOT, 'log')),
        f"suning_iqr_decision_analysis_{OUTPUT_VERSION}",
    )
    output_dir = ensure_dir(_dcml_paths.workspace_path(CAUSAL_ROOT, 'processed_data') / "suning" / "Final_Causal_Output")
    sensitivity_path = output_dir / (
        f"suning_ATE_specification_sensitivity_{SOURCE_SENSITIVITY_VERSION}.csv"
    )
    primary_h4_path = output_dir / f"suning_H4_reversal_classification_{SOURCE_PRIMARY_VERSION}.csv"
    full_h4_path = output_dir / (
        f"suning_H4_full_certified_reversal_classification_{SOURCE_SENSITIVITY_VERSION}.csv"
    )
    required = (sensitivity_path, primary_h4_path, full_h4_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen inputs:\n" + "\n".join(missing))

    estimates = pd.read_csv(sensitivity_path)
    estimates = estimates[estimates["Resolution"].astype(str).eq("Aggregate")].copy()
    estimates = estimates[
        estimates["Estimable"].astype(str).str.lower().eq("true")
    ].copy()
    estimates["Specification_Role"] = np.where(
        estimates["Model_Specification"].astype(str).str.startswith("Primary_support_gated_"),
        "primary_history_eligible_aggregate_6",
        "full_certified_aggregate_5_without_T_int_sem",
    )
    estimates = estimates[
        estimates["Model_Specification"].astype(str).str.startswith(
            ("Primary_support_gated_", "Full_certified_")
        )
    ].copy()

    classifications = pd.concat(
        [
            _classification_map(primary_h4_path, "primary_history_eligible_aggregate_6"),
            _classification_map(full_h4_path, "full_certified_aggregate_5_without_T_int_sem"),
        ],
        ignore_index=True,
    )
    profiles = estimates.merge(
        classifications,
        on=["Specification_Role", "Component"],
        how="left",
        validate="many_to_one",
    )
    profiles = profiles.merge(
        _analysis_sample_iqr(),
        on=["Specification_Role", "Stage", "Component"],
        how="left",
        validate="one_to_one",
    )

    for column in (
        "Coefficient",
        "CI_Lower_95",
        "CI_Upper_95",
        "Residual_IQR_Final",
        "Analysis_Sample_Residual_IQR",
    ):
        profiles[column] = pd.to_numeric(profiles[column], errors="coerce")
    profiles = profiles[profiles["Analysis_Sample_Residual_IQR"].gt(0)].copy()
    profiles["Analysis_Sample_N_Matches_Estimation_N"] = (
        pd.to_numeric(profiles["N"], errors="coerce")
        .eq(pd.to_numeric(profiles["Analysis_Sample_N_Audit"], errors="coerce"))
    )
    if not profiles["Analysis_Sample_N_Matches_Estimation_N"].all():
        raise RuntimeError("The reconstructed IQR analysis sample does not match the frozen estimate N.")
    profiles["Contrast_Definition"] = "Q25_to_Q75_increase_in_frozen_residualized_treatment"
    profiles["IQR_Scaled_Change"] = profiles["Analysis_Sample_Residual_IQR"]
    profiles["Outcome_Change_Probability"] = (
        profiles["Coefficient"] * profiles["Analysis_Sample_Residual_IQR"]
    )
    profiles["Outcome_Change_Percentage_Points"] = 100.0 * profiles["Outcome_Change_Probability"]
    profiles["Outcome_Change_CI_Lower_95_pp"] = (
        100.0 * profiles["CI_Lower_95"] * profiles["Analysis_Sample_Residual_IQR"]
    )
    profiles["Outcome_Change_CI_Upper_95_pp"] = (
        100.0 * profiles["CI_Upper_95"] * profiles["Analysis_Sample_Residual_IQR"]
    )
    profiles["Quarter_IQR_Change_Percentage_Points"] = (
        0.25 * profiles["Outcome_Change_Percentage_Points"]
    )
    profiles["Decision_Relevance"] = np.where(
        profiles["Pattern_Classification"].astype(str).eq("strict_reversal"),
        "stage-specific sign reversal; optimizing one funnel stage cannot be treated as a proxy for another",
        "stage-specific effect profile; no strict reversal conclusion",
    )
    profiles["Analysis_Type"] = "IQR-scaled model-based decision analysis"
    profiles["Formal_OPE"] = False
    profiles["Logged_Action_Propensity_Available"] = False
    profiles["Claim_Boundary"] = (
        "Decision-relevant transformation of frozen DCML coefficients among temporally certified items; "
        "not IPS/DR OPE, policy learning, realized uplift, or online experimental evidence."
    )
    profiles["Output_Version"] = OUTPUT_VERSION

    keep_columns = [
        "Specification_Role",
        "Model_Specification",
        "Stage",
        "Component",
        "N",
        "Analysis_Sample_N_Audit",
        "Analysis_Sample_N_Matches_Estimation_N",
        "Analysis_Sample_Treatment_Column",
        "Analysis_Sample_Q25",
        "Analysis_Sample_Q75",
        "Analysis_Sample_Residual_IQR",
        "Residual_IQR_Final",
        "Coefficient",
        "P_Value_Holm",
        "Significant_Holm_05",
        "Pattern_Classification",
        "Purchase_Minus_Click",
        "Difference_P_Holm",
        "Contrast_Definition",
        "Outcome_Change_Percentage_Points",
        "Outcome_Change_CI_Lower_95_pp",
        "Outcome_Change_CI_Upper_95_pp",
        "Quarter_IQR_Change_Percentage_Points",
        "Decision_Relevance",
        "Analysis_Type",
        "Formal_OPE",
        "Logged_Action_Propensity_Available",
        "Claim_Boundary",
        "Output_Version",
    ]
    profiles = profiles[keep_columns].sort_values(
        ["Specification_Role", "Component", "Stage"]
    )
    profile_path = output_dir / f"suning_IQR_Scaled_Effect_Profiles_{OUTPUT_VERSION}.csv"
    strict_path = output_dir / f"suning_Decision_Relevant_Strict_Reversals_{OUTPUT_VERSION}.csv"
    manifest_path = output_dir / f"suning_IQR_Decision_Analysis_Manifest_{OUTPUT_VERSION}.json"
    profiles.to_csv(profile_path, index=False)
    profiles[
        profiles["Pattern_Classification"].astype(str).eq("strict_reversal")
    ].to_csv(strict_path, index=False)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": "suning",
                "output_version": OUTPUT_VERSION,
                "source_primary_version": SOURCE_PRIMARY_VERSION,
                "source_sensitivity_version": SOURCE_SENSITIVITY_VERSION,
                "analysis": (
                    "Translate each Aggregate DCML coefficient into the model-implied change associated "
                    "with a Q25-to-Q75 increase in the corresponding residualized treatment."
                ),
                "confirmatory_status": "exploratory decision-relevant analysis; not a new hypothesis test",
                "formal_ope": False,
                "why_not_ope": (
                    "The observational log contains outcomes but no known logging-policy action "
                    "propensities for content interventions, so IPS/DR policy-value identification is unavailable."
                ),
                "recommended_paper_location": (
                    "Managerial implications or an exploratory decision-analysis subsection after RQ5; "
                    "do not use it as evidence that an offline policy would improve realized outcomes."
                ),
                "primary_output": str(profile_path),
                "strict_reversal_output": str(strict_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Wrote %d Suning IQR-scaled stage-effect rows", len(profiles))


if __name__ == "__main__":
    main()
