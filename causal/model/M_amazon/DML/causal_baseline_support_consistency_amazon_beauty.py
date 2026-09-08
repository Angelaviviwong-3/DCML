#!/usr/bin/env python3
"""Create a support-consistent, publication-ready Beauty causal baseline table.

The 20260719 baseline already contains the valid full-certified 5/9 estimates.
This script does not re-estimate the primary DML model. It audits treatment
support using the frozen 20260717 residuals and excludes specifications that
jointly include a treatment that failed the primary support gate.
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

import pandas as pd


CAUSAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_inference import (
    clean_component_name,
    inference_contract,
    prepare_orthogonalized_data,
    support_diagnostics,
)
from dcml_utils import ensure_dir, setup_logging


DATASET = "amazon_beauty"
OUTPUT_VERSION = "20260720"
SOURCE_BASELINE_VERSION = "20260719"
SOURCE_RESIDUAL_VERSION = "20260717"


def main() -> None:
    logger = setup_logging(
        ensure_dir(_dcml_paths.workspace_path(CAUSAL_ROOT, 'log')),
        f"causal_baseline_support_consistency_{DATASET}_{OUTPUT_VERSION}",
    )
    contract = inference_contract(CAUSAL_ROOT, DATASET)
    with open(contract["schema"], "r", encoding="utf-8") as handle:
        schema = json.load(handle)

    final, primary_specs, _, _ = prepare_orthogonalized_data(
        CAUSAL_ROOT,
        DATASET,
        "linear",
    )
    output_dir = ensure_dir(_dcml_paths.workspace_path(CAUSAL_ROOT, 'processed_data') / DATASET / "Final_Causal_Output")
    source_path = output_dir / f"{DATASET}_Causal_Baseline_Comparison_{SOURCE_BASELINE_VERSION}.csv"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing frozen baseline result: {source_path}. Run the 20260719 Beauty causal baseline first."
        )

    requested = {
        "Aggregate": list(schema["aggregate_treatments"]),
        "Disaggregate": list(schema["disaggregate_treatments"]),
    }
    audit_rows: list[dict] = []
    valid_components: dict[str, set[str]] = {}
    failed_components: dict[str, set[str]] = {}
    for resolution, treatments in requested.items():
        valid_components[resolution] = set()
        failed_components[resolution] = set()
        requested_by_component = {
            clean_component_name(treatment): treatment for treatment in treatments
        }
        for residual_column in primary_specs[resolution]:
            diagnostics = support_diagnostics(final, residual_column)
            component = clean_component_name(residual_column)
            treatment = requested_by_component[component]
            if diagnostics["Estimable"]:
                valid_components[resolution].add(component)
            else:
                failed_components[resolution].add(component)
            audit_rows.append(
                {
                    "Dataset": DATASET,
                    "Resolution": resolution,
                    "Component": component,
                    "Requested_Treatment_Column": treatment,
                    "Residual_Column": residual_column,
                    "Publication_Eligible": bool(diagnostics["Estimable"]),
                    "Support_Contract": "same frozen residual support gate as the 20260717 primary DML",
                    **diagnostics,
                }
            )

    audit = pd.DataFrame(audit_rows)
    baseline = pd.read_csv(source_path)
    baseline["Source_Model_Specification"] = baseline["Model_Specification"].astype(str)
    publication_parts: list[pd.DataFrame] = []
    excluded_parts: list[pd.DataFrame] = []

    for resolution in requested:
        subset = baseline[baseline["Resolution"].astype(str).eq(resolution)].copy()
        full_valid = subset["Model_Specification"].astype(str).str.startswith("Full_certified_")
        marginal = subset["Model_Specification"].astype(str).eq("Marginal_single_treatment")
        component_valid = subset["Component"].astype(str).isin(valid_components[resolution])
        keep = full_valid | (marginal & component_valid)

        eligible = subset.loc[keep].copy()
        joint = eligible["Model_Specification"].astype(str).str.startswith("Full_certified_")
        estimated_size = len(valid_components[resolution])
        eligible.loc[joint, "Model_Specification"] = (
            f"Primary_and_full_certified_support_gated_{estimated_size}_without_T_int_sem"
        )
        eligible["Publication_Eligible"] = True
        eligible["Specification_Role"] = "support-consistent joint benchmark"
        eligible.loc[marginal.loc[eligible.index], "Specification_Role"] = (
            "support-consistent marginal GPS diagnostic"
        )
        eligible["Support_Contract"] = (
            "T_int_sem excluded because it failed the frozen primary DML residual-support gate"
        )
        eligible["Primary_Full_Sensitivity_Independence"] = (
            "not_independent_for_beauty_primary_support_gate_equals_full_5_9"
        )
        eligible["Output_Version"] = OUTPUT_VERSION
        publication_parts.append(eligible)

        excluded = subset.loc[~keep].copy()
        excluded["Publication_Eligible"] = False
        excluded["Exclusion_Reason"] = (
            "joint specification contains non-estimable T_int_sem"
        )
        excluded.loc[
            excluded["Model_Specification"].astype(str).eq("Marginal_single_treatment"),
            "Exclusion_Reason",
        ] = "T_int_sem failed the primary residual-support gate"
        excluded["Output_Version"] = OUTPUT_VERSION
        excluded_parts.append(excluded)

    publication = pd.concat(publication_parts, ignore_index=True)
    excluded = pd.concat(excluded_parts, ignore_index=True)
    publication_path = output_dir / (
        f"{DATASET}_Causal_Baseline_Comparison_Support_Consistent_{OUTPUT_VERSION}.csv"
    )
    excluded_path = output_dir / f"{DATASET}_Causal_Baseline_Excluded_{OUTPUT_VERSION}.csv"
    audit_path = output_dir / f"{DATASET}_Causal_Baseline_Support_Audit_{OUTPUT_VERSION}.csv"
    manifest_path = output_dir / f"{DATASET}_Causal_Baseline_Support_Manifest_{OUTPUT_VERSION}.json"

    publication.to_csv(publication_path, index=False)
    excluded.to_csv(excluded_path, index=False)
    audit.to_csv(audit_path, index=False)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": DATASET,
                "output_version": OUTPUT_VERSION,
                "source_baseline_version": SOURCE_BASELINE_VERSION,
                "source_residual_version": SOURCE_RESIDUAL_VERSION,
                "data_interpretation": (
                    "Beauty T_int_sem has insufficient residual variation under the frozen support rule; "
                    "this is a data-support limitation, not a recoverable coding error."
                ),
                "correction": (
                    "Exclude joint 6/10 specifications and marginal T_int_sem diagnostics from the "
                    "publication-ready baseline; reuse the already estimated support-consistent 5/9 rows."
                ),
                "aggregate_requested": len(requested["Aggregate"]),
                "aggregate_estimable": len(valid_components["Aggregate"]),
                "disaggregate_requested": len(requested["Disaggregate"]),
                "disaggregate_estimable": len(valid_components["Disaggregate"]),
                "failed_components": {
                    key: sorted(value) for key, value in failed_components.items()
                },
                "publication_rows": int(len(publication)),
                "excluded_rows": int(len(excluded)),
                "primary_full_sensitivity_relationship": (
                    "For Beauty, the support-gated primary model and the full-certified 5/9 model are "
                    "the same estimable specification and are not independent sensitivity analyses."
                ),
                "claim_boundary": contract["claim"],
                "publication_result": str(publication_path),
                "excluded_audit": str(excluded_path),
                "support_audit": str(audit_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(
        "Wrote support-consistent Beauty baseline: publication_rows=%d excluded_rows=%d",
        len(publication),
        len(excluded),
    )


if __name__ == "__main__":
    main()
