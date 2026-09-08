#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate completeness and key contracts of the 20260719 supplements."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[0]))
import project_paths as _dcml_paths


import argparse
import json
from pathlib import Path

import pandas as pd

from dcml_utils import ensure_dir, find_causal_root


VERSION = "20260719"
DATASETS = ("suning", "amazon_appliances", "amazon_beauty")


def _paths(root: Path, dataset: str, include_multiseed: bool) -> list[tuple[str, Path]]:
    base = _dcml_paths.workspace_path(root, 'processed_data') / dataset
    final = base / "Final_Causal_Output"
    dml = base / "DML_Results"
    audit = base / "audit"
    comparison = _dcml_paths.workspace_path(root, 'results_of_comparison') / dataset / "dcml_performance"
    channel_name = (
        f"Final_Channel_Analysis_Reporting_{VERSION}.csv"
        if dataset == "suning"
        else f"{dataset}_Final_Channel_Analysis_Reporting_{VERSION}.csv"
    )
    paths = [
        ("h3_contrasts", final / f"{dataset}_H3_interaction_contrasts_{VERSION}.csv"),
        ("h3_omnibus", final / f"{dataset}_H3_omnibus_tests_{VERSION}.csv"),
        ("h3_level_contract", final / f"{dataset}_H3_level_contract_{VERSION}.json"),
        ("ate_specification", final / f"{dataset}_ATE_specification_sensitivity_{VERSION}.csv"),
        ("ate_specification_comparison", final / f"{dataset}_ATE_specification_sensitivity_comparison_{VERSION}.csv"),
        ("causal_baselines", final / f"{dataset}_Causal_Baseline_Comparison_{VERSION}.csv"),
        ("causal_baseline_schema", final / f"{dataset}_Causal_Baseline_Schema_{VERSION}.json"),
        ("channel_reporting", final / channel_name),
        ("nuisance_transport", dml / f"{dataset}_nuisance_temporal_transport_diagnostics_{VERSION}.csv"),
        ("nuisance_transport_manifest", dml / f"{dataset}_nuisance_temporal_transport_manifest_{VERSION}.json"),
        ("selection_model", audit / f"{dataset}_temporal_certification_selection_model_{VERSION}.csv"),
        ("selection_balance", audit / f"{dataset}_temporal_certification_weighted_balance_{VERSION}.csv"),
        ("selection_manifest", audit / f"{dataset}_temporal_certification_selection_manifest_{VERSION}.json"),
        ("prediction_feature_ablation", comparison / f"{dataset}_Treatment_Feature_Ablation_{VERSION}.csv"),
        ("prediction_feature_manifest", comparison / f"{dataset}_Treatment_Feature_Ablation_Manifest_{VERSION}.json"),
    ]
    if dataset == "suning":
        paths.extend(
            [
                ("h4_full_certified", final / f"suning_H4_full_certified_specification_sensitivity_{VERSION}.csv"),
                ("h4_full_certified_classification", final / f"suning_H4_full_certified_reversal_classification_{VERSION}.csv"),
            ]
        )
    if include_multiseed:
        paths.extend(
            [
                ("multiseed_runs", comparison / f"Recommendation_Training_Seed_Runs_{VERSION}.csv"),
                ("multiseed_summary", comparison / f"Recommendation_Training_Seed_Summary_{VERSION}.csv"),
                ("multiseed_manifest", comparison / f"Recommendation_Training_Seed_Manifest_{VERSION}.json"),
            ]
        )
    return paths


def _semantic_checks(root: Path, dataset: str) -> list[dict]:
    output = _dcml_paths.workspace_path(root, 'processed_data') / dataset / "Final_Causal_Output"
    comparison = _dcml_paths.workspace_path(root, 'results_of_comparison') / dataset / "dcml_performance"
    rows: list[dict] = []

    level_path = output / f"{dataset}_H3_level_contract_{VERSION}.json"
    contrasts_path = output / f"{dataset}_H3_interaction_contrasts_{VERSION}.csv"
    if level_path.is_file() and contrasts_path.is_file():
        with open(level_path, "r", encoding="utf-8") as handle:
            levels = set(json.load(handle)["observed_h_levels"])
        contrasts = pd.read_csv(contrasts_path)
        invalid = []
        for value in contrasts.loc[contrasts["Contrast_Valid"].fillna(False), "Contrast"].astype(str):
            mentioned = [int(token[1:]) for token in value.split("-") if token.startswith("H")]
            if any(level not in levels for level in mentioned):
                invalid.append(value)
        rows.append(
            {
                "Dataset": dataset,
                "Check": "H3 contrasts use observed levels only",
                "Status": "PASS" if not invalid else "FAIL",
                "Detail": "|".join(sorted(set(invalid))) if invalid else f"observed={sorted(levels)}",
            }
        )

    specification_path = output / f"{dataset}_ATE_specification_sensitivity_{VERSION}.csv"
    if specification_path.is_file():
        frame = pd.read_csv(specification_path)
        sizes = set(pd.to_numeric(frame["Requested_Treatment_Count"], errors="coerce").dropna().astype(int))
        rows.append(
            {
                "Dataset": dataset,
                "Check": "ATE sensitivity contains 6/10 primary and 5/9 full specifications",
                "Status": "PASS" if {5, 6, 9, 10}.issubset(sizes) else "FAIL",
                "Detail": f"requested_sizes={sorted(sizes)}",
            }
        )

    baseline_path = output / f"{dataset}_Causal_Baseline_Comparison_{VERSION}.csv"
    if baseline_path.is_file():
        estimators = set(pd.read_csv(baseline_path)["Estimator"].dropna().astype(str))
        expected = {"Naive OLS", "Standard DML (No GS)", "Marginal GPS-IPW"}
        rows.append(
            {
                "Dataset": dataset,
                "Check": "causal baseline estimator contract",
                "Status": "PASS" if expected.issubset(estimators) else "FAIL",
                "Detail": "|".join(sorted(estimators)),
            }
        )

    ablation_path = comparison / f"{dataset}_Treatment_Feature_Ablation_{VERSION}.csv"
    if ablation_path.is_file():
        specifications = set(pd.read_csv(ablation_path)["Specification"].dropna().astype(str))
        expected = {"Base-X", "Aggregate-6", "Disaggregate-10", "Full-12"}
        rows.append(
            {
                "Dataset": dataset,
                "Check": "predictive feature ablation specifications",
                "Status": "PASS" if specifications == expected else "FAIL",
                "Detail": "|".join(sorted(specifications)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-multiseed", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    root = find_causal_root(__file__)
    inventory: list[dict] = []
    checks: list[dict] = []
    for dataset in DATASETS:
        for role, path in _paths(root, dataset, args.require_multiseed):
            exists = path.is_file()
            inventory.append(
                {
                    "Dataset": dataset,
                    "Role": role,
                    "Path": str(path),
                    "Exists": exists,
                    "Bytes": int(path.stat().st_size) if exists else 0,
                    "Status": "PASS" if exists and path.stat().st_size > 0 else "FAIL",
                }
            )
        checks.extend(_semantic_checks(root, dataset))

    if args.require_multiseed:
        for dataset in DATASETS:
            path = _dcml_paths.workspace_path(root, 'results_of_comparison') / dataset / "dcml_performance" / f"Recommendation_Training_Seed_Runs_{VERSION}.csv"
            if path.is_file():
                runs = pd.read_csv(path)
                count = int(runs["Seed_Offset"].nunique())
                checks.append(
                    {
                        "Dataset": dataset,
                        "Check": "five independent training-seed offsets",
                        "Status": "PASS" if count == 5 else "FAIL",
                        "Detail": f"seed_count={count}",
                    }
                )

    audit = ensure_dir(_dcml_paths.workspace_path(root, 'processed_data') / "audit")
    inventory_frame = pd.DataFrame(inventory)
    checks_frame = pd.DataFrame(checks)
    inventory_path = audit / f"supplemental_output_inventory_{VERSION}.csv"
    checks_path = audit / f"supplemental_quality_checks_{VERSION}.csv"
    manifest_path = audit / f"supplemental_validation_manifest_{VERSION}.json"
    inventory_frame.to_csv(inventory_path, index=False)
    checks_frame.to_csv(checks_path, index=False)
    failed_inventory = int(inventory_frame["Status"].eq("FAIL").sum())
    failed_checks = int(checks_frame["Status"].eq("FAIL").sum()) if not checks_frame.empty else 0
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "script_version": VERSION,
                "require_multiseed": args.require_multiseed,
                "status": "PASS" if failed_inventory == 0 and failed_checks == 0 else "FAIL",
                "failed_inventory_count": failed_inventory,
                "failed_semantic_check_count": failed_checks,
                "inventory_file": str(inventory_path),
                "quality_checks_file": str(checks_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"supplemental validation: missing_or_empty={failed_inventory}, failed_checks={failed_checks}")
    if args.fail_on_error and (failed_inventory or failed_checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
