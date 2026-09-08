#!/usr/bin/env python3
"""Validate the lightweight 20260729 RQ5 outputs."""

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
ROOT = Path(__file__).resolve().parent
DATASETS = ("suning", "amazon_appliances", "amazon_beauty")


def expected_paths() -> list[Path]:
    paths = [
        _dcml_paths.workspace_path(ROOT, 'processed_data') / "audit" / f"DML_OVB_Postprocess_Manifest_{VERSION}.json",
        _dcml_paths.workspace_path(ROOT, 'processed_data') / "audit" / f"RQ5_Robustness_Evidence_{VERSION}.csv",
        _dcml_paths.workspace_path(ROOT, 'processed_data') / "audit" / f"Dataset_Estimand_Adjustment_Set_{VERSION}.csv",
        _dcml_paths.workspace_path(ROOT, 'processed_data') / "audit" / f"RQ5_Robustness_Manifest_{VERSION}.json",
        _dcml_paths.workspace_path(ROOT, "Unified_Visualization/output") / f"DML_OVB_Focal_Bounds_{VERSION}.csv",
        _dcml_paths.workspace_path(ROOT, "Unified_Visualization/output") / f"Table_DML_OVB_Focal_Summary_{VERSION}.csv",
        _dcml_paths.workspace_path(ROOT, "Unified_Visualization/output") / f"Table_RQ5_Robustness_Summary_{VERSION}.csv",
        _dcml_paths.workspace_path(ROOT, "Unified_Visualization/output") / f"Figure_DML_OVB_Sensitivity_{VERSION}.pdf",
        _dcml_paths.workspace_path(ROOT, "Unified_Visualization/output") / f"Figure_DML_OVB_Sensitivity_{VERSION}.png",
    ]
    paths.extend(
        _dcml_paths.workspace_path(ROOT, 'processed_data')
        / dataset
        / "Final_Causal_Output"
        / f"{dataset}_DML_OVB_Sensitivity_{VERSION}.csv"
        for dataset in DATASETS
    )
    return paths


def validate() -> list[dict]:
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"Check": name, "Passed": bool(passed), "Detail": detail})

    for path in expected_paths():
        add(f"exists:{path.name}", path.is_file() and path.stat().st_size > 0, str(path))

    evidence_path = _dcml_paths.workspace_path(ROOT, 'processed_data') / "audit" / f"RQ5_Robustness_Evidence_{VERSION}.csv"
    if evidence_path.is_file():
        evidence = pd.read_csv(evidence_path)
        add("rq5_row_count", len(evidence) == 21, f"rows={len(evidence)}; expected=21")
        add(
            "rq5_dataset_coverage",
            set(evidence["Dataset_Key"].astype(str)) == set(DATASETS),
            ",".join(sorted(evidence["Dataset_Key"].astype(str).unique())),
        )

    for dataset in DATASETS:
        path = (
            _dcml_paths.workspace_path(ROOT, 'processed_data')
            / dataset
            / "Final_Causal_Output"
            / f"{dataset}_DML_OVB_Sensitivity_{VERSION}.csv"
        )
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        boundary = frame["Bound_Status"].astype(str).isin(
            {"UNBOUNDED_CF_D_BOUNDARY", "NONINFORMATIVE_ZERO_OUTCOME_GAIN"}
        )
        numeric = pd.to_numeric(frame["Adversarial_Bias_Adjusted_ATE"], errors="coerce")
        add(
            f"ovb_boundary_not_numeric:{dataset}",
            numeric[boundary].isna().all(),
            f"boundary_rows={int(boundary.sum())}",
        )
        finite = frame["Bound_Status"].astype(str).isin({"FINITE", "REFERENCE"})
        add(
            f"ovb_finite_values_present:{dataset}",
            np.isfinite(numeric[finite]).all(),
            f"finite_rows={int(finite.sum())}",
        )

    manifest_path = _dcml_paths.workspace_path(ROOT, 'processed_data') / "audit" / f"RQ5_Robustness_Manifest_{VERSION}.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        excluded = manifest.get("excluded_from_rq5", {})
        add(
            "legacy_rv_retired",
            "legacy_20260717_partial_r2_robustness_values" in excluded,
            str(excluded.get("legacy_20260717_partial_r2_robustness_values", "missing")),
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    checks = validate()
    output = _dcml_paths.workspace_path(ROOT, 'processed_data') / "audit" / f"RQ5_Output_Validation_{VERSION}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(output, index=False)
    failed = [row for row in checks if not row["Passed"]]
    print(f"RQ5 validation: checks={len(checks)}, failed={len(failed)}")
    print(output)
    if failed:
        for row in failed:
            print(f"FAILED: {row['Check']} | {row['Detail']}")
        if args.fail_on_error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
