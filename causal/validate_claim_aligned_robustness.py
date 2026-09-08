#!/usr/bin/env python3
"""Validate the 20260729-2 claim-aligned robustness package."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[0]))
import project_paths as _dcml_paths


import argparse
from pathlib import Path

import pandas as pd


VERSION = "20260729-2"
ROOT = Path(__file__).resolve().parent
AUDIT_DIR = _dcml_paths.workspace_path(ROOT, 'processed_data') / "audit"
VIS_DIR = _dcml_paths.workspace_path(ROOT, "Unified_Visualization/output")


def validate() -> list[dict]:
    expected = [
        AUDIT_DIR / f"Claim_Aligned_Robustness_Matrix_{VERSION}.csv",
        AUDIT_DIR / f"Claim_Gate_Assessment_{VERSION}.csv",
        AUDIT_DIR / f"Claim_Aligned_Robustness_Manifest_{VERSION}.json",
        VIS_DIR / f"Table_Claim_Aligned_Robustness_{VERSION}.csv",
        VIS_DIR / f"Figure_Claim_Aligned_Robustness_{VERSION}.pdf",
        VIS_DIR / f"Figure_Claim_Aligned_Robustness_{VERSION}.png",
    ]
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"Check": name, "Passed": bool(passed), "Detail": detail})

    for path in expected:
        add(f"exists:{path.name}", path.is_file() and path.stat().st_size > 0, str(path))

    matrix_path = AUDIT_DIR / f"Claim_Aligned_Robustness_Matrix_{VERSION}.csv"
    if matrix_path.is_file():
        matrix = pd.read_csv(matrix_path)
        add("matrix_rows", len(matrix) == 21, f"rows={len(matrix)}")
        add("matrix_datasets", matrix["Dataset"].nunique() == 3, f"datasets={matrix['Dataset'].nunique()}")
        add("matrix_analyses", matrix["Analysis"].nunique() == 7, f"analyses={matrix['Analysis'].nunique()}")

    gates_path = AUDIT_DIR / f"Claim_Gate_Assessment_{VERSION}.csv"
    if gates_path.is_file():
        gates = pd.read_csv(gates_path).set_index("Claim_ID")
        add("claim_rows", len(gates) == 5, f"rows={len(gates)}")
        add(
            "strong_ovb_not_overclaimed",
            gates.loc["C2", "Assessment"] == "NOT_SUPPORTED_BY_CURRENT_RESULTS",
            str(gates.loc["C2", "Assessment"]),
        )
        add(
            "generalizability_not_overclaimed",
            gates.loc["C4", "Assessment"] == "NOT_SUPPORTED_BY_CURRENT_DESIGN",
            str(gates.loc["C4", "Assessment"]),
        )
        add(
            "external_replication_distinguished",
            gates.loc["C3", "Assessment"] == "SUPPORTED_AS_DIRECTIONAL_EXTERNAL_REPLICATION",
            str(gates.loc["C3", "Assessment"]),
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    checks = validate()
    output = AUDIT_DIR / f"Claim_Aligned_Robustness_Validation_{VERSION}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(output, index=False)
    failed = [row for row in checks if not row["Passed"]]
    print(f"claim-aligned validation: checks={len(checks)}, failed={len(failed)}")
    print(output)
    if failed:
        for row in failed:
            print(f"FAILED: {row['Check']} | {row['Detail']}")
        if args.fail_on_error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
