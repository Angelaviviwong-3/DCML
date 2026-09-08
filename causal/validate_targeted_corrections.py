#!/usr/bin/env python3
"""Validate the targeted 20260720 support, seed-summary, and decision outputs."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[0]))
import project_paths as _dcml_paths


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CAUSAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_utils import ensure_dir, find_causal_root


VERSION = "20260720"
DATASETS = ("suning", "amazon_appliances", "amazon_beauty")


def add(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"Check": name, "Passed": bool(passed), "Detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    root = find_causal_root(__file__)
    checks: list[dict] = []

    beauty_dir = _dcml_paths.workspace_path(root, 'processed_data') / "amazon_beauty" / "Final_Causal_Output"
    beauty_result = beauty_dir / (
        f"amazon_beauty_Causal_Baseline_Comparison_Support_Consistent_{VERSION}.csv"
    )
    beauty_manifest = beauty_dir / f"amazon_beauty_Causal_Baseline_Support_Manifest_{VERSION}.json"
    add(checks, "beauty_result_exists", beauty_result.exists(), str(beauty_result))
    add(checks, "beauty_manifest_exists", beauty_manifest.exists(), str(beauty_manifest))
    if beauty_result.exists() and beauty_manifest.exists():
        result = pd.read_csv(beauty_result)
        with open(beauty_manifest, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        add(
            checks,
            "beauty_support_sizes_5_9",
            manifest.get("aggregate_estimable") == 5 and manifest.get("disaggregate_estimable") == 9,
            f"aggregate={manifest.get('aggregate_estimable')} disaggregate={manifest.get('disaggregate_estimable')}",
        )
        add(
            checks,
            "beauty_no_publication_T_int_sem",
            not result["Component"].astype(str).eq("T_int_sem").any(),
            "publication-ready rows must exclude T_int_sem",
        )
        add(
            checks,
            "beauty_no_history_complete_joint_spec",
            not result["Model_Specification"].astype(str).str.startswith("History_complete_").any(),
            "joint 6/10 history-complete specifications are not publication eligible",
        )

    for dataset in DATASETS:
        path = _dcml_paths.workspace_path(root, 'results_of_comparison') / dataset / "dcml_performance" / (
            f"Recommendation_Training_Seed_Summary_Academic_{VERSION}.csv"
        )
        add(checks, f"{dataset}_academic_seed_summary_exists", path.exists(), str(path))
        if path.exists():
            frame = pd.read_csv(path)
            add(
                checks,
                f"{dataset}_five_independent_runs",
                pd.to_numeric(frame["Independent_Training_Runs"], errors="coerce").eq(5).all(),
                f"rows={len(frame)}",
            )
            critical_columns = [column for column in frame if column.endswith("_T_Critical_DF_4")]
            critical_ok = bool(critical_columns) and all(
                np.allclose(pd.to_numeric(frame[column], errors="coerce"), 2.7764451051977987)
                for column in critical_columns
            )
            add(
                checks,
                f"{dataset}_student_t_df4",
                critical_ok,
                f"critical_columns={len(critical_columns)}",
            )

    suning_dir = _dcml_paths.workspace_path(root, 'processed_data') / "suning" / "Final_Causal_Output"
    decision_path = suning_dir / f"suning_IQR_Scaled_Effect_Profiles_{VERSION}.csv"
    add(checks, "suning_decision_profiles_exist", decision_path.exists(), str(decision_path))
    if decision_path.exists():
        decision = pd.read_csv(decision_path)
        formal_ope = decision["Formal_OPE"].astype(str).str.lower().eq("true")
        add(checks, "suning_not_formal_ope", not formal_ope.any(), "Formal_OPE must be false")
        sample_match = decision["Analysis_Sample_N_Matches_Estimation_N"].astype(str).str.lower().eq("true")
        add(
            checks,
            "suning_iqr_sample_matches_estimation",
            sample_match.all(),
            f"matched={int(sample_match.sum())}/{len(sample_match)}",
        )

    table = pd.DataFrame(checks)
    failed = table[~table["Passed"]]
    audit_dir = ensure_dir(_dcml_paths.workspace_path(root, 'processed_data') / "audit")
    csv_path = audit_dir / f"targeted_corrections_validation_{VERSION}.csv"
    json_path = audit_dir / f"targeted_corrections_validation_{VERSION}.json"
    table.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": VERSION,
                "status": "PASS" if failed.empty else "FAIL",
                "checks": int(len(table)),
                "failed_checks": int(len(failed)),
                "csv": str(csv_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"targeted corrections validation: failed_checks={len(failed)}")
    if args.fail_on_error and not failed.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
