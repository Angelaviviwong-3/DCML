#!/usr/bin/env python3
"""Postprocess the completed 20260728 DML omitted-confounding analysis.

This script does not refit nuisance models or repeat the bootstrap. It preserves
the completed source results, identifies non-finite benchmark boundaries, and
creates publication-facing focal summaries without presenting clipped
``cf_d = 1`` bounds as large finite adjusted effects.
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


SOURCE_VERSION = "20260728"
OUTPUT_VERSION = "20260729"
CAUSAL_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = _dcml_paths.workspace_path(CAUSAL_ROOT, "Unified_Visualization/output")

DATASETS = {
    "suning": "Suning",
    "amazon_appliances": "Amazon Appliances",
    "amazon_beauty": "Amazon Beauty",
}

FOCAL = {
    "suning": [
        ("click", "T_con_soc", "Social proof: Click"),
        ("purchase", "T_con_soc", "Social proof: Purchase"),
        ("click", "T_int_sem", "Category alignment: Click"),
        ("purchase", "T_int_sem", "Category alignment: Purchase"),
    ],
    "amazon_appliances": [
        ("purchase", "T_con_soc", "Social proof"),
        ("purchase", "T_con_rat", "Rating valence"),
        ("purchase", "T_int_sem", "Category alignment"),
    ],
    "amazon_beauty": [("purchase", "T_con_soc", "Social proof")],
}


def locate_source(dataset: str, source_root: Path | None) -> Path:
    basename = f"{dataset}_DML_OVB_Sensitivity_{SOURCE_VERSION}.csv"
    canonical = (
        _dcml_paths.workspace_path(CAUSAL_ROOT, 'processed_data')
        / dataset
        / "Final_Causal_Output"
        / basename
    )
    if canonical.is_file():
        return canonical
    if source_root is not None:
        candidates = sorted(source_root.resolve().rglob(basename))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RuntimeError(f"Multiple source files found for {dataset}: {candidates}")
    raise FileNotFoundError(canonical)


def original_direction_preserved(theta: float, adjusted: float) -> bool | float:
    if not np.isfinite(theta) or not np.isfinite(adjusted) or theta == 0:
        return np.nan
    return bool(np.sign(theta) == np.sign(adjusted))


def ci_preserves_direction(theta: float, lower: float, upper: float) -> bool | float:
    if not all(np.isfinite(value) for value in [theta, lower, upper]) or theta == 0:
        return np.nan
    return bool(lower > 0) if theta > 0 else bool(upper < 0)


def postprocess(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Dataset",
        "Stage",
        "Specification",
        "Treatment",
        "Benchmark_Multiplier",
        "Assumed_cf_y",
        "Assumed_cf_d",
        "DML_ATE",
        "Bias_Adjusted_ATE_Toward_Zero",
        "Bias_Adjusted_CI_Lower",
        "Bias_Adjusted_CI_Upper",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"OVB source is missing columns: {missing}")

    out = frame.copy()
    for column in [
        "Benchmark_Multiplier",
        "Assumed_cf_y",
        "Assumed_cf_d",
        "DML_ATE",
        "Bias_Adjusted_ATE_Toward_Zero",
        "Bias_Adjusted_CI_Lower",
        "Bias_Adjusted_CI_Upper",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["Source_Bias_Adjusted_ATE"] = out["Bias_Adjusted_ATE_Toward_Zero"]
    out["Source_Bias_Adjusted_CI_Lower"] = out["Bias_Adjusted_CI_Lower"]
    out["Source_Bias_Adjusted_CI_Upper"] = out["Bias_Adjusted_CI_Upper"]

    reference = out["Benchmark_Multiplier"].eq(0)
    boundary = (~reference) & out["Assumed_cf_d"].ge(1.0 - 1e-8)
    zero_outcome_gain = boundary & out["Assumed_cf_y"].le(1e-12)
    unbounded = boundary & ~zero_outcome_gain

    out["Bound_Status"] = "FINITE"
    out.loc[reference, "Bound_Status"] = "REFERENCE"
    out.loc[zero_outcome_gain, "Bound_Status"] = "NONINFORMATIVE_ZERO_OUTCOME_GAIN"
    out.loc[unbounded, "Bound_Status"] = "UNBOUNDED_CF_D_BOUNDARY"

    invalid_numeric_bound = zero_outcome_gain | unbounded
    out["Adversarial_Bias_Adjusted_ATE"] = out["Source_Bias_Adjusted_ATE"]
    out["Adversarial_Bias_Adjusted_CI_Lower"] = out["Source_Bias_Adjusted_CI_Lower"]
    out["Adversarial_Bias_Adjusted_CI_Upper"] = out["Source_Bias_Adjusted_CI_Upper"]
    out.loc[
        invalid_numeric_bound,
        [
            "Adversarial_Bias_Adjusted_ATE",
            "Adversarial_Bias_Adjusted_CI_Lower",
            "Adversarial_Bias_Adjusted_CI_Upper",
        ],
    ] = np.nan

    out["Point_Direction_Preserved"] = [
        original_direction_preserved(theta, adjusted)
        for theta, adjusted in zip(
            out["DML_ATE"], out["Adversarial_Bias_Adjusted_ATE"], strict=True
        )
    ]
    out["CI_Excludes_Zero_In_Original_Direction"] = [
        ci_preserves_direction(theta, lower, upper)
        for theta, lower, upper in zip(
            out["DML_ATE"],
            out["Adversarial_Bias_Adjusted_CI_Lower"],
            out["Adversarial_Bias_Adjusted_CI_Upper"],
            strict=True,
        )
    ]

    conclusions = np.full(len(out), "FINITE_BOUND_CROSSES_ZERO", dtype=object)
    conclusions[reference.to_numpy()] = "UNADJUSTED_REFERENCE"
    conclusions[zero_outcome_gain.to_numpy()] = "BENCHMARK_NOT_INFORMATIVE_FOR_BIAS_BOUND"
    conclusions[unbounded.to_numpy()] = "ADVERSARIAL_BOUND_NOT_FINITE"
    finite_nonreference = (~reference & ~invalid_numeric_bound).to_numpy()
    preserved = out["Point_Direction_Preserved"].map(lambda value: value is True).to_numpy()
    conclusions[finite_nonreference & preserved] = "POINT_DIRECTION_PRESERVED"
    out["Benchmark_Conclusion"] = conclusions
    out["Source_Version"] = SOURCE_VERSION
    out["Postprocess_Version"] = OUTPUT_VERSION
    out["Uncertainty_Scope"] = (
        "Completed source analysis uses a user-cluster wild bootstrap; point sensitivity "
        "bounds are retained, while boundary rows are not reported as finite effects."
    )
    return out


def select_focal(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    aggregate = frame[frame["Specification"].astype(str).eq("Aggregate")].copy()
    aggregate["Stage"] = aggregate["Stage"].astype(str).str.lower()
    for stage, treatment, label in FOCAL[dataset]:
        rows = aggregate[
            aggregate["Stage"].eq(stage)
            & aggregate["Treatment"].astype(str).eq(treatment)
        ].copy()
        if rows.empty:
            raise RuntimeError(f"Missing focal OVB row: {dataset}/{stage}/{treatment}")
        rows["Dataset_Key"] = dataset
        rows["Dataset_Label"] = DATASETS[dataset]
        rows["Focal_Label"] = label
        selected.append(rows)
    return pd.concat(selected, ignore_index=True)


def compact_focal(focal: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "Dataset_Key",
        "Dataset_Label",
        "Stage",
        "Treatment",
        "Focal_Label",
        "Strongest_Observed_Benchmark",
        "DML_ATE",
        "Equal_Strength_CF_To_Move_Point_Estimate_To_Zero",
        "Approx_Equal_Strength_CF_To_Move_Unadjusted_95CI_To_Zero",
    ]
    rows: list[dict] = []
    for key, group in focal.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, key, strict=True))
        row["Point_RV_Percent"] = 100.0 * float(
            row["Equal_Strength_CF_To_Move_Point_Estimate_To_Zero"]
        )
        row["Approx_CI_RV_Percent_UserCluster"] = 100.0 * float(
            row["Approx_Equal_Strength_CF_To_Move_Unadjusted_95CI_To_Zero"]
        )
        for multiplier in [1, 2, 3]:
            match = group[group["Benchmark_Multiplier"].eq(multiplier)]
            if len(match) != 1:
                raise RuntimeError(f"Expected one {multiplier}x row for {key}")
            value = match.iloc[0]
            row[f"ATE_At_{multiplier}x_Benchmark"] = value[
                "Adversarial_Bias_Adjusted_ATE"
            ]
            row[f"Status_At_{multiplier}x_Benchmark"] = value["Benchmark_Conclusion"]
        rows.append(row)
    return pd.DataFrame(rows)


def run(source_root: Path | None) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    focal_frames: list[pd.DataFrame] = []
    output_paths: list[Path] = []
    source_paths: dict[str, str] = {}
    boundary_counts: dict[str, dict[str, int]] = {}

    for dataset in DATASETS:
        source = locate_source(dataset, source_root)
        source_paths[dataset] = str(source)
        processed = postprocess(pd.read_csv(source, low_memory=False))
        output = (
            _dcml_paths.workspace_path(CAUSAL_ROOT, 'processed_data')
            / dataset
            / "Final_Causal_Output"
            / f"{dataset}_DML_OVB_Sensitivity_{OUTPUT_VERSION}.csv"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(output, index=False)
        output_paths.append(output)
        focal_frames.append(select_focal(dataset, processed))
        boundary_counts[dataset] = {
            key: int(value)
            for key, value in processed["Bound_Status"].value_counts().to_dict().items()
        }

    focal = pd.concat(focal_frames, ignore_index=True).sort_values(
        ["Dataset_Key", "Focal_Label", "Benchmark_Multiplier"]
    )
    focal_path = OUTPUT_DIR / f"DML_OVB_Focal_Bounds_{OUTPUT_VERSION}.csv"
    focal.to_csv(focal_path, index=False)
    compact_path = OUTPUT_DIR / f"Table_DML_OVB_Focal_Summary_{OUTPUT_VERSION}.csv"
    compact_focal(focal).to_csv(compact_path, index=False)
    output_paths.extend([focal_path, compact_path])

    manifest_path = (
        _dcml_paths.workspace_path(CAUSAL_ROOT, 'processed_data')
        / "audit"
        / f"DML_OVB_Postprocess_Manifest_{OUTPUT_VERSION}.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "source_version": SOURCE_VERSION,
                "postprocess_version": OUTPUT_VERSION,
                "source_files": source_paths,
                "boundary_counts": boundary_counts,
                "method_change": (
                    "No model or bootstrap rerun. Rows at the cf_d=1 boundary are marked "
                    "non-finite or non-informative instead of reported as large finite ATEs."
                ),
                "claim_boundary": (
                    "The outputs quantify sensitivity to omitted confounding under benchmarked "
                    "causal-ML assumptions; they do not establish exchangeability."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_paths.append(manifest_path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional archive root containing the completed 20260728 source CSVs.",
    )
    args = parser.parse_args()
    for path in run(args.source_root):
        print(path)


if __name__ == "__main__":
    main()
