#!/usr/bin/env python3
"""Summarize five-seed recommendation results with publication-safe uncertainty labels."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CAUSAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_utils import ensure_dir, find_causal_root


OUTPUT_VERSION = "20260720"
SOURCE_VERSION = "20260719"
DATASETS = ("suning", "amazon_appliances", "amazon_beauty")
METRICS = ("HR@10", "NDCG@10", "HR@20", "NDCG@20")


def summarize_dataset(root: Path, dataset: str) -> pd.DataFrame:
    output_dir = ensure_dir(_dcml_paths.workspace_path(root, 'results_of_comparison') / dataset / "dcml_performance")
    source = output_dir / f"Recommendation_Training_Seed_Runs_{SOURCE_VERSION}.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    runs = pd.read_csv(source)
    required = {"Dataset", "Seed_Offset", "Stage", "Model", "Actual_Training_Seed", *METRICS}
    missing = sorted(required.difference(runs.columns))
    if missing:
        raise RuntimeError(f"Missing columns in {source}: {missing}")

    rows: list[dict] = []
    for (stage, model), group in runs.groupby(["Stage", "Model"], sort=True):
        group = group.sort_values("Seed_Offset")
        n = int(len(group))
        if n < 2:
            raise RuntimeError(f"At least two independent runs are required for {dataset}/{stage}/{model}")
        row = {
            "Dataset": dataset,
            "Stage": stage,
            "Model": model,
            "Independent_Training_Runs": n,
            "Seed_Offsets": "|".join(group["Seed_Offset"].astype(str)),
            "Actual_Training_Seeds": "|".join(group["Actual_Training_Seed"].astype(str)),
            "Primary_Reporting_Convention": "mean_plus_or_minus_sample_SD",
            "Interval_Interpretation": (
                "t interval for the mean across training runs; descriptive algorithmic variability, "
                "not user-sampling or population-generalization uncertainty"
            ),
        }
        critical = float(stats.t.ppf(0.975, df=n - 1))
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) != n:
                raise RuntimeError(f"Non-numeric {metric} values for {dataset}/{stage}/{model}")
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            se = float(sd / np.sqrt(n))
            margin = critical * se
            row.update(
                {
                    f"{metric}_Mean": mean,
                    f"{metric}_Sample_SD": sd,
                    f"{metric}_Mean_Plus_Minus_SD": f"{mean:.4f} +/- {sd:.4f}",
                    f"{metric}_Min": float(values.min()),
                    f"{metric}_Max": float(values.max()),
                    f"{metric}_SE_of_Seed_Mean": se,
                    f"{metric}_T_CI_Lower_95": mean - margin,
                    f"{metric}_T_CI_Upper_95": mean + margin,
                    f"{metric}_T_Critical_DF_{n - 1}": critical,
                }
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    output = output_dir / f"Recommendation_Training_Seed_Summary_Academic_{OUTPUT_VERSION}.csv"
    manifest = output_dir / f"Recommendation_Training_Seed_Summary_Academic_{OUTPUT_VERSION}.json"
    summary.to_csv(output, index=False)
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset,
                "output_version": OUTPUT_VERSION,
                "source_version": SOURCE_VERSION,
                "source_runs": str(source),
                "primary_table_reporting": (
                    "Report mean +/- sample standard deviation over five independent training runs."
                ),
                "optional_appendix_reporting": (
                    "Report min-max and the Student-t interval using df=n-1. The t interval summarizes "
                    "variation in the mean across the chosen training seeds only."
                ),
                "prohibited_interpretation": (
                    "Do not interpret across-seed intervals as confidence intervals for new users, "
                    "new items, new datasets, or population generalization."
                ),
                "recommended_table_note": (
                    "Values are mean +/- sample SD over five independent runs with fixed data splits "
                    "and candidate sets; higher is better."
                ),
                "result_file": str(output),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    args = parser.parse_args()
    root = find_causal_root(__file__)
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    summaries = [summarize_dataset(root, dataset) for dataset in selected]
    if args.dataset == "all":
        combined = pd.concat(summaries, ignore_index=True)
        combined_path = ensure_dir(_dcml_paths.workspace_path(root, 'results_of_comparison')) / (
            f"Recommendation_Training_Seed_Summary_All_Datasets_{OUTPUT_VERSION}.csv"
        )
        combined.to_csv(combined_path, index=False)
        print(f"Wrote {combined_path}")


if __name__ == "__main__":
    main()
