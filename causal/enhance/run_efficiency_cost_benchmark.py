#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize and extend DCML efficiency/cost benchmarks.

The default mode uses existing latency measurements when available and adds a
lightweight cached-treatment online scoring benchmark.  It does not call an MLLM
unless you run a separate offline extraction benchmark.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from enhance_common import (  # noqa: E402
    CAUSAL_ROOT,
    OUTPUT_DIR,
    VERSION,
    ensure_output_dir,
    first_existing,
    read_table,
    write_json,
)


SCALAR_CANDIDATES = {
    "suning": [
        _dcml_paths.workspace_path(CAUSAL_ROOT, 'processed_data/suning/item_multimodal_scalars/checkpoint_scalars_full_3.csv'),
        _dcml_paths.workspace_path(CAUSAL_ROOT, 'processed_data/suning/item_multimodal_scalars/item_multimodal_scalars_full_3.parquet'),
    ],
    "amazon_appliances": [
        _dcml_paths.workspace_path(CAUSAL_ROOT, 'processed_data/amazon_appliances/item_multimodal_scalars/amazon_appliances_item_multimodal_scalars_merged_full_2.csv'),
        _dcml_paths.workspace_path(CAUSAL_ROOT, 'processed_data/amazon_appliances/item_multimodal_scalars/amazon_appliances_item_multimodal_scalars_merged_full_2.parquet'),
    ],
    "amazon_beauty": [
        _dcml_paths.workspace_path(CAUSAL_ROOT, 'processed_data/amazon_beauty/item_multimodal_scalars/amazon_beauty_item_multimodal_scalars_merged_full.csv'),
        _dcml_paths.workspace_path(CAUSAL_ROOT, 'processed_data/amazon_beauty/item_multimodal_scalars/amazon_beauty_item_multimodal_scalars_merged_full.parquet'),
    ],
}

TREATMENT_ALIASES = {
    "T_con_mkt": ["T_con_mkt_calib", "T_con_mkt_calibrated", "T_con_mkt"],
    "T_int_vis": ["T_int_vis_calib", "T_int_vis_calibrated", "T_int_vis"],
    "T_int_fac": ["T_int_fac_calib", "T_int_fac_calibrated", "T_int_fac"],
    "atomic_a_pri": ["atomic_a_pri_calib", "atomic_a_pri_calibrated", "atomic_a_pri", "atomic_mkt_price_calib"],
    "atomic_a_gft": ["atomic_a_gft_calib", "atomic_a_gft_calibrated", "atomic_a_gft"],
    "atomic_a_sub": ["atomic_a_sub_calib", "atomic_a_sub_calibrated", "atomic_a_sub"],
    "atomic_a_urg": ["atomic_a_urg_calib", "atomic_a_urg_calibrated", "atomic_a_urg"],
    "atomic_a_spec": ["atomic_a_spec_calib", "atomic_a_spec_calibrated", "atomic_a_spec"],
    "atomic_a_str": ["atomic_a_str_calib", "atomic_a_str_calibrated", "atomic_a_str"],
}


def resolve_scalar_path(dataset: str) -> Path:
    path = first_existing(SCALAR_CANDIDATES[dataset])
    if path is None:
        raise FileNotFoundError(f"No scalar treatment table found for {dataset}")
    return path


def resolve_treatment_columns(frame: pd.DataFrame) -> list[str]:
    cols = []
    for aliases in TREATMENT_ALIASES.values():
        hit = next((column for column in aliases if column in frame.columns), None)
        if hit is not None:
            cols.append(hit)
    if not cols:
        raise KeyError("No cached MCRE treatment columns found in scalar table")
    return cols


def summarize_existing_latency(gpu_hour_cost_usd: float | None) -> pd.DataFrame:
    path = first_existing(
        [
            OUTPUT_DIR / f"dcml_latency_benchmark_{VERSION}.csv",
            _dcml_paths.workspace_path(CAUSAL_ROOT, "Unified_Visualization/output/dcml_latency_benchmark.csv"),
        ]
    )
    if path is None:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    rows = []
    for row in frame.itertuples(index=False):
        mean_ms = float(row.mean_ms)
        stage = str(row.stage)
        record = {
            "Benchmark": "existing_latency_file",
            "Stage": stage,
            "Dataset": "suning_or_original_benchmark_sample",
            "Candidates_Per_Request": np.nan,
            "Mean_ms": mean_ms,
            "Median_ms": np.nan,
            "P95_ms": np.nan,
            "Std_ms": float(getattr(row, "std_ms", np.nan)),
            "Measurements": int(getattr(row, "num_measurements", 0)),
            "Source_File": str(path),
            "Note": str(getattr(row, "notes", "")),
        }
        if "LMM" in stage or "MLLM" in stage:
            items_per_hour = 3600.0 / (mean_ms / 1000.0) if mean_ms > 0 else np.nan
            record["Items_Per_GPU_Hour"] = items_per_hour
            record["Estimated_USD_Per_1000_Items"] = (
                1000.0 / items_per_hour * gpu_hour_cost_usd if gpu_hour_cost_usd and items_per_hour > 0 else np.nan
            )
        else:
            record["Requests_Per_Second"] = 1000.0 / mean_ms if mean_ms > 0 else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def online_cached_benchmark(dataset: str, requests: int, k_values: list[int], seed: int) -> pd.DataFrame:
    scalar_path = resolve_scalar_path(dataset)
    frame = read_table(scalar_path)
    columns = resolve_treatment_columns(frame)
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if matrix.shape[0] < 2:
        raise RuntimeError(f"{dataset}: not enough items for online benchmark")
    rng = np.random.default_rng(seed)
    coef = rng.normal(loc=0.0, scale=0.01, size=matrix.shape[1])
    rows = []
    for k in k_values:
        kk = min(k, matrix.shape[0])
        latencies = []
        for _ in range(requests):
            idx = rng.choice(matrix.shape[0], size=kk, replace=False)
            base = rng.normal(size=kk)
            start = time.perf_counter()
            causal_lift = matrix[idx] @ coef
            _ = base + causal_lift
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)
        arr = np.asarray(latencies, dtype=float)
        rows.append(
            {
                "Benchmark": "cached_treatment_online_scoring",
                "Stage": "online_cached_treatment_score_per_request",
                "Dataset": dataset,
                "Candidates_Per_Request": int(kk),
                "Mean_ms": float(arr.mean()),
                "Median_ms": float(np.median(arr)),
                "P95_ms": float(np.percentile(arr, 95)),
                "Std_ms": float(arr.std(ddof=1)),
                "Measurements": int(len(arr)),
                "Source_File": str(scalar_path),
                "Treatment_Columns": "|".join(columns),
                "Note": "vectorized cached MCRE treatment dot product plus base relevance score; no live MLLM call",
            }
        )
    return pd.DataFrame(rows)


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    rows = []
    for row in results.itertuples(index=False):
        mean_ms = float(row.Mean_ms)
        stage = str(row.Stage)
        if "offline" in stage.lower() or "inference_per_item" in stage.lower():
            managerial = (
                "Offline item-level MCRE cost can be amortized through caching; it is not part of the real-time request path."
            )
        elif "online" in stage.lower():
            managerial = (
                "Cached treatment scoring is lightweight enough to be used as a validation-gated reranking layer."
            )
        else:
            managerial = "Auxiliary computational benchmark."
        rows.append(
            {
                "Stage": stage,
                "Dataset": getattr(row, "Dataset", ""),
                "Candidates_Per_Request": getattr(row, "Candidates_Per_Request", np.nan),
                "Mean_ms": mean_ms,
                "P95_ms": getattr(row, "P95_ms", np.nan),
                "Approx_Throughput_Per_Second": 1000.0 / mean_ms if mean_ms > 0 else np.nan,
                "Managerial_Interpretation": managerial,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["summarize-existing", "online-cache-benchmark", "all"], default="all")
    parser.add_argument("--dataset", choices=["suning", "amazon_appliances", "amazon_beauty", "all"], default="all")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--k-values", type=str, default="50,100,200,500")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--gpu-hour-cost-usd", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dir()
    frames = []
    if args.mode in {"summarize-existing", "all"}:
        existing = summarize_existing_latency(args.gpu_hour_cost_usd)
        if not existing.empty:
            frames.append(existing)
    if args.mode in {"online-cache-benchmark", "all"}:
        datasets = ["suning", "amazon_appliances", "amazon_beauty"] if args.dataset == "all" else [args.dataset]
        k_values = [int(v.strip()) for v in args.k_values.split(",") if v.strip()]
        for dataset in datasets:
            frames.append(online_cached_benchmark(dataset, args.requests, k_values, args.seed))
    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result_path = OUTPUT_DIR / f"Efficiency_Cost_Benchmark_{VERSION}.csv"
    summary_path = OUTPUT_DIR / f"Efficiency_Cost_Summary_{VERSION}.csv"
    results.to_csv(result_path, index=False)
    summary_table(results).to_csv(summary_path, index=False)
    write_json(
        {
            "version": VERSION,
            "experiment": "DCML efficiency and cost benchmark",
            "mode": args.mode,
            "dataset": args.dataset,
            "requests": args.requests,
            "k_values": args.k_values,
            "gpu_hour_cost_usd": args.gpu_hour_cost_usd,
            "python": sys.version,
            "platform": platform.platform(),
            "output_files": [str(result_path), str(summary_path)],
            "interpretation_boundary": (
                "The default online benchmark evaluates cached treatment scoring only. "
                "It supports deployment-cost discussion but does not replace full MLLM extraction latency measurement."
            ),
        },
        OUTPUT_DIR / f"Efficiency_Cost_Manifest_{VERSION}.json",
    )
    print(f"Efficiency benchmark written to {result_path}")
    print(f"Efficiency summary written to {summary_path}")


if __name__ == "__main__":
    main()
