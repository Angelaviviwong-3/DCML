#!/usr/bin/env python3
"""Evaluate recommendation metrics on deterministic subsets of the 99 negatives."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_CORE = Path(__file__).resolve().parents[1] / "A_base"
if not (BASELINE_CORE / "baseline_utils.py").is_file():
    raise FileNotFoundError(f"Missing 20260717 baseline utilities: {BASELINE_CORE}")
sys.path.insert(0, str(BASELINE_CORE))

from baseline_utils import (  # noqa: E402
    BASE_SEED,
    SCRIPT_VERSION,
    STAGES,
    candidate_items,
    find_causal_root,
    load_candidates,
    performance_dir,
    prediction_path,
    read_json,
    setup_logger,
    sha256_file,
    write_json,
)

MODELS = ("MBGCN", "KMCLR", "MGAT", "MICRO", "SLMRec", "DICE", "CIRS", "MGCE", "DCML (Ours)")
METRICS = ("HR@10", "NDCG@10", "HR@20", "NDCG@20")


def parse_int_grid(text: str, maximum: int | None = None) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in text.split(",") if part.strip()}))
    if not values or any(value <= 0 or (maximum is not None and value > maximum) for value in values):
        raise argparse.ArgumentTypeError("invalid positive integer grid")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--negative-counts",
        type=lambda text: parse_int_grid(text, 99),
        default=(49, 79, 99),
        help="Comma-separated numbers of negatives retained from each audited 99-negative set.",
    )
    parser.add_argument("--replicate-seeds", type=lambda text: parse_int_grid(text), default=(1, 2, 3, 4, 5))
    return parser.parse_args()


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("None of these paths exist: " + "; ".join(map(str, paths)))


def score_file(root: Path, model: str, stage: str) -> Path:
    filename = f"DCML_{stage}_predictions_{SCRIPT_VERSION}.json" if model == "DCML (Ours)" else f"{model}_{stage}_predictions_{SCRIPT_VERSION}.json"
    primary = performance_dir(root) / filename if model == "DCML (Ours)" else prediction_path(root, model, stage)
    return first_existing([primary, _dcml_paths.workspace_path(root.parent, f"{SCRIPT_VERSION}_results", filename)])


def expected_metrics(scores: np.ndarray, target_index: int) -> dict[str, float]:
    target = float(scores[target_index])
    tied = np.isclose(scores, target, rtol=1e-12, atol=1e-12)
    better = int(np.sum((scores > target) & ~tied))
    tie_size = int(np.sum(tied))
    positions = np.arange(better + 1, better + tie_size + 1, dtype=float)
    output = {}
    for k in (10, 20):
        inside = positions <= k
        output[f"HR@{k}"] = float(np.mean(inside))
        output[f"NDCG@{k}"] = float(np.mean(np.where(inside, 1.0 / np.log2(positions + 1.0), 0.0)))
    return output


def deterministic_negative_indices(user: str, stage: str, count: int, replicate: int, available: int) -> np.ndarray:
    if count == available:
        return np.arange(available, dtype=int)
    user_number = int(user) if str(user).isdigit() else sum((index + 1) * ord(char) for index, char in enumerate(str(user)))
    seed = BASE_SEED + STAGES.index(stage) * 1_000_000 + replicate * 100_000 + count * 1000 + user_number
    return np.sort(np.random.default_rng(seed).choice(available, size=count, replace=False))


def aggregate(values: list[float], metric: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return {
        metric: mean,
        f"{metric}_User_SE": se,
        f"{metric}_User_CI_Lower": max(0.0, mean - 1.96 * se),
        f"{metric}_User_CI_Upper": min(1.0, mean + 1.96 * se),
    }


def main() -> None:
    args = parse_args()
    root = find_causal_root(__file__)
    output = performance_dir(root)
    logger = setup_logger(root, "recommendation_candidate_sensitivity")
    rows = []

    for stage in STAGES:
        candidates = load_candidates(root, stage)
        for model in MODELS:
            path = score_file(root, model, stage)
            predictions = read_json(path)
            if set(map(str, predictions)) != set(map(str, candidates)):
                raise RuntimeError(f"{model}/{stage} user set does not match the audited candidates")
            for negative_count in args.negative_counts:
                replicates = (args.replicate_seeds[0],) if negative_count == 99 else args.replicate_seeds
                for replicate in replicates:
                    metric_values: dict[str, list[float]] = defaultdict(list)
                    for user, candidate in candidates.items():
                        all_ids = candidate_items(candidate)
                        negatives, target = all_ids[:-1], all_ids[-1]
                        if len(negatives) != 99:
                            raise RuntimeError(f"Expected 99 negatives for {stage}/{user}, found {len(negatives)}")
                        selected_indices = deterministic_negative_indices(str(user), stage, negative_count, replicate, len(negatives))
                        selected_ids = [negatives[index] for index in selected_indices] + [target]
                        scores_by_item = predictions[str(user)]
                        if not {str(item) for item in selected_ids}.issubset(scores_by_item):
                            raise RuntimeError(f"Incomplete scores for {model}/{stage}/{user}")
                        scores = np.asarray([float(scores_by_item[str(item)]) for item in selected_ids], dtype=float)
                        metrics = expected_metrics(scores, len(selected_ids) - 1)
                        for metric, value in metrics.items():
                            metric_values[metric].append(value)
                    row = {
                        "Stage": stage,
                        "Model": model,
                        "Negative_Count": negative_count,
                        "Candidate_Set_Size": negative_count + 1,
                        "Subset_Replicate": replicate,
                        "Evaluated_Users": len(candidates),
                        "Sensitivity_Design": "deterministic nested subsamples of each audited 99-negative candidate set",
                        "Score_File": str(path),
                        "Score_File_SHA256": sha256_file(path),
                    }
                    for metric in METRICS:
                        row.update(aggregate(metric_values[metric], metric))
                    rows.append(row)
            logger.info("Completed candidate sensitivity for %s/%s", model, stage)

    result = pd.DataFrame(rows)
    result.to_csv(output / f"Recommendation_Candidate_Sensitivity_{SCRIPT_VERSION}.csv", index=False)
    across_replicates = result.groupby(
        ["Stage", "Model", "Negative_Count", "Candidate_Set_Size"]
    )[list(METRICS)].agg(["mean", "std", "min", "max"]).reset_index()
    across_replicates.columns = ["_".join(filter(None, map(str, column))) for column in across_replicates.columns]
    across_replicates.to_csv(output / f"Recommendation_Candidate_Sensitivity_Summary_{SCRIPT_VERSION}.csv", index=False)
    write_json(
        {
            "script_version": SCRIPT_VERSION,
            "models": list(MODELS),
            "stages": list(STAGES),
            "negative_counts": list(args.negative_counts),
            "replicate_seeds": list(args.replicate_seeds),
            "expected_rows": 297,
            "actual_rows": int(len(result)),
            "all_score_files_hashed": bool(result["Score_File_SHA256"].notna().all()),
        },
        output / f"Recommendation_Candidate_Sensitivity_Metadata_{SCRIPT_VERSION}.json",
        indent=2,
    )
    logger.info("Wrote %d candidate-sensitivity rows to %s", len(result), output)


if __name__ == "__main__":
    main()
