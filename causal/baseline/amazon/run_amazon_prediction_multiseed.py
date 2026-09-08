#!/usr/bin/env python3
"""Rerun the Amazon Purchase-only recommenders under multiple training seeds and summarize metrics."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


import argparse
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_CORE = Path(__file__).resolve().parent / "A_base"
if not (BASELINE_CORE / "amazon_baseline_utils.py").is_file():
    raise FileNotFoundError(f"Missing 20260718 baseline utilities: {BASELINE_CORE}")
sys.path.insert(0, str(BASELINE_CORE))

from amazon_baseline_utils import (  # noqa: E402
    SCRIPT_VERSION,
    STAGES,
    baseline_data_dir,
    candidate_items,
    ensure_dir,
    find_causal_root,
    load_candidates,
    performance_dir,
    prediction_metadata_path,
    prediction_path,
    read_json,
    setup_logger,
)

MODELS = ("MBGCN", "KMCLR", "MGAT", "MICRO", "SLMRec", "DICE", "CIRS", "MGCE", "DCML (Ours)")
METRICS = ("HR@10", "NDCG@10", "HR@20", "NDCG@20")
DEFAULT_OFFSETS = (100_000, 200_000, 300_000, 400_000, 0)


def parse_offsets(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("seed offsets must be a nonempty comma-separated list without duplicates")
    if 0 not in values:
        raise argparse.ArgumentTypeError("seed offsets must include 0 so the primary run can be restored")
    return tuple(value for value in values if value != 0) + (0,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-offsets",
        type=parse_offsets,
        default=DEFAULT_OFFSETS,
        help="Comma-separated offsets. Offset 0 is always run last and remains the primary output.",
    )
    parser.add_argument("--resolution", choices=("Macro",), default="Macro")
    parser.add_argument("--models", nargs="+", choices=MODELS[:-1], default=list(MODELS[:-1]))
    return parser.parse_args()


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


def evaluate_bundle(candidates: dict, predictions: dict) -> dict[str, float]:
    if set(map(str, candidates)) != set(map(str, predictions)):
        raise RuntimeError("prediction users do not match candidate users")
    values: dict[str, list[float]] = defaultdict(list)
    for user, candidate in candidates.items():
        ids = candidate_items(candidate)
        scores_by_item = predictions[str(user)]
        if set(scores_by_item) != {str(item) for item in ids}:
            raise RuntimeError(f"candidate item mismatch for user {user}")
        scores = np.asarray([float(scores_by_item[str(item)]) for item in ids], dtype=float)
        metrics = expected_metrics(scores, len(ids) - 1)
        for metric, value in metrics.items():
            values[metric].append(value)
    output = {"Evaluated_Users": len(candidates)}
    for metric in METRICS:
        array = np.asarray(values[metric], dtype=float)
        output[metric] = float(array.mean())
        output[f"{metric}_User_SE"] = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return output


def archive_metadata(root: Path, seed_offset: int, models: tuple[str, ...]) -> Path:
    destination = ensure_dir(performance_dir(root) / f"training_seed_metadata_{SCRIPT_VERSION}" / f"offset_{seed_offset}")
    baseline = baseline_data_dir(root)
    performance = performance_dir(root)
    sources = [
        *(prediction_metadata_path(root, model, stage) for model in models for stage in STAGES),
        *(baseline / f"{model}_training_diagnostics_{SCRIPT_VERSION}.json" for model in models),
        performance / f"DCML_Recommender_Metadata_{SCRIPT_VERSION}.json",
        performance / f"DCML_Recommendation_Preflight_{SCRIPT_VERSION}.json",
        performance / f"DCML_Recommender_Training_Diagnostics_{SCRIPT_VERSION}.csv",
        performance / f"DCML_Item_Category_Audit_{SCRIPT_VERSION}.json",
        performance / f"DCML_Recommendation_Validation_Tuning_{SCRIPT_VERSION}.csv",
        performance / f"DCML_Recommendation_Causal_Prior_{SCRIPT_VERSION}.csv",
        performance / f"DCML_Recommendation_Orthogonalization_{SCRIPT_VERSION}.csv",
        performance / f"DCML_Recommendation_Prior_Stability_{SCRIPT_VERSION}.csv",
        performance / f"DCML_User_State_Audit_{SCRIPT_VERSION}.csv",
        *(performance / f"DCML_{stage}_prediction_metadata_{SCRIPT_VERSION}.json" for stage in STAGES),
        performance / f"DCML_Evaluation_Metadata_{SCRIPT_VERSION}.json",
        performance / f"Performance_Results_Summary_Hybrid_{SCRIPT_VERSION}.csv",
    ]
    for source in sources:
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    return destination


def main() -> None:
    args = parse_args()
    if tuple(args.models) != MODELS[:-1]:
        raise ValueError("All eight baselines are required for the publication multiseed comparison")
    root = find_causal_root(__file__)
    logger = setup_logger(root, "amazon_purchase_training_seed_robustness")
    output = performance_dir(root)
    baseline_runner = Path(__file__).resolve().parent / f"run_prediction_baselines_amazon.py"
    dcml_trainer = Path(__file__).resolve().parent / f"step2_train_dcml_recommender_amazon.py"
    dcml_evaluator = Path(__file__).resolve().parent / f"step3_evaluate_dcml_recommender_amazon.py"
    sensitivity_runner = Path(__file__).resolve().parent / f"step5_candidate_sensitivity_amazon.py"
    comparison_runner = Path(__file__).resolve().parent / f"step4_compare_prediction_models_amazon.py"
    rows = []

    for seed_offset in args.seed_offsets:
        logger.info("Starting all-model training run with seed offset %d", seed_offset)
        environment = os.environ.copy()
        environment["AMAZON_TRAINING_SEED_OFFSET"] = str(seed_offset)
        subprocess.run(
            [
                sys.executable,
                str(baseline_runner),
                "--only-train",
            ],
            check=True,
            cwd=root.parent,
            env=environment,
        )
        subprocess.run(
            [sys.executable, str(dcml_trainer)],
            check=True,
            cwd=root.parent,
            env=environment,
        )
        subprocess.run(
            [sys.executable, str(dcml_evaluator), "--resolution", args.resolution],
            check=True,
            cwd=root.parent,
            env=environment,
        )
        for stage in STAGES:
            candidates = load_candidates(root, stage)
            for model in MODELS:
                path = (
                    output / f"DCML_{stage}_predictions_{SCRIPT_VERSION}.json"
                    if model == "DCML (Ours)"
                    else prediction_path(root, model, stage)
                )
                result = evaluate_bundle(candidates, read_json(path))
                metadata = (
                    read_json(output / f"DCML_{stage}_prediction_metadata_{SCRIPT_VERSION}.json")
                    if model == "DCML (Ours)"
                    else read_json(prediction_metadata_path(root, model, stage))
                )
                rows.append(
                    {
                        "Seed_Offset": seed_offset,
                        "Stage": stage,
                        "Model": model,
                        "Actual_Training_Seed": metadata.get("seed"),
                        **result,
                    }
                )
        archive_metadata(root, seed_offset, tuple(args.models))
        pd.DataFrame(rows).to_csv(output / f"Recommendation_Training_Seed_Runs_{SCRIPT_VERSION}.csv", index=False)

    runs = pd.DataFrame(rows)
    summary_rows = []
    for (stage, model), group in runs.groupby(["Stage", "Model"], sort=False):
        row = {
            "Stage": stage,
            "Model": model,
            "Training_Seed_Runs": len(group),
            "Seed_Offsets": "|".join(map(str, sorted(group["Seed_Offset"].astype(int).tolist()))),
            "Actual_Training_Seeds": "|".join(map(str, sorted(group["Actual_Training_Seed"].astype(int).tolist()))),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            se = sd / math.sqrt(len(values)) if len(values) else np.nan
            row[f"{metric}_Mean_Across_Seeds"] = float(values.mean())
            row[f"{metric}_SD_Across_Seeds"] = sd
            row[f"{metric}_SE_Across_Seeds"] = se
            row[f"{metric}_CI_Lower_Across_Seeds"] = max(0.0, float(values.mean() - 1.96 * se))
            row[f"{metric}_CI_Upper_Across_Seeds"] = min(1.0, float(values.mean() + 1.96 * se))
            row[f"{metric}_Min_Across_Seeds"] = float(values.min())
            row[f"{metric}_Max_Across_Seeds"] = float(values.max())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output / f"Recommendation_Training_Seed_Summary_{SCRIPT_VERSION}.csv", index=False)
    final_environment = os.environ.copy()
    final_environment["AMAZON_TRAINING_SEED_OFFSET"] = "0"
    subprocess.run([sys.executable, str(sensitivity_runner)], check=True, cwd=root.parent, env=final_environment)
    subprocess.run([sys.executable, str(comparison_runner)], check=True, cwd=root.parent, env=final_environment)
    logger.info("Completed %d training-seed runs; offset 0 is restored as the primary output", len(args.seed_offsets))


if __name__ == "__main__":
    main()
