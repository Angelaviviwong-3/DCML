#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the three frozen prediction pipelines under five seeds and publish 20260719 summaries."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType


OUTPUT_VERSION = "20260719"
DEFAULT_OFFSETS = "100000,200000,300000,400000,0"
MODELS = ("MBGCN", "KMCLR", "MGAT", "MICRO", "SLMRec", "DICE", "CIRS", "MGCE", "DCML (Ours)")
METRICS = ("HR@10", "NDCG@10", "HR@20", "NDCG@20")
DATASET_CONTRACT = {
    "suning": {
        "source_version": "20260717",
        "environment": None,
        "seed_environment": "SUNING_TRAINING_SEED_OFFSET",
        "utility": "A_base/baseline_utils.py",
        "baseline_runner": "A_base/run_suning_baselines.py",
        "dcml_trainer": "suning/step2_train_dcml_recommender.py",
        "dcml_evaluator": "suning/step3_4_evaluate_dcml.py",
        "sensitivity_runner": "suning/step5_candidate_sensitivity.py",
        "comparison_runner": "suning/step4_compare_all_models.py",
    },
    "amazon_appliances": {
        "source_version": "20260718",
        "environment": "amazon_appliances",
        "seed_environment": "AMAZON_TRAINING_SEED_OFFSET",
        "utility": "amazon/A_base/amazon_baseline_utils.py",
        "baseline_runner": "amazon/run_prediction_baselines_amazon.py",
        "dcml_trainer": "amazon/step2_train_dcml_recommender_amazon.py",
        "dcml_evaluator": "amazon/step3_evaluate_dcml_recommender_amazon.py",
        "sensitivity_runner": "amazon/step5_candidate_sensitivity_amazon.py",
        "comparison_runner": "amazon/step4_compare_prediction_models_amazon.py",
    },
    "amazon_beauty": {
        "source_version": "20260718",
        "environment": "amazon_beauty",
        "seed_environment": "AMAZON_TRAINING_SEED_OFFSET",
        "utility": "amazon/A_base/amazon_baseline_utils.py",
        "baseline_runner": "amazon/run_prediction_baselines_amazon.py",
        "dcml_trainer": "amazon/step2_train_dcml_recommender_amazon.py",
        "dcml_evaluator": "amazon/step3_evaluate_dcml_recommender_amazon.py",
        "sensitivity_runner": "amazon/step5_candidate_sensitivity_amazon.py",
        "comparison_runner": "amazon/step4_compare_prediction_models_amazon.py",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract_paths(dataset: str, baseline_root: Path) -> dict[str, Path]:
    contract = DATASET_CONTRACT[dataset]
    keys = ("utility", "baseline_runner", "dcml_trainer", "dcml_evaluator", "sensitivity_runner", "comparison_runner")
    return {key: baseline_root / contract[key] for key in keys}


def _preflight(dataset: str, baseline_root: Path) -> dict[str, Path]:
    paths = _contract_paths(dataset, baseline_root)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing retained training-pipeline files for {dataset}:\n{formatted}")
    return paths


def _load_utility(dataset: str, path: Path) -> ModuleType:
    module_name = f"prediction_multiseed_utility_{dataset}_{OUTPUT_VERSION}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load baseline utility module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _parse_offsets(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("seed offsets must be a nonempty comma-separated list without duplicates")
    if 0 not in values:
        raise ValueError("seed offsets must include 0 so the primary run can be restored")
    return tuple(value for value in values if value != 0) + (0,)


def _expected_metrics(scores, target_index: int) -> dict[str, float]:
    import numpy as np

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


def _evaluate_bundle(candidates: dict, predictions: dict, utility: ModuleType) -> dict[str, float]:
    import numpy as np

    if set(map(str, candidates)) != set(map(str, predictions)):
        raise RuntimeError("prediction users do not match candidate users")
    values: dict[str, list[float]] = defaultdict(list)
    for user, candidate in candidates.items():
        ids = utility.candidate_items(candidate)
        scores_by_item = predictions[str(user)]
        if set(scores_by_item) != {str(item) for item in ids}:
            raise RuntimeError(f"candidate item mismatch for user {user}")
        scores = np.asarray([float(scores_by_item[str(item)]) for item in ids], dtype=float)
        metrics = _expected_metrics(scores, len(ids) - 1)
        for metric, value in metrics.items():
            values[metric].append(value)
    output = {"Evaluated_Users": len(candidates)}
    for metric in METRICS:
        array = np.asarray(values[metric], dtype=float)
        output[metric] = float(array.mean())
        output[f"{metric}_User_SE"] = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return output


def _archive_metadata(
    root: Path,
    seed_offset: int,
    source_version: str,
    stages: tuple[str, ...],
    utility: ModuleType,
) -> Path:
    destination = utility.ensure_dir(
        utility.performance_dir(root) / f"training_seed_metadata_{source_version}" / f"offset_{seed_offset}"
    )
    baseline = utility.baseline_data_dir(root)
    performance = utility.performance_dir(root)
    sources = [
        *(utility.prediction_metadata_path(root, model, stage) for model in MODELS[:-1] for stage in stages),
        *(baseline / f"{model}_training_diagnostics_{source_version}.json" for model in MODELS[:-1]),
        performance / f"DCML_Recommender_Metadata_{source_version}.json",
        performance / f"DCML_Recommendation_Preflight_{source_version}.json",
        performance / f"DCML_Recommender_Training_Diagnostics_{source_version}.csv",
        performance / f"DCML_Item_Category_Audit_{source_version}.json",
        performance / f"DCML_Recommendation_Validation_Tuning_{source_version}.csv",
        performance / f"DCML_Recommendation_Causal_Prior_{source_version}.csv",
        performance / f"DCML_Recommendation_Orthogonalization_{source_version}.csv",
        performance / f"DCML_Recommendation_Prior_Stability_{source_version}.csv",
        performance / f"DCML_User_State_Audit_{source_version}.csv",
        *(performance / f"DCML_{stage}_prediction_metadata_{source_version}.json" for stage in stages),
        performance / f"DCML_Evaluation_Metadata_{source_version}.json",
        performance / f"Performance_Results_Summary_Hybrid_{source_version}.csv",
    ]
    for source in sources:
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    return destination


def _summarize_runs(runs):
    import numpy as np
    import pandas as pd

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
    return pd.DataFrame(summary_rows)


def _rename_versioned_tree(source: Path, destination: Path, old_version: str) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination.joinpath(*(part.replace(old_version, OUTPUT_VERSION) for part in relative.parts))
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _run_five_seed_pipeline(
    dataset: str,
    offsets: tuple[int, ...],
    paths: dict[str, Path],
    baseline_root: Path,
    utility: ModuleType,
) -> tuple[Path, Path]:
    import pandas as pd

    contract = DATASET_CONTRACT[dataset]
    source_version = contract["source_version"]
    causal_root = baseline_root.parent
    workspace = causal_root.parent
    root = utility.find_causal_root(__file__)
    output = utility.performance_dir(root)
    stages = tuple(utility.STAGES)
    rows: list[dict] = []
    logger = utility.setup_logger(root, f"prediction_multiseed_{OUTPUT_VERSION}")

    for seed_offset in offsets:
        logger.info("Starting %s all-model training run with seed offset %d", dataset, seed_offset)
        environment = os.environ.copy()
        if contract["environment"]:
            environment["AMAZON_BASELINE_DATASET"] = contract["environment"]
        environment[contract["seed_environment"]] = str(seed_offset)
        subprocess.run(
            [sys.executable, str(paths["baseline_runner"]), "--only-train"],
            check=True,
            cwd=workspace,
            env=environment,
        )
        subprocess.run([sys.executable, str(paths["dcml_trainer"])], check=True, cwd=workspace, env=environment)
        subprocess.run(
            [sys.executable, str(paths["dcml_evaluator"]), "--resolution", "Macro"],
            check=True,
            cwd=workspace,
            env=environment,
        )
        for stage in stages:
            candidates = utility.load_candidates(root, stage)
            for model in MODELS:
                prediction_file = (
                    output / f"DCML_{stage}_predictions_{source_version}.json"
                    if model == "DCML (Ours)"
                    else utility.prediction_path(root, model, stage)
                )
                metadata_file = (
                    output / f"DCML_{stage}_prediction_metadata_{source_version}.json"
                    if model == "DCML (Ours)"
                    else utility.prediction_metadata_path(root, model, stage)
                )
                result = _evaluate_bundle(candidates, utility.read_json(prediction_file), utility)
                metadata = utility.read_json(metadata_file)
                rows.append(
                    {
                        "Seed_Offset": seed_offset,
                        "Stage": stage,
                        "Model": model,
                        "Actual_Training_Seed": metadata.get("seed"),
                        **result,
                    }
                )
        _archive_metadata(root, seed_offset, source_version, stages, utility)
        pd.DataFrame(rows).to_csv(output / f"Recommendation_Training_Seed_Runs_{source_version}.csv", index=False)

    runs = pd.DataFrame(rows)
    summary = _summarize_runs(runs)
    runs_path = output / f"Recommendation_Training_Seed_Runs_{source_version}.csv"
    summary_path = output / f"Recommendation_Training_Seed_Summary_{source_version}.csv"
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)

    final_environment = os.environ.copy()
    if contract["environment"]:
        final_environment["AMAZON_BASELINE_DATASET"] = contract["environment"]
    final_environment[contract["seed_environment"]] = "0"
    subprocess.run([sys.executable, str(paths["sensitivity_runner"])], check=True, cwd=workspace, env=final_environment)
    subprocess.run([sys.executable, str(paths["comparison_runner"])], check=True, cwd=workspace, env=final_environment)
    logger.info("Completed %d seed runs for %s; offset 0 is restored as the primary output", len(offsets), dataset)
    return runs_path, summary_path


def run(dataset: str, seed_offsets: str, summarize_existing: bool, preflight_only: bool) -> None:
    contract = DATASET_CONTRACT[dataset]
    source_version = contract["source_version"]
    offsets = _parse_offsets(seed_offsets)
    baseline_root = Path(__file__).resolve().parent
    causal_root = baseline_root.parent
    paths = _preflight(dataset, baseline_root)
    if preflight_only:
        environment = os.environ.copy()
        if contract["environment"]:
            environment["AMAZON_BASELINE_DATASET"] = contract["environment"]
        environment[contract["seed_environment"]] = "0"
        subprocess.run(
            [sys.executable, str(paths["baseline_runner"]), "--only-train", "--check-only"],
            check=True,
            cwd=causal_root.parent,
            env=environment,
        )
        print(f"Preflight passed for {dataset}: {len(paths)} retained training-pipeline files are available.")
        print("The eight prediction baselines also passed their script and dependency checks.")
        print("No historical multiseed runner is required.")
        return

    import pandas as pd

    environment_value = contract["environment"]
    if environment_value:
        os.environ["AMAZON_BASELINE_DATASET"] = environment_value
    utility = _load_utility(dataset, paths["utility"])
    output = _dcml_paths.workspace_path(causal_root, 'results_of_comparison') / dataset / "dcml_performance"
    source_runs = output / f"Recommendation_Training_Seed_Runs_{source_version}.csv"
    source_summary = output / f"Recommendation_Training_Seed_Summary_{source_version}.csv"
    if not summarize_existing:
        source_runs, source_summary = _run_five_seed_pipeline(dataset, offsets, paths, baseline_root, utility)
    if not source_runs.is_file() or not source_summary.is_file():
        raise FileNotFoundError(
            f"Missing complete source-version seed results: {source_runs} / {source_summary}. "
            "Run without --summarize-existing first."
        )

    runs = pd.read_csv(source_runs)
    summary = pd.read_csv(source_summary)
    expected_offsets = sorted(offsets)
    observed_offsets = sorted(pd.to_numeric(runs["Seed_Offset"], errors="raise").astype(int).unique().tolist())
    if observed_offsets != expected_offsets:
        raise RuntimeError(f"Seed offsets do not match: observed={observed_offsets}, expected={expected_offsets}")
    expected_rows = len(expected_offsets) * len(tuple(utility.STAGES)) * len(MODELS)
    if len(runs) != expected_rows:
        raise RuntimeError(f"Incomplete seed result table: rows={len(runs)}, expected={expected_rows}")
    for frame in (runs, summary):
        frame.insert(0, "Dataset", dataset)
        frame["Supplement_Version"] = OUTPUT_VERSION
        frame["Frozen_Training_Pipeline_Version"] = source_version
    runs_path = output / f"Recommendation_Training_Seed_Runs_{OUTPUT_VERSION}.csv"
    summary_path = output / f"Recommendation_Training_Seed_Summary_{OUTPUT_VERSION}.csv"
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)

    source_archive = output / f"training_seed_metadata_{source_version}"
    destination_archive = output / f"training_seed_metadata_{OUTPUT_VERSION}"
    if source_archive.is_dir():
        _rename_versioned_tree(source_archive, destination_archive, source_version)
    manifest = {
        "dataset": dataset,
        "supplement_version": OUTPUT_VERSION,
        "frozen_training_pipeline_version": source_version,
        "self_contained_multiseed_orchestrator": str(Path(__file__).resolve()),
        "source_pipeline_files": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in paths.items()},
        "historical_multiseed_runner_required": False,
        "seed_offsets": expected_offsets,
        "seed_run_count": int(runs["Seed_Offset"].nunique()),
        "runs_file": str(runs_path),
        "summary_file": str(summary_path),
        "archived_metadata_directory": str(destination_archive) if destination_archive.is_dir() else None,
    }
    with open(output / f"Recommendation_Training_Seed_Manifest_{OUTPUT_VERSION}.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_CONTRACT), required=True)
    parser.add_argument("--seed-offsets", default=DEFAULT_OFFSETS)
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check retained training-pipeline files without training or requiring an older multiseed runner.",
    )
    args = parser.parse_args()
    run(args.dataset, args.seed_offsets, args.summarize_existing, args.preflight_only)


if __name__ == "__main__":
    main()
