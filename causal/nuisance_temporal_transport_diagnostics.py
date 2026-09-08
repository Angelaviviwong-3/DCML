#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Set-B-to-Set-C transport of the frozen nuisance outcome models."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[0]))
import project_paths as _dcml_paths


import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from dcml_inference import inference_contract
from dcml_nuisance import dataset_contract
from dcml_utils import ensure_dir, find_causal_root, read_table, setup_logging
from temporal_certification import filter_certified_frame, load_certification


OUTPUT_VERSION = "20260719"
SOURCE_VERSION = "20260717"
DATASETS = ("suning", "amazon_appliances", "amazon_beauty")


def _calibration(y: np.ndarray, prediction: np.ndarray, maximum: int, seed: int) -> tuple[float, float]:
    index = np.arange(len(y))
    if maximum > 0 and len(index) > maximum:
        rng = np.random.default_rng(seed)
        index = rng.choice(index, size=maximum, replace=False)
    p = np.clip(prediction[index], 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    design = sm.add_constant(logit, has_constant="add")
    try:
        model = sm.GLM(y[index], design, family=sm.families.Binomial()).fit()
        return float(model.params[0]), float(model.params[1])
    except Exception:
        return np.nan, np.nan


def _ece(y: np.ndarray, prediction: np.ndarray, bins: int = 10) -> float:
    frame = pd.DataFrame({"y": y, "p": prediction})
    try:
        frame["bin"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    except ValueError:
        return np.nan
    grouped = frame.groupby("bin", observed=True).agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
    return float(np.sum(grouped["n"] / len(frame) * np.abs(grouped["y"] - grouped["p"])))


def run_dataset(root: Path, dataset: str, calibration_max_rows: int, seed: int) -> None:
    logger = setup_logging(ensure_dir(_dcml_paths.workspace_path(root, 'log')), f"nuisance_transport_{dataset}_{OUTPUT_VERSION}")
    start = time.time()
    data = dataset_contract(root, dataset)
    inference = inference_contract(root, dataset)
    certified_items, _, _ = load_certification(root, dataset)
    set_b = filter_certified_frame(read_table(data["b"]), certified_items)
    residual = read_table(inference["final_residual"])
    rows: list[dict] = []
    for outcome_column in inference["outcomes"]:
        outcome = outcome_column.replace("res_", "")
        raw_column = f"raw_{outcome}"
        prediction_column = f"yhat_{outcome}"
        if raw_column not in residual.columns or prediction_column not in residual.columns:
            raise RuntimeError(f"{dataset}: residual table lacks {raw_column}/{prediction_column}")
        y = pd.to_numeric(residual[raw_column], errors="coerce").fillna(0).to_numpy(dtype=int)
        prediction = np.clip(
            pd.to_numeric(residual[prediction_column], errors="coerce").to_numpy(dtype=float),
            1e-6,
            1 - 1e-6,
        )
        train_y = pd.to_numeric(set_b[outcome], errors="coerce").fillna(0).to_numpy(dtype=int)
        train_rate = float(train_y.mean())
        test_rate = float(y.mean())
        null_prediction = np.full(len(y), np.clip(train_rate, 1e-6, 1 - 1e-6), dtype=float)
        model_brier = float(brier_score_loss(y, prediction))
        null_brier = float(brier_score_loss(y, null_prediction))
        calibration_intercept, calibration_slope = _calibration(
            y, prediction, calibration_max_rows, seed
        )
        rows.append(
            {
                "Dataset": dataset,
                "Stage": outcome.replace("y_", ""),
                "Source_Model_Version": SOURCE_VERSION,
                "Set_B_N": int(len(train_y)),
                "Set_C_N": int(len(y)),
                "Set_B_Prevalence": train_rate,
                "Set_C_Prevalence": test_rate,
                "Absolute_Prevalence_Drift": abs(test_rate - train_rate),
                "Relative_Prevalence_Ratio_C_over_B": test_rate / train_rate if train_rate > 0 else np.nan,
                "Mean_Predicted_Probability_Set_C": float(prediction.mean()),
                "ROC_AUC_Set_C": float(roc_auc_score(y, prediction)) if len(np.unique(y)) == 2 else np.nan,
                "Brier_Model_Set_C": model_brier,
                "Brier_Train_Prevalence_Null_Set_C": null_brier,
                "Brier_Skill_vs_Train_Prevalence_Null": 1.0 - model_brier / null_brier if null_brier > 0 else np.nan,
                "Log_Loss_Model_Set_C": float(log_loss(y, prediction, labels=[0, 1])),
                "Log_Loss_Train_Prevalence_Null_Set_C": float(log_loss(y, null_prediction, labels=[0, 1])),
                "Calibration_Intercept": calibration_intercept,
                "Calibration_Slope": calibration_slope,
                "ECE_10_Quantile_Bins": _ece(y, prediction),
                "Model_Beats_Train_Prevalence_Null": bool(model_brier < null_brier),
                "Calibration_Slope_0_8_to_1_2": bool(
                    np.isfinite(calibration_slope) and 0.8 <= calibration_slope <= 1.2
                ),
                "Interpretation": "out-of-time nuisance transport diagnostic; weak calibration limits precision but does not by itself prove causal bias",
            }
        )
    output = ensure_dir(_dcml_paths.workspace_path(root, 'processed_data') / dataset / "DML_Results")
    result_path = output / f"{dataset}_nuisance_temporal_transport_diagnostics_{OUTPUT_VERSION}.csv"
    manifest_path = output / f"{dataset}_nuisance_temporal_transport_manifest_{OUTPUT_VERSION}.json"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset,
                "script_version": OUTPUT_VERSION,
                "source_model_version": SOURCE_VERSION,
                "training_period": "temporally certified Set B",
                "evaluation_period": "temporally certified Set C",
                "null_model": "constant Set B outcome prevalence evaluated on Set C",
                "calibration_sample_max_rows": calibration_max_rows,
                "diagnostics_file": str(result_path),
                "claim_boundary": (
                    "These diagnostics assess temporal transport and calibration. They do not refit nuisance models "
                    "and do not change the frozen 20260717 residuals."
                ),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Completed nuisance transport diagnostics for %s in %.2f seconds", dataset, time.time() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--calibration-max-rows", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    root = find_causal_root(__file__)
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        run_dataset(root, dataset, args.calibration_max_rows, args.seed)


if __name__ == "__main__":
    main()
