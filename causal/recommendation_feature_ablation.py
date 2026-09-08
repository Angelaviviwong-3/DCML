#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Held-out predictive feature ablations, separate from causal estimation."""

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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from dcml_inference import inference_contract
from dcml_nuisance import dataset_contract
from dcml_utils import ensure_dir, find_causal_root, read_table, setup_logging
from temporal_certification import filter_certified_frame, load_certification

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


OUTPUT_VERSION = "20260719"
SOURCE_VERSION = "20260717"
DATASETS = ("suning", "amazon_appliances", "amazon_beauty")


def _classifier(seed: int):
    if lgb is not None:
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=260,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.04,
        max_leaf_nodes=31,
        random_state=seed,
    )


def _prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_x = pd.DataFrame(index=train.index)
    test_x = pd.DataFrame(index=test.index)
    manifest: dict[str, dict] = {}
    for column in columns:
        train_value = pd.to_numeric(train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        test_value = pd.to_numeric(test[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(train_value.median()) if train_value.notna().any() else 0.0
        train_x[column] = train_value.fillna(median).astype(float)
        test_x[column] = test_value.fillna(median).astype(float)
        missing_indicator = bool(train_value.isna().any() or test_value.isna().any())
        if missing_indicator:
            indicator = f"{column}__missing"
            train_x[indicator] = train_value.isna().astype(float)
            test_x[indicator] = test_value.isna().astype(float)
        manifest[column] = {
            "set_b_median": median,
            "missing_indicator": missing_indicator,
            "set_b_missing_rate": float(train_value.isna().mean()),
            "set_c_missing_rate": float(test_value.isna().mean()),
        }
    return train_x, test_x, manifest


def _predict(model, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        if probabilities.shape[1] == 1:
            return np.full(len(features), float(np.asarray(model.classes_)[0]))
        return probabilities[:, 1]
    return np.asarray(model.predict(features), dtype=float)


def _ranking_metrics(frame: pd.DataFrame, labels: np.ndarray, scores: np.ndarray) -> dict:
    ranked = pd.DataFrame(
        {
            "user_id": frame["user_id"].astype(str).to_numpy(),
            "label": labels,
            "score": scores,
            "index": np.arange(len(frame)),
        }
    )
    values = {"HR@10": [], "NDCG@10": [], "HR@20": [], "NDCG@20": []}
    users = 0
    for _, group in ranked.groupby("user_id", sort=False):
        if group["label"].sum() <= 0 or group["label"].sum() >= len(group):
            continue
        ordered = group.sort_values(["score", "index"], ascending=[False, True], kind="mergesort")
        relevance = ordered["label"].to_numpy(dtype=float)
        ideal_n = int(relevance.sum())
        users += 1
        for k in (10, 20):
            top = relevance[:k]
            values[f"HR@{k}"].append(float(top.sum() > 0))
            discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
            dcg = float(np.sum(top * discounts))
            ideal = np.ones(min(ideal_n, k), dtype=float)
            idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
            values[f"NDCG@{k}"].append(dcg / idcg if idcg > 0 else 0.0)
    return {**{key: float(np.mean(value)) if value else np.nan for key, value in values.items()}, "Ranking_Users": users}


def _classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    two_classes = len(np.unique(labels)) == 2
    return {
        "ROC_AUC": float(roc_auc_score(labels, scores)) if two_classes else np.nan,
        "Average_Precision": float(average_precision_score(labels, scores)) if two_classes else np.nan,
        "Brier": float(brier_score_loss(labels, scores)),
        "Log_Loss": float(log_loss(labels, clipped, labels=[0, 1])),
    }


def run_dataset(root: Path, dataset: str, seed: int) -> None:
    logger = setup_logging(ensure_dir(_dcml_paths.workspace_path(root, 'log')), f"recommendation_ablation_{dataset}_{OUTPUT_VERSION}")
    start = time.time()
    data = dataset_contract(root, dataset)
    inference = inference_contract(root, dataset)
    with open(inference["schema"], "r", encoding="utf-8") as handle:
        schema = json.load(handle)
    certified_items, _, _ = load_certification(root, dataset)
    train = filter_certified_frame(read_table(data["b"]), certified_items).reset_index(drop=True)
    test = filter_certified_frame(read_table(data["c"]), certified_items).reset_index(drop=True)
    for frame in (train, test):
        no_history = pd.to_numeric(frame["H_level"], errors="coerce").fillna(0).eq(0)
        frame.loc[no_history, "T_int_sem"] = np.nan

    all_treatments = list(schema.get("treatments", schema.get("all_treatments", [])))
    aggregate = list(schema["aggregate_treatments"])
    disaggregate = list(schema["disaggregate_treatments"])
    confounders = list(schema["confounder_whitelist"])
    if (len(all_treatments), len(aggregate), len(disaggregate)) != (12, 6, 10):
        raise RuntimeError(
            f"{dataset}: expected Full/Aggregate/Disaggregate sizes 12/6/10, found "
            f"{len(all_treatments)}/{len(aggregate)}/{len(disaggregate)}"
        )
    specifications = {
        "Base-X": confounders,
        "Aggregate-6": list(dict.fromkeys([*confounders, *aggregate])),
        "Disaggregate-10": list(dict.fromkeys([*confounders, *disaggregate])),
        "Full-12": list(dict.fromkeys([*confounders, *all_treatments])),
    }
    required = set().union(*specifications.values())
    for split_name, frame in (("Set B", train), ("Set C", test)):
        if missing := sorted(required.difference(frame.columns)):
            raise RuntimeError(f"{dataset} {split_name} missing predictive columns: {missing}")

    rows: list[dict] = []
    feature_manifest: dict[str, dict] = {}
    for specification_index, (specification, columns) in enumerate(specifications.items()):
        train_x, test_x, preprocessing = _prepare_features(train, test, columns)
        feature_manifest[specification] = {
            "source_columns": columns,
            "model_input_columns": list(train_x.columns),
            "preprocessing": preprocessing,
        }
        for outcome_index, outcome in enumerate(schema["outcomes"]):
            train_y = pd.to_numeric(train[outcome], errors="coerce").fillna(0).astype(int).to_numpy()
            test_y = pd.to_numeric(test[outcome], errors="coerce").fillna(0).astype(int).to_numpy()
            if len(np.unique(train_y)) < 2:
                model = DummyClassifier(strategy="constant", constant=int(train_y[0]))
            else:
                model = _classifier(seed + specification_index * 100 + outcome_index)
            model.fit(train_x, train_y)
            probabilities = _predict(model, test_x)
            rows.append(
                {
                    "Dataset": dataset,
                    "Specification": specification,
                    "Stage": outcome.replace("y_", ""),
                    "N_Train": int(len(train_y)),
                    "N_Test": int(len(test_y)),
                    "Test_Positive_Rate": float(test_y.mean()),
                    "Feature_Count": int(train_x.shape[1]),
                    **_classification_metrics(test_y, probabilities),
                    **_ranking_metrics(test, test_y, probabilities),
                    "Model": type(model).__name__,
                    "Evaluation_Scope": "held-out certified Set C rows grouped by user",
                    "Feature_Ablation_Type": "auxiliary gradient-boosted held-out row scorer",
                    "Causal_Interpretation": "none; predictive treatment-feature ablation only",
                    "Source_Data_Version": SOURCE_VERSION,
                }
            )

    output = ensure_dir(_dcml_paths.workspace_path(root, 'results_of_comparison') / dataset / "dcml_performance")
    result_path = output / f"{dataset}_Treatment_Feature_Ablation_{OUTPUT_VERSION}.csv"
    manifest_path = output / f"{dataset}_Treatment_Feature_Ablation_Manifest_{OUTPUT_VERSION}.json"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset,
                "script_version": OUTPUT_VERSION,
                "source_data_version": SOURCE_VERSION,
                "training_split": "temporally certified Set B",
                "evaluation_split": "temporally certified Set C",
                "causal_interpretation": "none; predictive treatment-feature ablation only",
                "semantic_missing_policy": (
                    "T_int_sem remains N/A when pre-event purchase history is unavailable; predictive models use "
                    "a Set-B-fitted median and an explicit missing-history indicator."
                ),
                "full_12_policy": (
                    "Aggregate parents and their atomic children coexist only in Full-12 predictive ablation; "
                    "Full-12 is never used as a joint causal specification."
                ),
                "specifications": feature_manifest,
                "result_file": str(result_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Completed predictive feature ablation for %s in %.2f seconds", dataset, time.time() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    root = find_causal_root(__file__)
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        run_dataset(root, dataset, args.seed)


if __name__ == "__main__":
    main()
