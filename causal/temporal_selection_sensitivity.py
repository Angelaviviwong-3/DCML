#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Out-of-item selection diagnostics for the temporally certified population."""

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from dcml_inference import inference_contract
from dcml_nuisance import dataset_contract
from dcml_utils import ensure_dir, find_causal_root, normalize_ids, read_table, setup_logging
from temporal_certification import load_certification

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
            n_estimators=180,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.04,
        max_leaf_nodes=31,
        random_state=seed,
    )


def _predict(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x), dtype=float)
        return probabilities[:, 1]
    return np.asarray(model.predict(x), dtype=float)


def _stratified_row_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0 or len(frame) <= maximum:
        return frame
    pieces = []
    for value, group in frame.groupby("__certified", sort=False):
        target = max(1, int(round(maximum * len(group) / len(frame))))
        pieces.append(group.sample(n=min(target, len(group)), random_state=seed + int(value)))
    sampled = pd.concat(pieces, ignore_index=False)
    if len(sampled) > maximum:
        sampled = sampled.sample(n=maximum, random_state=seed)
    return sampled


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)


def _weighted_balance(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    columns: list[str],
    roles: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    selected = frame["__certified"].eq(1).to_numpy()
    selected_p = np.clip(probabilities[selected], 0.01, 0.99)
    raw_weights = 1.0 / selected_p
    cap = float(np.quantile(raw_weights, 0.99))
    weights = np.minimum(raw_weights, cap)
    weights /= weights.mean()
    rows: list[dict] = []
    for column in dict.fromkeys(columns):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        full = values.dropna()
        certified = values[selected]
        observed = certified.notna().to_numpy()
        if full.empty or not observed.any():
            continue
        full_mean = float(full.mean())
        full_sd = float(full.std(ddof=1)) if len(full) > 1 else 0.0
        certified_mean = float(certified.dropna().mean())
        weighted_mean = float(np.average(certified.to_numpy(dtype=float)[observed], weights=weights[observed]))
        rows.append(
            {
                "Column": column,
                "Role": roles.get(column, "diagnostic"),
                "Heldout_Full_N": int(len(full)),
                "Heldout_Certified_N": int(observed.sum()),
                "Heldout_Full_Mean": full_mean,
                "Heldout_Certified_Mean": certified_mean,
                "IPCW_Certified_Mean": weighted_mean,
                "Unweighted_SMD_vs_Full": (certified_mean - full_mean) / full_sd if full_sd > 0 else np.nan,
                "IPCW_SMD_vs_Full": (weighted_mean - full_mean) / full_sd if full_sd > 0 else np.nan,
                "Weight_Trim_Quantile": 0.99,
                "Interpretation": "selection sensitivity on held-out items; not restoration of full-sample identification",
            }
        )
    diagnostics = {
        "heldout_certified_rows": int(selected.sum()),
        "ipcw_weight_mean": float(weights.mean()),
        "ipcw_weight_p99_before_trim": cap,
        "ipcw_weight_max_after_trim": float(weights.max()),
        "ipcw_effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
    }
    return pd.DataFrame(rows), diagnostics


def run_dataset(root: Path, dataset: str, max_train_rows: int, seed: int) -> None:
    logger = setup_logging(ensure_dir(_dcml_paths.workspace_path(root, 'log')), f"temporal_selection_{dataset}_{OUTPUT_VERSION}")
    start = time.time()
    contract = dataset_contract(root, dataset)
    inference = inference_contract(root, dataset)
    with open(inference["schema"], "r", encoding="utf-8") as handle:
        schema = json.load(handle)
    frame = read_table(contract["c"])
    normalize_ids(frame, ["user_id", "item_id"])
    certified_items, certification_summary, item_table = load_certification(root, dataset)
    frame["__certified"] = frame["item_id"].astype(str).isin(certified_items).astype(np.int8)

    item_labels = item_table[["item_id", "strictly_pre_set_c"]].copy()
    item_labels["__certified"] = item_labels["item_id"].astype(str).isin(certified_items).astype(np.int8)
    train_items, test_items = train_test_split(
        item_labels,
        test_size=0.25,
        random_state=seed,
        stratify=item_labels["__certified"],
    )
    train_item_ids = set(train_items["item_id"].astype(str))
    test_item_ids = set(test_items["item_id"].astype(str))
    train = frame[frame["item_id"].astype(str).isin(train_item_ids)]
    heldout = frame[frame["item_id"].astype(str).isin(test_item_ids)].copy()
    train = _stratified_row_sample(train, max_train_rows, seed)

    x_columns = [column for column in schema.get("confounder_whitelist", []) if column in frame.columns]
    if not x_columns:
        raise RuntimeError(f"{dataset}: no pre-specified confounders are available for selection diagnostics")
    model = _classifier(seed)
    model.fit(_numeric(train, x_columns), train["__certified"].to_numpy(dtype=int))
    probabilities = np.clip(_predict(model, _numeric(heldout, x_columns)), 1e-6, 1 - 1e-6)
    labels = heldout["__certified"].to_numpy(dtype=int)
    auc = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else np.nan
    brier = float(brier_score_loss(labels, probabilities))

    treatments = list(schema.get("treatments", schema.get("all_treatments", [])))
    outcomes = list(schema.get("outcomes", []))
    balance_columns = [*x_columns, "H_level", "M_norm", *treatments, *outcomes]
    roles = {
        **{column: "pre-specified confounder" for column in x_columns},
        **{column: "treatment" for column in treatments},
        **{column: "outcome_descriptive_only" for column in outcomes},
        "H_level": "moderator",
        "M_norm": "moderator_proxy",
    }
    balance, weight_diagnostics = _weighted_balance(heldout, probabilities, balance_columns, roles)
    max_unweighted_smd = float(balance["Unweighted_SMD_vs_Full"].abs().max()) if not balance.empty else np.nan
    max_weighted_smd = float(balance["IPCW_SMD_vs_Full"].abs().max()) if not balance.empty else np.nan
    audit_dir = ensure_dir(_dcml_paths.workspace_path(root, 'processed_data') / dataset / "audit")
    balance_path = audit_dir / f"{dataset}_temporal_certification_weighted_balance_{OUTPUT_VERSION}.csv"
    metrics_path = audit_dir / f"{dataset}_temporal_certification_selection_model_{OUTPUT_VERSION}.csv"
    manifest_path = audit_dir / f"{dataset}_temporal_certification_selection_manifest_{OUTPUT_VERSION}.json"
    balance.to_csv(balance_path, index=False)
    metrics = pd.DataFrame(
        [
            {
                "Dataset": dataset,
                "Source_Version": SOURCE_VERSION,
                "Train_Item_Count": int(len(train_items)),
                "Heldout_Item_Count": int(len(test_items)),
                "Model_Train_Rows": int(len(train)),
                "Heldout_Rows": int(len(heldout)),
                "Heldout_Certification_Rate": float(labels.mean()),
                "Out_of_Item_ROC_AUC": auc,
                "Out_of_Item_Brier": brier,
                "Propensity_P01": float(np.quantile(probabilities, 0.01)),
                "Propensity_P50": float(np.quantile(probabilities, 0.50)),
                "Propensity_P99": float(np.quantile(probabilities, 0.99)),
                "Common_Support_001_099_Rate": float(np.mean((probabilities >= 0.01) & (probabilities <= 0.99))),
                **weight_diagnostics,
                "Model_Features": "|".join(x_columns),
                "Selection_Model_Uses_Outcomes": False,
                "Selection_Model_Uses_Treatments": False,
                "Max_Absolute_Unweighted_SMD": max_unweighted_smd,
                "Max_Absolute_IPCW_SMD": max_weighted_smd,
                "Claim_Boundary": "diagnoses observable certification selection; cannot remove selection on unobserved item attributes",
            }
        ]
    )
    metrics.to_csv(metrics_path, index=False)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset,
                "script_version": OUTPUT_VERSION,
                "source_version": SOURCE_VERSION,
                "full_sample_audit_status": certification_summary.get("audit_status"),
                "analysis_estimand": "conditional effects among temporally certified items",
                "model_split": "item-disjoint 75/25 train/held-out diagnostic split",
                "model_features": x_columns,
                "selection_model_uses_outcomes": False,
                "selection_model_uses_treatments": False,
                "weighted_balance_file": str(balance_path),
                "selection_metrics_file": str(metrics_path),
                "identification_warning": (
                    "IPCW balance is a sensitivity diagnostic for observed X only. It does not certify exchangeability "
                    "between certified and failed items and is not used to replace the primary certified-population estimand."
                ),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Completed temporal-selection sensitivity for %s in %.2f seconds", dataset, time.time() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--max-train-rows", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    root = find_causal_root(__file__)
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        run_dataset(root, dataset, args.max_train_rows, args.seed)


if __name__ == "__main__":
    main()
