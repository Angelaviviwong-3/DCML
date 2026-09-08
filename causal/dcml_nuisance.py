#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared nuisance residualization for the 20260717 DCML pipelines."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[0]))
import project_paths as _dcml_paths


import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dcml_utils import ensure_dir, read_table, setup_logging, write_json, write_table
from temporal_certification import restrict_analysis_splits

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    import joblib
except Exception:  # pragma: no cover
    from sklearn.externals import joblib  # type: ignore

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, mean_squared_error, r2_score, roc_auc_score


VERSION = "20260717"
FORBIDDEN_AMAZON_X = {
    "user_review_count",
    "user_avg_rating",
    "user_review_count_scaled",
    "user_avg_rating_scaled",
}


def make_regressor(seed: int):
    if lgb is not None:
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=250,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=31,
        random_state=seed,
    )


def make_classifier(seed: int):
    if lgb is not None:
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=250,
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
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=31,
        random_state=seed,
    )


def numeric_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def predict_binary(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=float)
        if proba.shape[1] == 1:
            return np.full(len(x), float(np.asarray(model.classes_)[0]))
        return proba[:, 1]
    return np.asarray(model.predict(x), dtype=float)


def treatment_residuals(
    train: pd.DataFrame,
    test: pd.DataFrame,
    t_cols: list[str],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    model_dir: Path,
    seed: int,
    logger,
) -> tuple[pd.DataFrame, list[dict]]:
    keep = ["user_id", "item_id", "timestamp", "H_level", "M_norm"]
    keep += [c for c in ["T_int_sem_available"] if c in test.columns]
    residuals = test[keep].copy()
    residuals["cluster_user_id"] = test["user_id"].astype(str)
    residuals["cluster_item_id"] = test["item_id"].astype(str)
    diagnostics: list[dict] = []

    for index, treatment in enumerate(t_cols):
        train_t = pd.to_numeric(train[treatment], errors="coerce")
        test_t = pd.to_numeric(test[treatment], errors="coerce")
        observed_train = train_t.notna()
        observed_test = test_t.notna()
        if not observed_train.any():
            logger.warning("No observed training values for %s; residuals remain missing", treatment)
            residuals[f"raw_{treatment}"] = test_t
            residuals[f"mhat_{treatment}"] = np.nan
            residuals[f"res_{treatment}"] = np.nan
            diagnostics.append(
                {
                    "target": treatment,
                    "target_type": "treatment",
                    "train_n": 0,
                    "test_n": int(observed_test.sum()),
                    "rmse": np.nan,
                    "r2": np.nan,
                }
            )
            continue
        unique = train_t.loc[observed_train].nunique()
        model = (
            DummyRegressor(strategy="constant", constant=float(train_t.loc[observed_train].iloc[0]))
            if unique <= 1
            else make_regressor(seed + index)
        )
        model.fit(x_train.loc[observed_train], train_t.loc[observed_train])
        prediction = np.clip(np.asarray(model.predict(x_test), dtype=float), 0.0, 1.0)
        raw = test_t.to_numpy(dtype=float)
        prediction[~observed_test.to_numpy()] = np.nan
        residuals[f"raw_{treatment}"] = raw
        residuals[f"mhat_{treatment}"] = prediction
        residuals[f"res_{treatment}"] = raw - prediction
        joblib.dump(model, model_dir / f"model_{treatment}_{VERSION}.pkl".replace("/", "_"))
        if observed_test.any():
            y_true = test_t.loc[observed_test].to_numpy(dtype=float)
            y_pred = prediction[observed_test.to_numpy()]
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
        else:
            rmse = r2 = np.nan
        diagnostics.append(
            {
                "target": treatment,
                "target_type": "treatment",
                "train_n": int(observed_train.sum()),
                "test_n": int(observed_test.sum()),
                "rmse": rmse,
                "r2": r2,
            }
        )
        del model
        gc.collect()
    return residuals, diagnostics


def outcome_residuals(
    residuals: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    outcomes: list[str],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    model_dir: Path,
    seed: int,
) -> list[dict]:
    diagnostics: list[dict] = []
    for index, outcome in enumerate(outcomes):
        train_y = pd.to_numeric(train[outcome], errors="coerce").fillna(0).astype(int)
        model = (
            DummyClassifier(strategy="constant", constant=int(train_y.iloc[0]))
            if train_y.nunique() <= 1
            else make_classifier(seed + 100 + index)
        )
        model.fit(x_train, train_y)
        prediction = np.clip(predict_binary(model, x_test), 0.0, 1.0)
        raw = pd.to_numeric(test[outcome], errors="coerce").fillna(0).to_numpy(dtype=float)
        residuals[f"raw_{outcome}"] = raw
        residuals[f"yhat_{outcome}"] = prediction
        residuals[f"res_{outcome}"] = raw - prediction
        joblib.dump(model, model_dir / f"model_{outcome}_{VERSION}.pkl")
        try:
            auc = float(roc_auc_score(raw, prediction)) if len(np.unique(raw)) > 1 else np.nan
        except ValueError:
            auc = np.nan
        diagnostics.append(
            {
                "target": outcome,
                "target_type": "outcome",
                "train_n": int(len(train_y)),
                "test_n": int(len(raw)),
                "brier": float(brier_score_loss(raw, prediction)),
                "auc": auc,
            }
        )
        del model
        gc.collect()
    return diagnostics


def dataset_contract(root: Path, dataset: str) -> dict:
    build_dir = _dcml_paths.workspace_path(root, 'processed_data') / dataset / "build_dataset"
    if dataset == "suning":
        dml_dir = _dcml_paths.workspace_path(root, 'processed_data') / dataset / "DML_Results"
        return {
            "a": build_dir / "DCML_A_20260712.parquet",
            "b": build_dir / "DCML_B_20260712.parquet",
            "c": build_dir / "DCML_C_20260712.parquet",
            "source_schema": build_dir / "DCML_dataset_schema_20260712.json",
            "source_feature_schema": dml_dir / "DML_feature_schema_20260712.json",
            "schema": dml_dir / f"DML_feature_schema_{VERSION}.json",
            "outcomes": ["y_click", "y_cart", "y_purchase"],
            "residual": f"DCML_Residuals_{VERSION}.parquet",
            "gs_residual": f"DCML_GS_Training_Residuals_{VERSION}.parquet",
            "feature_schema": f"DML_feature_schema_{VERSION}.json",
        }
    prefix = f"{dataset}_DCML"
    return {
        "a": build_dir / f"{prefix}_Set_A_{VERSION}.parquet",
        "b": build_dir / f"{prefix}_Set_B_{VERSION}.parquet",
        "c": build_dir / f"{prefix}_Set_C_{VERSION}.parquet",
        "source_schema": build_dir / f"{prefix}_schema_{VERSION}.json",
        "schema": build_dir / f"{prefix}_schema_{VERSION}.json",
        "outcomes": ["y_purchase"],
        "residual": f"{dataset}_DML_Residuals_Final_{VERSION}.parquet",
        "gs_residual": f"{dataset}_DML_GS_Training_Residuals_{VERSION}.parquet",
        "feature_schema": f"{dataset}_DML_feature_schema_{VERSION}.json",
    }


def resolve_treatment_specs(treatments: list[str]) -> tuple[list[str], list[str]]:
    def find(base: str) -> str:
        aliases = [base, f"{base}_calib", f"{base}_calibrated"]
        matches = [column for column in aliases if column in treatments]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one treatment column for {base}, found {matches}")
        return matches[0]

    aggregate = [
        find("T_con_mkt"),
        find("T_con_soc"),
        find("T_con_rat"),
        find("T_int_fac"),
        find("T_int_vis"),
        find("T_int_sem"),
    ]
    disaggregate = [
        find("atomic_a_pri"),
        find("atomic_a_gft"),
        find("atomic_a_sub"),
        find("atomic_a_urg"),
        find("T_con_soc"),
        find("T_con_rat"),
        find("atomic_a_spec"),
        find("atomic_a_str"),
        find("T_int_fac"),
        find("T_int_sem"),
    ]
    return aggregate, disaggregate


def load_analysis_schema(contract: dict, dataset: str) -> dict:
    with open(contract["source_schema"], "r", encoding="utf-8") as f:
        source = json.load(f)
    treatments = list(source["all_treatments"])
    if dataset != "suning":
        return source
    with open(contract["source_feature_schema"], "r", encoding="utf-8") as f:
        frozen_feature_schema = json.load(f)
    aggregate, disaggregate = resolve_treatment_specs(treatments)
    return {
        "dataset": "suning",
        "source_data_version": "20260712_frozen",
        "all_treatments": treatments,
        "aggregate_treatments": aggregate,
        "disaggregate_treatments": disaggregate,
        "confounder_whitelist": list(frozen_feature_schema["confounders"]),
        "gs_confounder_whitelist": list(frozen_feature_schema["confounders"]),
        "semantic_construct": "Behaviorally Revealed Category-Semantic Affinity",
        "semantic_history_event": "pre-event Click for frozen Suning; H0 is treated as unavailable",
    }


def run_nuisance_pipeline(root: Path, dataset: str) -> None:
    contract = dataset_contract(root, dataset)
    logger = setup_logging(ensure_dir(_dcml_paths.workspace_path(root, 'log')), f"dml_step1_{dataset}_{VERSION}")
    start = time.time()
    set_a = read_table(contract["a"])
    set_b = read_table(contract["b"])
    set_c = read_table(contract["c"])
    schema = load_analysis_schema(contract, dataset)
    treatments = list(schema["all_treatments"])
    primary_x = list(schema["confounder_whitelist"])
    gs_x = list(schema.get("gs_confounder_whitelist", primary_x))
    outcomes = list(contract["outcomes"])
    save_dir = ensure_dir(_dcml_paths.workspace_path(root, 'processed_data') / dataset / "DML_Results")
    balance_columns = [*primary_x, "H_level", "M_norm", *treatments, *outcomes]
    balance_roles = {
        **{column: "pre-specified confounder" for column in primary_x},
        **{column: "treatment" for column in treatments},
        **{column: "outcome_descriptive_only" for column in outcomes},
        "H_level": "moderator",
        "M_norm": "moderator_proxy",
    }
    certified, certification_manifest = restrict_analysis_splits(
        root,
        dataset,
        {"Set_A": set_a, "Set_B": set_b, "Set_C": set_c},
        save_dir,
        dataset,
        outcomes=outcomes,
        balance_columns=balance_columns,
        balance_roles=balance_roles,
    )
    set_a, set_b, set_c = certified["Set_A"], certified["Set_B"], certified["Set_C"]
    logger.info(
        "%s certified analysis population: %s Set C rows (%.4f retained)",
        dataset,
        len(set_c),
        certification_manifest["set_c_row_certification_rate"],
    )
    if dataset == "suning":
        for frame in [set_a, set_b, set_c]:
            no_history = pd.to_numeric(frame["H_level"], errors="coerce").fillna(0).eq(0)
            frame.loc[no_history, "T_int_sem"] = np.nan
            frame["T_int_sem_available"] = (~no_history).astype(int)
    if len(treatments) != 12:
        raise RuntimeError(f"{dataset}: expected 12 measurement treatments, found {treatments}")
    if dataset.startswith("amazon") and FORBIDDEN_AMAZON_X.intersection(primary_x + gs_x):
        raise RuntimeError(f"{dataset}: full-period user aggregates entered X")
    for split_name, frame in [("Set A", set_a), ("Set B", set_b), ("Set C", set_c)]:
        required = {"user_id", "item_id", "timestamp", "H_level", "M_norm", *treatments, *outcomes}
        if missing := sorted(required.difference(frame.columns)):
            raise RuntimeError(f"{dataset} {split_name} missing columns: {missing}")
    for cols, train_name, test_name in [(gs_x, "Set A", "Set B"), (primary_x, "Set B", "Set C")]:
        for name, frame in [(train_name, set_a if train_name == "Set A" else set_b), (test_name, set_b if test_name == "Set B" else set_c)]:
            if missing := sorted(set(cols).difference(frame.columns)):
                raise RuntimeError(f"{dataset} {name} missing X whitelist columns: {missing}")

    primary_model_dir = ensure_dir(save_dir / f"nuisance_models_{VERSION}")
    gs_model_dir = ensure_dir(save_dir / f"nuisance_models_gs_A_to_B_{VERSION}")
    logger.info("%s primary X whitelist: %s", dataset, primary_x)
    logger.info("%s GS X whitelist: %s", dataset, gs_x)

    x_a = numeric_matrix(set_a, gs_x)
    x_b_gs = numeric_matrix(set_b, gs_x)
    gs_residuals, gs_diagnostics = treatment_residuals(
        set_a,
        set_b,
        treatments,
        x_a,
        x_b_gs,
        gs_model_dir,
        20260717,
        logger,
    )
    write_table(gs_residuals, save_dir / contract["gs_residual"], logger)

    x_b = numeric_matrix(set_b, primary_x)
    x_c = numeric_matrix(set_c, primary_x)
    residuals, primary_diagnostics = treatment_residuals(
        set_b,
        set_c,
        treatments,
        x_b,
        x_c,
        primary_model_dir,
        20260717,
        logger,
    )
    outcome_diagnostics = outcome_residuals(
        residuals,
        set_b,
        set_c,
        outcomes,
        x_b,
        x_c,
        primary_model_dir,
        20260717,
    )
    write_table(residuals, save_dir / contract["residual"], logger)
    pd.DataFrame(
        [
            dict(
                row,
                residual_role="A_to_B_GS_training",
                analysis_temporal_status=certification_manifest["analysis_temporal_status"],
            )
            for row in gs_diagnostics
        ]
        + [
            dict(
                row,
                residual_role="B_to_C_final",
                analysis_temporal_status=certification_manifest["analysis_temporal_status"],
            )
            for row in primary_diagnostics + outcome_diagnostics
        ]
    ).to_csv(save_dir / f"{dataset}_nuisance_diagnostics_{VERSION}.csv", index=False)
    write_json(
        {
            "dataset": dataset,
            "script_version": VERSION,
            "outcomes": outcomes,
            "treatments": treatments,
            "aggregate_treatments": schema["aggregate_treatments"],
            "disaggregate_treatments": schema["disaggregate_treatments"],
            "confounder_whitelist": primary_x,
            "gs_confounder_whitelist": gs_x,
            "forbidden_amazon_full_period_x": sorted(FORBIDDEN_AMAZON_X),
            "final_residual_design": "nuisance fitted on Set B and evaluated on Set C",
            "gs_residual_design": "nuisance fitted on Set A and evaluated on Set B",
            "semantic_missing_policy": (
                "Suning H0 is N/A under frozen pre-event Click affinity"
                if dataset == "suning"
                else "T_int_sem residual remains NaN when pre-purchase semantic history is unavailable"
            ),
            "analysis_population": certification_manifest["analysis_population"],
            "estimand_scope": certification_manifest["estimand_scope"],
            "analysis_temporal_status": certification_manifest["analysis_temporal_status"],
            "full_sample_mcre_temporal_audit_status": certification_manifest["full_sample_audit_status"],
            "temporal_certification_manifest": certification_manifest["manifest_file"],
            "temporal_certification_item_sha256": certification_manifest["certification_item_sha256"],
            "set_c_item_certification_rate": certification_manifest["set_c_item_certification_rate"],
            "set_c_row_certification_rate": certification_manifest["set_c_row_certification_rate"],
            "source_schema": str(contract["source_schema"]),
            "source_feature_schema": str(contract.get("source_feature_schema", contract["source_schema"])),
        },
        save_dir / contract["feature_schema"],
    )
    logger.info("Completed %s nuisance pipeline in %.2f seconds", dataset, time.time() - start)
