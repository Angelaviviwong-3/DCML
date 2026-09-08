#!/usr/bin/env python3
"""Audit and compare the independent DCML recommender with Amazon Purchase-only baselines."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_CORE = Path(__file__).resolve().parent / "A_base"
sys.path.insert(0, str(BASELINE_CORE))

from amazon_baseline_utils import (  # noqa: E402
    SCRIPT_VERSION,
    STAGES,
    baseline_data_dir,
    candidate_items,
    candidate_path,
    find_causal_root,
    id_map_paths,
    load_candidates,
    performance_dir,
    prediction_metadata_path,
    prediction_path,
    read_json,
    setup_logger,
    sha256_file,
    write_json,
)

METRICS = ("HR@10", "NDCG@10", "HR@20", "NDCG@20")
MODEL_GROUPS = {
    "Graph and knowledge-aware recommendation": ("MBGCN", "KMCLR"),
    "Multimodal recommendation": ("MGAT", "MICRO", "SLMRec"),
    "Causal and disentangled recommendation": ("DICE", "CIRS", "MGCE"),
}
MODELS = tuple(model for group in MODEL_GROUPS.values() for model in group)
MISSING_COLUMNS = ("Model", "Stage", "Path", "Reason")
INVALID_COLUMNS = ("Model", "Stage", "Reason", "Detail")
PAIRED_COLUMNS = (
    "Stage",
    "Metric",
    "Baseline",
    "Paired_Users",
    "Mean_Difference_DCML_Minus_Baseline",
    "Paired_SE",
    "CI_Lower_95",
    "CI_Upper_95",
    "P_Value",
    "P_Value_Holm",
    "Multiple_Testing_Correction",
    "Direction",
    "Holm_Significant",
    "Interpretation",
)
DCML_AUDIT_COLUMNS = (
    "Stage",
    "Independent_Recommender",
    "External_Backbone",
    "Training_Split",
    "Validation_Split",
    "Evaluation_Split",
    "Test_Outcomes_Used_For_Training_Or_Selection",
    "Set_C_Table_Read_During_Training",
    "Set_C_Candidate_Targets_Used_For_Training_Or_Selection",
    "Causal_Prior_Nuisance_Split",
    "Causal_Prior_Effect_Split",
    "Validation_Selected_Causal_Lambda",
    "Selected_Causal_Lambda",
    "Validation_Prior_Stability_Gate_Passed",
    "Refit_Prior_Stability_Gate_Passed",
    "Candidate_H0_Users",
    "Candidate_Users",
    "Audit_Passed",
    "Audit_Error",
)


def as_bool(value) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in ("true", "1", "yes")


def expected_metrics(scores: np.ndarray, target_index: int):
    target = float(scores[target_index])
    tied = np.isclose(scores, target, rtol=1e-12, atol=1e-12)
    better = int(np.sum((scores > target) & ~tied))
    tie_size = int(tied.sum())
    positions = np.arange(better + 1, better + tie_size + 1, dtype=float)
    metrics = {}
    for k in (10, 20):
        inside = positions <= k
        metrics[f"HR@{k}"] = float(np.mean(inside))
        metrics[f"NDCG@{k}"] = float(np.mean(np.where(inside, 1.0 / np.log2(positions + 1.0), 0.0)))
    return metrics, {
        "Target_Tie_Size": tie_size,
        "Target_Tied": bool(tie_size > 1),
        "All_Scores_Tied": bool(tie_size == len(scores)),
        "Unique_Scores": int(len(np.unique(scores))),
        "Score_SD": float(np.std(scores)),
    }


def aggregate(values: list[float], metric: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return {
        metric: mean,
        f"{metric}_SE": se,
        f"{metric}_CI_Lower": max(0.0, mean - 1.96 * se),
        f"{metric}_CI_Upper": min(1.0, mean + 1.96 * se),
    }


def evaluate_predictions(candidates: dict, predictions: dict, model: str, stage: str):
    if set(map(str, predictions)) != set(map(str, candidates)):
        raise ValueError("prediction user set does not exactly match candidate users")
    values: dict[str, list[float]] = defaultdict(list)
    user_rows, ties, uniques, score_sds = [], [], [], []
    all_tied = 0
    for user, candidate in candidates.items():
        items = [str(item) for item in candidate_items(candidate)]
        by_item = predictions[str(user)]
        if set(by_item) != set(items):
            raise ValueError(f"candidate item set mismatch for user {user}")
        scores = np.asarray([float(by_item[item]) for item in items], dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError(f"non-finite score for user {user}")
        metrics, diagnostics = expected_metrics(scores, len(items) - 1)
        for metric, value in metrics.items():
            values[metric].append(value)
        ties.append(int(diagnostics["Target_Tie_Size"]))
        uniques.append(int(diagnostics["Unique_Scores"]))
        score_sds.append(float(diagnostics["Score_SD"]))
        all_tied += int(diagnostics["All_Scores_Tied"])
        user_rows.append({"Model": model, "Stage": stage, "User_Index": str(user), **metrics, **diagnostics})
    valid = len(candidates) > 0 and all_tied / len(candidates) < 0.99
    summary = {
        "Candidate_Users": len(candidates),
        "Evaluated_Users": len(candidates),
        "Skipped_Missing_User": 0,
        "Skipped_Missing_Target": 0,
        "Skipped_Incomplete_Item_Scores": 0,
        "Skipped_Nonfinite_Scores": 0,
        "Target_Tied_Users": int(sum(value > 1 for value in ties)),
        "All_Scores_Tied_Users": all_tied,
        "All_Scores_Tied_Rate": float(all_tied / len(candidates)),
        "Mean_Target_Tie_Size": float(np.mean(ties)),
        "Mean_Unique_Scores": float(np.mean(uniques)),
        "Mean_Within_User_Score_SD": float(np.mean(score_sds)),
        "Valid_Model_Output": valid,
        "Invalid_Reason": "" if valid else "degenerate_all_tied_scores",
        "Tie_Handling": "expected metric under uniform random tie-breaking",
    }
    for metric in METRICS:
        summary.update(aggregate(values[metric], metric))
    return summary, user_rows


def validate_baseline_bundle(root: Path, model: str, stage: str, candidates: dict, metadata: dict, stage_metadata: dict) -> list[str]:
    errors = []
    prediction = prediction_path(root, model, stage)
    user_map, item_map = id_map_paths(root)
    expected = {
        "script_version": SCRIPT_VERSION,
        "model": model,
        "stage": stage,
        "training_target": f"y_{stage}",
        "candidate_sha256": sha256_file(candidate_path(root, stage)),
        "user_map_sha256": sha256_file(user_map),
        "item_map_sha256": sha256_file(item_map),
        "prediction_sha256": sha256_file(prediction),
    }
    for key, value in expected.items():
        if str(metadata.get(key)) != str(value):
            errors.append(f"{key}: expected {value}, found {metadata.get(key)}")
    if not as_bool(metadata.get("stage_specific_training", False)):
        errors.append("stage_specific_training is not true")
    if int(metadata.get("candidate_users", -1)) != len(candidates):
        errors.append("candidate user count differs from sidecar")
    if int(metadata.get("training_edges", 0)) <= 0:
        errors.append("training_edges is not positive")
    for key in ("first_epoch_loss", "last_epoch_loss"):
        try:
            if not np.isfinite(float(metadata.get(key))):
                errors.append(f"{key} is not finite")
        except (TypeError, ValueError):
            errors.append(f"{key} is missing")
    expected_edges = int(stage_metadata.get("stage_train_edges", 0))
    actual_edges = int(metadata.get("training_edges", 0))
    if model == "CIRS":
        if actual_edges < expected_edges:
            errors.append(f"CIRS training events {actual_edges} are fewer than {expected_edges} unique stage pairs")
    elif expected_edges and actual_edges != expected_edges:
        errors.append(f"training_edges: expected {expected_edges}, found {actual_edges}")
    return errors


def validate_dcml_bundle(root: Path, stage: str, candidates: dict, metadata: dict, prediction_file: Path) -> list[str]:
    errors = []
    user_map, item_map = id_map_paths(root)
    expected = {
        "script_version": SCRIPT_VERSION,
        "model": "DCML (Ours)",
        "stage": stage,
        "candidate_sha256": sha256_file(candidate_path(root, stage)),
        "prediction_sha256": sha256_file(prediction_file),
        "user_map_sha256": sha256_file(user_map),
        "item_map_sha256": sha256_file(item_map),
    }
    for key, value in expected.items():
        if str(metadata.get(key)) != str(value):
            errors.append(f"{key}: expected {value}, found {metadata.get(key)}")
    if not as_bool(metadata.get("independent_recommender", False)):
        errors.append("independent_recommender is not true")
    if metadata.get("external_backbone") is not None:
        errors.append("external_backbone is not null")
    if metadata.get("training_split") != "Set B" or metadata.get("evaluation_split") != "Set C":
        errors.append("train/evaluation split provenance is invalid")
    if as_bool(metadata.get("test_outcomes_used_for_training_or_selection", True)):
        errors.append("Set C outcomes were used in training or model selection")
    if as_bool(metadata.get("set_c_table_read_during_training", True)):
        errors.append("Set C table was read during training")
    if as_bool(metadata.get("set_c_candidate_targets_used_for_training_or_selection", True)):
        errors.append("Set C candidate targets were used in training or model selection")
    if metadata.get("causal_prior_nuisance_split") != "Set A" or metadata.get("causal_prior_effect_split") != "Set B":
        errors.append("training-only DML prior did not use A-to-B sample splitting")
    if metadata.get("reported_causal_estimand_source") != "none; recommendation prior is separate from the primary Set C DML estimates":
        errors.append("primary Set C causal estimates may have leaked into recommendation scoring")
    calibration = metadata.get("score_calibration", {})
    calibration_values = {}
    for key in ("relevance_mean", "relevance_sd", "causal_mean", "causal_sd"):
        try:
            calibration_values[key] = float(calibration.get(key))
            if not np.isfinite(calibration_values[key]):
                errors.append(f"non-finite score calibration: {key}")
        except (TypeError, ValueError):
            errors.append(f"missing score calibration: {key}")
    if any(calibration_values.get(key, 0.0) <= 0 for key in ("relevance_sd", "causal_sd")):
        errors.append("score-calibration standard deviations must be positive")
    if int(metadata.get("item_category_audit", {}).get("missing_items", -1)) != 0:
        errors.append("item-category audit did not establish complete coverage")
    if int(metadata.get("candidate_h0_users", -1)) != 0:
        errors.append("cutoff-correct Set C candidate state contains H=0 users")
    if metadata.get("orthogonalization_scoring_basis") != "algebraically equivalent coefficients on original treatment values":
        errors.append("candidate scores and orthogonalized prior use inconsistent treatment bases")
    deployed_lambda = float(metadata.get("selected_causal_lambda", 0.0))
    if deployed_lambda > 0 and not (
        as_bool(metadata.get("validation_prior_stability_gate_passed", False))
        and as_bool(metadata.get("refit_prior_stability_gate_passed", False))
    ):
        errors.append("positive causal lambda bypassed the pre-C stability gate")
    if int(metadata.get("candidate_users", -1)) != len(candidates):
        errors.append("candidate user count mismatch")
    if as_bool(metadata.get("multi_task_training", True)):
        errors.append("multi_task_training must be false for Amazon Purchase-only prediction")
    return errors


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min((len(order) - rank) * float(p_values[index]), 1.0))
        adjusted[index] = running
    return adjusted


def paired_tests(user_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    user_metrics = user_metrics.copy()
    user_metrics["User_Index"] = user_metrics["User_Index"].astype(str)
    for stage in STAGES:
        dcml = user_metrics[(user_metrics["Model"] == "DCML (Ours)") & (user_metrics["Stage"] == stage)]
        for metric in METRICS:
            left = dcml[["User_Index", metric]].rename(columns={metric: "dcml"})
            for baseline in MODELS:
                right = user_metrics[(user_metrics["Model"] == baseline) & (user_metrics["Stage"] == stage)]
                paired = left.merge(right[["User_Index", metric]].rename(columns={metric: "baseline"}), on="User_Index", validate="one_to_one")
                difference = paired["dcml"].to_numpy(float) - paired["baseline"].to_numpy(float)
                if len(difference) != len(left):
                    raise RuntimeError(f"Incomplete paired user match for {baseline}/{stage}/{metric}")
                mean = float(difference.mean())
                se = float(difference.std(ddof=1) / math.sqrt(len(difference))) if len(difference) > 1 else 0.0
                p_value = math.erfc(abs(mean / se) / math.sqrt(2.0)) if se > 0 else (0.0 if mean else 1.0)
                rows.append(
                    {
                        "Stage": stage,
                        "Metric": metric,
                        "Baseline": baseline,
                        "Paired_Users": len(difference),
                        "Mean_Difference_DCML_Minus_Baseline": mean,
                        "Paired_SE": se,
                        "CI_Lower_95": mean - 1.96 * se,
                        "CI_Upper_95": mean + 1.96 * se,
                        "P_Value": p_value,
                    }
                )
    output = pd.DataFrame(rows)
    output["P_Value_Holm"] = np.nan
    for _, indices in output.groupby(["Stage", "Metric"]).groups.items():
        index = list(indices)
        output.loc[index, "P_Value_Holm"] = holm_adjust(output.loc[index, "P_Value"].to_numpy(float))
    output["Multiple_Testing_Correction"] = "Holm within stage and metric"
    output["Direction"] = np.where(
        output["Mean_Difference_DCML_Minus_Baseline"] > 0,
        "DCML higher",
        np.where(output["Mean_Difference_DCML_Minus_Baseline"] < 0, "DCML lower", "No difference"),
    )
    output["Holm_Significant"] = output["P_Value_Holm"] < 0.05
    output["Interpretation"] = np.where(
        output["Holm_Significant"] & (output["Mean_Difference_DCML_Minus_Baseline"] > 0),
        "DCML significantly higher",
        np.where(
            output["Holm_Significant"] & (output["Mean_Difference_DCML_Minus_Baseline"] < 0),
            "DCML significantly lower",
            "No statistically significant difference",
        ),
    )
    return output[list(PAIRED_COLUMNS)]


def main() -> None:
    root = find_causal_root(__file__)
    logger = setup_logger(root, "independent_dcml_baseline_comparison")
    output = performance_dir(root)
    baseline_dir = baseline_data_dir(root)
    candidate_metadata_path = baseline_dir / f"candidate_metadata_{SCRIPT_VERSION}.json"
    dcml_summary_path = output / f"Performance_Results_Summary_Hybrid_{SCRIPT_VERSION}.csv"
    dcml_user_path = output / f"DCML_User_Metrics_{SCRIPT_VERSION}.csv"
    dcml_global_metadata_path = output / f"DCML_Recommender_Metadata_{SCRIPT_VERSION}.json"
    tuning_path = output / f"DCML_Recommendation_Validation_Tuning_{SCRIPT_VERSION}.csv"
    prior_path = output / f"DCML_Recommendation_Causal_Prior_{SCRIPT_VERSION}.csv"
    projection_path = output / f"DCML_Recommendation_Orthogonalization_{SCRIPT_VERSION}.csv"
    stability_path = output / f"DCML_Recommendation_Prior_Stability_{SCRIPT_VERSION}.csv"
    state_audit_path = output / f"DCML_User_State_Audit_{SCRIPT_VERSION}.csv"
    missing_rows, invalid_rows, baseline_audit_rows, dcml_audit_rows, summary_rows, user_rows = [], [], [], [], [], []

    if not candidate_metadata_path.is_file():
        raise FileNotFoundError(f"Missing candidate metadata: {candidate_metadata_path}")
    candidate_metadata = read_json(candidate_metadata_path)
    for stage in STAGES:
        stage_metadata = candidate_metadata.get("stages", {}).get(stage, {})
        errors = []
        sampling_method = stage_metadata.get("user_sampling_method")
        if sampling_method not in {
            "all_purchase_warm_users",
            "uniform_without_replacement_from_purchase_warm_users",
        }:
            errors.append(f"invalid Purchase-warm user sampling method: {sampling_method}")
        evaluated = int(stage_metadata.get("evaluated_users", -1))
        eligible = int(stage_metadata.get("stage_warm_start_eligible_users", -2))
        if evaluated <= 0 or evaluated > eligible:
            errors.append("evaluated users must be a nonempty subset of Purchase-warm eligible users")
        if str(stage_metadata.get("candidate_sha256")) != sha256_file(candidate_path(root, stage)):
            errors.append("candidate metadata SHA mismatch")
        if errors:
            invalid_rows.append({"Model": "CANDIDATES", "Stage": stage, "Reason": "candidate_audit_failed", "Detail": " | ".join(errors)})

    for category, models in MODEL_GROUPS.items():
        for model in models:
            for stage in STAGES:
                prediction_file = prediction_path(root, model, stage)
                sidecar_file = prediction_metadata_path(root, model, stage)
                for path, reason in ((prediction_file, "prediction_file"), (sidecar_file, "prediction_metadata")):
                    if not path.is_file():
                        missing_rows.append({"Model": model, "Stage": stage, "Path": str(path), "Reason": reason})
                if not prediction_file.is_file() or not sidecar_file.is_file():
                    continue
                candidates = load_candidates(root, stage)
                sidecar = read_json(sidecar_file)
                errors = validate_baseline_bundle(root, model, stage, candidates, sidecar, candidate_metadata["stages"].get(stage, {}))
                try:
                    result, model_user_rows = evaluate_predictions(candidates, read_json(prediction_file), model, stage)
                except Exception as exc:
                    errors.append(str(exc))
                    result, model_user_rows = None, []
                baseline_audit_rows.append(
                    {
                        "Model": model,
                        "Stage": stage,
                        "Bundle_Valid": not errors,
                        "Audit_Error": " | ".join(errors),
                        "Prediction_SHA256": sha256_file(prediction_file),
                        "Training_Target": sidecar.get("training_target"),
                        "Training_Edges": sidecar.get("training_edges"),
                        "First_Epoch_Loss": sidecar.get("first_epoch_loss"),
                        "Last_Epoch_Loss": sidecar.get("last_epoch_loss"),
                    }
                )
                if errors:
                    invalid_rows.append({"Model": model, "Stage": stage, "Reason": "bundle_audit_failed", "Detail": " | ".join(errors)})
                    continue
                if not result["Valid_Model_Output"]:
                    invalid_rows.append({"Model": model, "Stage": stage, "Reason": result["Invalid_Reason"], "Detail": ""})
                stage_metadata = candidate_metadata["stages"].get(stage, {})
                summary_rows.append(
                    {
                        "Category": category,
                        "Model": model,
                        "Stage": stage,
                        "Resolution": "Recommendation baseline",
                        **result,
                        "Independent_Recommender": True,
                        "External_Backbone": "None",
                        "Multi_Task_Training": False,
                        "Training_Target": sidecar["training_target"],
                        "Training_Edges": int(sidecar["training_edges"]),
                        "Training_Protocol": sidecar["training_protocol"],
                        "Prediction_File": str(prediction_file),
                        "Prediction_Metadata_File": str(sidecar_file),
                        "Candidate_SHA256": sidecar["candidate_sha256"],
                        "Positive_Test_Users": stage_metadata.get("positive_test_users"),
                        "Stage_Warm_Start_Eligible_Users": stage_metadata.get("stage_warm_start_eligible_users"),
                        "User_Sampling_Method": stage_metadata.get("user_sampling_method"),
                        "Evaluation_Population": stage_metadata.get("evaluation_population"),
                        "Candidate_Set_Size": int(stage_metadata.get("negatives_per_user", 99)) + 1,
                        "Candidate_Seed": stage_metadata.get("stage_seed"),
                    }
                )
                user_rows.extend(model_user_rows)

    for path, reason in (
        (dcml_summary_path, "dcml_summary"),
        (dcml_user_path, "dcml_user_metrics"),
        (dcml_global_metadata_path, "dcml_recommender_metadata"),
        (tuning_path, "dcml_validation_tuning"),
        (prior_path, "dcml_causal_prior_support_audit"),
        (projection_path, "dcml_orthogonalization_audit"),
        (stability_path, "dcml_prior_stability_audit"),
        (state_audit_path, "dcml_user_state_audit"),
    ):
        if not path.is_file():
            missing_rows.append({"Model": "DCML (Ours)", "Stage": "ALL", "Path": str(path), "Reason": reason})

    if not any(row["Model"] == "DCML (Ours)" for row in missing_rows):
        global_metadata = read_json(dcml_global_metadata_path)
        if not as_bool(global_metadata.get("independent_recommender", False)) or global_metadata.get("external_backbone") is not None:
            invalid_rows.append({"Model": "DCML (Ours)", "Stage": "ALL", "Reason": "not_independent", "Detail": "external model dependency detected"})
        if as_bool(global_metadata.get("primary_causal_results_used_to_score_set_c", True)):
            invalid_rows.append({"Model": "DCML (Ours)", "Stage": "ALL", "Reason": "set_c_causal_leakage", "Detail": "primary Set C estimates were used in Set C ranking"})
        if as_bool(global_metadata.get("set_c_table_read_during_training", True)):
            invalid_rows.append({"Model": "DCML (Ours)", "Stage": "ALL", "Reason": "set_c_training_access", "Detail": "Set C table was read during training"})
        if as_bool(global_metadata.get("set_c_candidate_targets_used_for_training_or_selection", True)):
            invalid_rows.append({"Model": "DCML (Ours)", "Stage": "ALL", "Reason": "set_c_target_leakage", "Detail": "Set C candidate targets were used in training or model selection"})
        if not as_bool(global_metadata.get("causal_prior_support_guard_applied", False)):
            invalid_rows.append({"Model": "DCML (Ours)", "Stage": "ALL", "Reason": "missing_support_guard", "Detail": "causal-prior residual support guard was not applied"})
        if global_metadata.get("orthogonalization_scoring_basis") != "original_treatment_equivalent":
            invalid_rows.append({"Model": "DCML (Ours)", "Stage": "ALL", "Reason": "scoring_basis_mismatch", "Detail": "orthogonalized estimation and candidate scoring bases differ"})
        if as_bool(global_metadata.get("prior_stability_gate_uses_set_c", True)):
            invalid_rows.append({"Model": "DCML (Ours)", "Stage": "ALL", "Reason": "stability_gate_leakage", "Detail": "prior stability gate used Set C"})
        prior = pd.read_csv(prior_path)
        required_prior_columns = {
            "Estimable", "Support_Flag", "Fallback_to_ATE", "Scoring_Basis", "Pure_Basis_Coefficient"
        }
        if not required_prior_columns.issubset(prior.columns):
            invalid_rows.append({"Model": "DCML (Ours)", "Stage": "ALL", "Reason": "incomplete_support_audit", "Detail": "causal-prior support columns are missing"})
        else:
            h0_semantic = prior[
                (prior["Prior_Sample"].astype(str) == "Full Set B")
                & (prior["H_level"].astype(str) == "0")
                & (prior["Component"].astype(str) == "T_int_sem")
            ]
            if not h0_semantic.empty and not h0_semantic["Fallback_to_ATE"].map(as_bool).all():
                invalid_rows.append({"Model": "DCML (Ours)", "Stage": "ALL", "Reason": "unsupported_h0_semantic_prior", "Detail": "Any estimated H=0 semantic prior must fall back to ATE"})
            elif not h0_semantic.empty:
                ate_semantic = prior[
                    (prior["Prior_Sample"].astype(str) == "Full Set B")
                    & (prior["H_level"].astype(str) == "ATE")
                    & (prior["Component"].astype(str) == "T_int_sem")
                ][["Stage", "Coefficient"]].rename(columns={"Coefficient": "ATE_Coefficient"})
                fallback_check = h0_semantic.merge(ate_semantic, on="Stage", validate="one_to_one")
                if not np.allclose(fallback_check["Coefficient"], fallback_check["ATE_Coefficient"], rtol=1e-10, atol=1e-12):
                    invalid_rows.append({"Model": "DCML (Ours)", "Stage": "ALL", "Reason": "invalid_h0_ate_fallback", "Detail": "H=0 semantic scoring coefficients differ from ATE"})
        projection = pd.read_csv(projection_path)
        if len(projection) != 24:
            invalid_rows.append({"Model": "DCML (Ours)", "Stage": "ALL", "Reason": "incomplete_projection_audit", "Detail": f"orthogonalization rows={len(projection)}, expected 24"})
        stability = pd.read_csv(stability_path)
        if len(stability) != 12 or not stability["Gate_Use"].astype(str).str.contains("not used", regex=False).all():
            invalid_rows.append({"Model": "DCML (Ours)", "Stage": "ALL", "Reason": "invalid_stability_audit", "Detail": f"stability rows={len(stability)}, expected 12"})
        state_audit = pd.read_csv(state_audit_path)
        final_state = state_audit[state_audit["Context"].astype(str) == "Set C start"]
        if len(final_state) != len(STAGES) or (pd.to_numeric(final_state["Corrected_H0_Users"], errors="coerce") != 0).any():
            invalid_rows.append({"Model": "DCML (Ours)", "Stage": "ALL", "Reason": "invalid_cutoff_user_state", "Detail": "Set C warm-start users must be H=1/2/3"})
        tuning = pd.read_csv(tuning_path)
        for stage in STAGES:
            selected = tuning[(tuning["Stage"].astype(str) == stage) & tuning["Selected"].map(as_bool)]
            if len(selected) != 1:
                invalid_rows.append({"Model": "DCML (Ours)", "Stage": stage, "Reason": "invalid_validation_selection", "Detail": f"selected rows={len(selected)}"})
            elif not (
                as_bool(selected.iloc[0].get("Meets_Mean_Retention_Floor", False))
                and as_bool(selected.iloc[0].get("Passes_Paired_Noninferiority", False))
                and as_bool(selected.iloc[0].get("Eligible_After_Stability_Gate", False))
            ):
                invalid_rows.append(
                    {
                        "Model": "DCML (Ours)",
                        "Stage": stage,
                        "Reason": "failed_validation_noninferiority",
                        "Detail": "selected causal weight did not pass both validation relevance safeguards",
                    }
                )
            deployed = pd.to_numeric(
                tuning.loc[tuning["Stage"].astype(str) == stage, "Deployed_Causal_Lambda"], errors="coerce"
            ).dropna().unique()
            if len(deployed) != 1:
                invalid_rows.append({"Model": "DCML (Ours)", "Stage": stage, "Reason": "invalid_deployed_lambda", "Detail": f"deployed lambda values={deployed.tolist()}"})
        dcml_summary = pd.read_csv(dcml_summary_path)
        for record in dcml_summary.to_dict("records"):
            stage = str(record["Stage"])
            candidates = load_candidates(root, stage)
            prediction_file = output / f"DCML_{stage}_predictions_{SCRIPT_VERSION}.json"
            metadata_path = output / f"DCML_{stage}_prediction_metadata_{SCRIPT_VERSION}.json"
            if not prediction_file.is_file() or not metadata_path.is_file():
                missing_rows.append({"Model": "DCML (Ours)", "Stage": stage, "Path": str(prediction_file), "Reason": "dcml_prediction_bundle"})
                continue
            metadata = read_json(metadata_path)
            errors = validate_dcml_bundle(root, stage, candidates, metadata, prediction_file)
            dcml_audit_rows.append(
                {
                    "Stage": stage,
                    "Independent_Recommender": as_bool(metadata.get("independent_recommender", False)),
                    "External_Backbone": metadata.get("external_backbone"),
                    "Training_Split": metadata.get("training_split"),
                    "Validation_Split": metadata.get("validation_split"),
                    "Evaluation_Split": metadata.get("evaluation_split"),
                    "Test_Outcomes_Used_For_Training_Or_Selection": as_bool(metadata.get("test_outcomes_used_for_training_or_selection", True)),
                    "Set_C_Table_Read_During_Training": as_bool(metadata.get("set_c_table_read_during_training", True)),
                    "Set_C_Candidate_Targets_Used_For_Training_Or_Selection": as_bool(
                        metadata.get("set_c_candidate_targets_used_for_training_or_selection", True)
                    ),
                    "Causal_Prior_Nuisance_Split": metadata.get("causal_prior_nuisance_split"),
                    "Causal_Prior_Effect_Split": metadata.get("causal_prior_effect_split"),
                    "Validation_Selected_Causal_Lambda": metadata.get("validation_selected_causal_lambda"),
                    "Selected_Causal_Lambda": metadata.get("selected_causal_lambda"),
                    "Validation_Prior_Stability_Gate_Passed": metadata.get("validation_prior_stability_gate_passed"),
                    "Refit_Prior_Stability_Gate_Passed": metadata.get("refit_prior_stability_gate_passed"),
                    "Candidate_H0_Users": metadata.get("candidate_h0_users"),
                    "Candidate_Users": len(candidates),
                    "Audit_Passed": not errors,
                    "Audit_Error": " | ".join(errors),
                }
            )
            if errors:
                invalid_rows.append({"Model": "DCML (Ours)", "Stage": stage, "Reason": "dcml_bundle_audit_failed", "Detail": " | ".join(errors)})
            stage_metadata = candidate_metadata["stages"].get(stage, {})
            record.update(
                {
                    "Training_Target": "y_purchase",
                    "Training_Edges": sum(int(value) for value in metadata.get("training_edges_by_stage", {}).values()),
                    "Training_Protocol": metadata.get("architecture"),
                    "Prediction_File": str(prediction_file),
                    "Prediction_Metadata_File": str(metadata_path),
                    "Positive_Test_Users": stage_metadata.get("positive_test_users"),
                    "Stage_Warm_Start_Eligible_Users": stage_metadata.get("stage_warm_start_eligible_users"),
                    "User_Sampling_Method": stage_metadata.get("user_sampling_method"),
                    "Evaluation_Population": stage_metadata.get("evaluation_population"),
                    "Candidate_Seed": stage_metadata.get("stage_seed"),
                }
            )
            summary_rows.append(record)
        dcml_users = pd.read_csv(dcml_user_path, dtype={"User_Index": str})
        dcml_users["User_Index"] = dcml_users["User_Index"].astype(str)
        user_rows.extend(dcml_users.to_dict("records"))

    table = pd.DataFrame(summary_rows)
    users = pd.DataFrame(user_rows)
    missing = pd.DataFrame(missing_rows, columns=MISSING_COLUMNS)
    invalid = pd.DataFrame(invalid_rows, columns=INVALID_COLUMNS)
    baseline_audit = pd.DataFrame(baseline_audit_rows)
    dcml_audit = pd.DataFrame(dcml_audit_rows, columns=DCML_AUDIT_COLUMNS)
    paired = pd.DataFrame(columns=PAIRED_COLUMNS)
    quality_errors = []
    if len(table) != 9:
        quality_errors.append(f"Table rows={len(table)}, expected 9")
    if len(baseline_audit) != 8 or (not baseline_audit.empty and not baseline_audit["Bundle_Valid"].all()):
        quality_errors.append("Not all 8 Amazon Purchase baseline prediction bundles passed audit")
    if len(dcml_audit) != 1 or (not dcml_audit.empty and not dcml_audit["Audit_Passed"].all()):
        quality_errors.append("The independent Amazon Purchase DCML prediction bundle did not pass audit")
    if not missing.empty:
        quality_errors.append(f"Missing files={len(missing)}")
    if not invalid.empty:
        quality_errors.append(f"Invalid outputs={len(invalid)}")
    if not users.empty and not quality_errors:
        users["User_Index"] = users["User_Index"].astype(str)
        for stage in STAGES:
            expected_users = set(load_candidates(root, stage))
            for model in (*MODELS, "DCML (Ours)"):
                actual = set(users.loc[(users["Stage"] == stage) & (users["Model"] == model), "User_Index"])
                if actual != expected_users:
                    quality_errors.append(f"User set mismatch: {model}/{stage}")
        if not quality_errors:
            paired = paired_tests(users)
            if len(paired) != 32 or (paired["Paired_Users"] <= 0).any():
                quality_errors.append(f"Paired test rows={len(paired)}, expected 32")

    candidate_sensitivity_path = output / f"Recommendation_Candidate_Sensitivity_{SCRIPT_VERSION}.csv"
    candidate_sensitivity_metadata_path = output / f"Recommendation_Candidate_Sensitivity_Metadata_{SCRIPT_VERSION}.json"
    training_seed_path = output / f"Recommendation_Training_Seed_Runs_{SCRIPT_VERSION}.csv"
    training_seed_summary_path = output / f"Recommendation_Training_Seed_Summary_{SCRIPT_VERSION}.csv"
    candidate_sensitivity_complete = False
    if candidate_sensitivity_path.is_file() and candidate_sensitivity_metadata_path.is_file():
        sensitivity = pd.read_csv(candidate_sensitivity_path)
        sensitivity_metadata = read_json(candidate_sensitivity_metadata_path)
        candidate_sensitivity_complete = bool(
            len(sensitivity) == 99
            and int(sensitivity_metadata.get("actual_rows", -1)) == 99
            and as_bool(sensitivity_metadata.get("all_score_files_hashed", False))
        )
    minimum_seed_runs = 0
    if training_seed_path.is_file():
        seed_results = pd.read_csv(training_seed_path)
        counts = seed_results.groupby(["Stage", "Model"])["Seed_Offset"].nunique()
        if len(counts) == 9:
            minimum_seed_runs = int(counts.min())

    method_audit_reasons = {
        "missing_support_guard",
        "scoring_basis_mismatch",
        "stability_gate_leakage",
        "incomplete_support_audit",
        "unsupported_h0_semantic_prior",
        "invalid_h0_ate_fallback",
        "incomplete_projection_audit",
        "invalid_stability_audit",
        "invalid_cutoff_user_state",
        "invalid_deployed_lambda",
    }
    method_audit_passed = bool(
        not invalid.empty and not invalid["Reason"].astype(str).isin(method_audit_reasons).any()
    ) if len(invalid) else True

    if training_seed_summary_path.is_file() and not table.empty:
        seed_summary = pd.read_csv(training_seed_summary_path)
        if len(seed_summary) == 9:
            table = table.merge(seed_summary, on=["Stage", "Model"], how="left", validate="one_to_one")
        else:
            quality_errors.append(f"Training seed summary rows={len(seed_summary)}, expected 9")
    elif not table.empty:
        table["Training_Seed_Runs"] = 0

    if not table.empty:
        table["Inference_Unit"] = "user"
        table["CI_Method"] = "normal approximation using user-level standard errors"
        table["stage_order"] = table["Stage"].map({stage: index for index, stage in enumerate(STAGES)})
        table = table.sort_values(["stage_order", "Category", "Model"]).drop(columns="stage_order")
    table.to_csv(output / f"Table4_Baseline_Comparison_Full_{SCRIPT_VERSION}.csv", index=False)
    user_metrics_path = output / f"Recommendation_User_Metrics_{SCRIPT_VERSION}.csv"
    users.to_csv(user_metrics_path, index=False)
    paired.to_csv(output / f"Paired_DCML_vs_Baselines_{SCRIPT_VERSION}.csv", index=False)
    baseline_audit.to_csv(output / f"Prediction_Bundle_Audit_{SCRIPT_VERSION}.csv", index=False)
    dcml_audit.to_csv(output / f"DCML_Independent_Model_Audit_{SCRIPT_VERSION}.csv", index=False)
    missing.to_csv(output / f"Missing_Baseline_Predictions_{SCRIPT_VERSION}.csv", index=False)
    invalid.to_csv(output / f"Invalid_Baseline_Outputs_{SCRIPT_VERSION}.csv", index=False)
    write_json(
        {
            "script_version": SCRIPT_VERSION,
            "file": str(user_metrics_path),
            "sha256": sha256_file(user_metrics_path),
            "rows": int(len(users)),
            "expected_model_stage_bundles": 9,
            "actual_model_stage_bundles": int(users.groupby(["Model", "Stage"]).ngroups) if not users.empty else 0,
            "dcml_user_metric_source": str(dcml_user_path),
            "dcml_user_metric_source_sha256": sha256_file(dcml_user_path) if dcml_user_path.is_file() else None,
        },
        output / f"Recommendation_User_Metrics_Metadata_{SCRIPT_VERSION}.json",
        indent=2,
    )
    write_json(
        {
            "script_version": SCRIPT_VERSION,
            "passed": not quality_errors,
            "quality_errors": quality_errors,
            "expected_table_rows": 9,
            "actual_table_rows": len(table),
            "expected_baseline_bundles": 8,
            "actual_baseline_bundles": len(baseline_audit),
            "expected_independent_dcml_bundles": 1,
            "actual_independent_dcml_bundles": len(dcml_audit),
            "expected_paired_tests": 32,
            "actual_paired_tests": len(paired),
            "dcml_independent_model_audit_passed": bool(len(dcml_audit) == 1 and dcml_audit["Audit_Passed"].all()),
            "set_c_causal_estimate_leakage_detected": bool(
                not dcml_audit.empty and dcml_audit["Test_Outcomes_Used_For_Training_Or_Selection"].any()
            ),
            "set_c_training_access_detected": bool(
                not dcml_audit.empty
                and (
                    dcml_audit["Set_C_Table_Read_During_Training"].any()
                    or dcml_audit["Set_C_Candidate_Targets_Used_For_Training_Or_Selection"].any()
                )
            ),
            "candidate_sensitivity_complete": candidate_sensitivity_complete,
            "dcml_method_audit_passed": method_audit_passed,
            "causal_prior_support_guard_verified": bool(
                not invalid["Reason"].astype(str).isin({"missing_support_guard", "incomplete_support_audit", "unsupported_h0_semantic_prior", "invalid_h0_ate_fallback"}).any()
            ) if not invalid.empty else True,
            "cutoff_user_state_audit_verified": bool(
                not invalid["Reason"].astype(str).eq("invalid_cutoff_user_state").any()
            ) if not invalid.empty else True,
            "orthogonalization_scoring_basis_verified": bool(
                not invalid["Reason"].astype(str).isin({"scoring_basis_mismatch", "incomplete_projection_audit"}).any()
            ) if not invalid.empty else True,
            "pre_c_prior_stability_gate_verified": bool(
                not invalid["Reason"].astype(str).isin({"stability_gate_leakage", "invalid_stability_audit"}).any()
            ) if not invalid.empty else True,
            "minimum_training_seed_runs_per_model_stage": minimum_seed_runs,
            "publication_training_randomness_target_met": bool(minimum_seed_runs >= 5),
            "publication_robustness_complete": bool(
                not quality_errors and method_audit_passed and candidate_sensitivity_complete and minimum_seed_runs >= 5
            ),
            "interpretation_guardrail": "statistical significance must be interpreted together with the sign and magnitude of DCML-minus-baseline differences",
        },
        output / f"Comparison_Quality_Gate_{SCRIPT_VERSION}.json",
        indent=2,
    )
    logger.info("Wrote independent DCML comparison outputs to %s", output)
    if quality_errors:
        raise RuntimeError("Comparison quality gate failed: " + "; ".join(quality_errors))


if __name__ == "__main__":
    main()
