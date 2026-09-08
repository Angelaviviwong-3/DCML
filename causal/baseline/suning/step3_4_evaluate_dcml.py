#!/usr/bin/env python3
"""Evaluate the independently trained DCML recommender on Set C candidates."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_CORE = Path(__file__).resolve().parents[1] / "A_base"
sys.path.insert(0, str(BASELINE_CORE))

from baseline_utils import (  # noqa: E402
    SCRIPT_VERSION,
    STAGES,
    candidate_items,
    candidate_path,
    find_causal_root,
    id_map_paths,
    load_candidates,
    performance_dir,
    read_json,
    setup_logger,
    sha256_file,
    write_json,
)

METRICS = ("HR@10", "NDCG@10", "HR@20", "NDCG@20")


def as_bool(value) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in ("true", "1", "yes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        choices=("Macro",),
        default="Macro",
        help="Retained for command compatibility; the independent DCML recommendation head uses macro cues.",
    )
    return parser.parse_args()


def expected_metrics(scores: np.ndarray, target_index: int) -> tuple[dict[str, float], dict]:
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


def main() -> None:
    parse_args()
    root = find_causal_root(__file__)
    output = performance_dir(root)
    logger = setup_logger(root, "independent_dcml_recommender_evaluation")
    user_map_path, item_map_path = id_map_paths(root)
    global_metadata_path = output / f"DCML_Recommender_Metadata_{SCRIPT_VERSION}.json"
    if not global_metadata_path.is_file():
        raise FileNotFoundError(f"Run step2_train_dcml_recommender.py first: {global_metadata_path}")
    global_metadata = read_json(global_metadata_path)
    if not as_bool(global_metadata.get("independent_recommender", False)) or global_metadata.get("external_backbone") is not None:
        raise RuntimeError("DCML metadata does not describe an independent recommender")
    if as_bool(global_metadata.get("primary_causal_results_used_to_score_set_c", True)):
        raise RuntimeError("Set C primary causal estimates must not be used to score Set C recommendation targets")
    if as_bool(global_metadata.get("set_c_table_read_during_training", True)):
        raise RuntimeError("The Set C table must remain unread during DCML training and model selection")
    if as_bool(global_metadata.get("set_c_candidate_targets_used_for_training_or_selection", True)):
        raise RuntimeError("Set C candidate targets must not be used during training or model selection")
    if not as_bool(global_metadata.get("causal_prior_support_guard_applied", False)):
        raise RuntimeError("The recommendation causal prior must apply the residual-support guard")
    if global_metadata.get("orthogonalization_scoring_basis") != "original_treatment_equivalent":
        raise RuntimeError("The recommendation prior and candidate scoring bases are inconsistent")
    if as_bool(global_metadata.get("prior_stability_gate_uses_set_c", True)):
        raise RuntimeError("Prior stability gating must use pre-C data only")
    required_audits = (
        output / f"DCML_Recommendation_Causal_Prior_{SCRIPT_VERSION}.csv",
        output / f"DCML_Recommendation_Orthogonalization_{SCRIPT_VERSION}.csv",
        output / f"DCML_Recommendation_Prior_Stability_{SCRIPT_VERSION}.csv",
        output / f"DCML_User_State_Audit_{SCRIPT_VERSION}.csv",
    )
    missing_audits = [str(path) for path in required_audits if not path.is_file()]
    if missing_audits:
        raise FileNotFoundError("Missing DCML method-audit outputs: " + "; ".join(missing_audits))
    state_audit = pd.read_csv(output / f"DCML_User_State_Audit_{SCRIPT_VERSION}.csv")
    final_state = state_audit[state_audit["Context"].astype(str) == "Set C start"]
    if len(final_state) != len(STAGES) or (pd.to_numeric(final_state["Corrected_H0_Users"], errors="coerce") != 0).any():
        raise RuntimeError("Set C warm-start users must have nonzero cutoff-correct H levels in every stage")

    summary_rows, user_rows, diagnostic_rows = [], [], []
    for stage in STAGES:
        candidates = load_candidates(root, stage)
        prediction_path = output / f"DCML_{stage}_predictions_{SCRIPT_VERSION}.json"
        metadata_path = output / f"DCML_{stage}_prediction_metadata_{SCRIPT_VERSION}.json"
        if not prediction_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing independent DCML prediction bundle for {stage}")
        predictions = read_json(prediction_path)
        metadata = read_json(metadata_path)
        errors = []
        if set(map(str, predictions)) != set(map(str, candidates)):
            errors.append("prediction user set differs from candidates")
        if not as_bool(metadata.get("independent_recommender", False)):
            errors.append("independent_recommender is not true")
        if metadata.get("external_backbone") is not None:
            errors.append("external_backbone must be null")
        if as_bool(metadata.get("test_outcomes_used_for_training_or_selection", True)):
            errors.append("Set C outcomes were marked as used for training or model selection")
        if as_bool(metadata.get("set_c_table_read_during_training", True)):
            errors.append("Set C table was marked as read during training")
        if as_bool(metadata.get("set_c_candidate_targets_used_for_training_or_selection", True)):
            errors.append("Set C candidate targets were marked as used during training or selection")
        if metadata.get("causal_prior_nuisance_split") != "Set A" or metadata.get("causal_prior_effect_split") != "Set B":
            errors.append("recommendation causal-prior split provenance is invalid")
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
            errors.append("Set C candidate users include an invalid cutoff-state H=0 assignment")
        if metadata.get("orthogonalization_scoring_basis") != "algebraically equivalent coefficients on original treatment values":
            errors.append("candidate causal score is not on the audited original-treatment basis")
        deployed_lambda = float(metadata.get("selected_causal_lambda", 0.0))
        if deployed_lambda > 0 and not (
            as_bool(metadata.get("validation_prior_stability_gate_passed", False))
            and as_bool(metadata.get("refit_prior_stability_gate_passed", False))
        ):
            errors.append("positive causal lambda bypassed the pre-C prior-stability gate")
        if str(metadata.get("candidate_sha256")) != sha256_file(candidate_path(root, stage)):
            errors.append("candidate SHA mismatch")
        if str(metadata.get("prediction_sha256")) != sha256_file(prediction_path):
            errors.append("prediction SHA mismatch")
        if str(metadata.get("user_map_sha256")) != sha256_file(user_map_path):
            errors.append("user map SHA mismatch")
        if str(metadata.get("item_map_sha256")) != sha256_file(item_map_path):
            errors.append("item map SHA mismatch")
        if errors:
            raise RuntimeError(f"Invalid DCML bundle for {stage}: " + " | ".join(errors))

        metric_values: dict[str, list[float]] = defaultdict(list)
        ties, unique_scores, score_sds = [], [], []
        all_tied = 0
        for user, candidate in candidates.items():
            items = [str(item) for item in candidate_items(candidate)]
            by_item = predictions[str(user)]
            if set(by_item) != set(items):
                raise RuntimeError(f"Candidate item mismatch for DCML/{stage}/{user}")
            scores = np.asarray([float(by_item[item]) for item in items], dtype=float)
            if not np.isfinite(scores).all():
                raise RuntimeError(f"Non-finite DCML score for {stage}/{user}")
            metrics, diagnostics = expected_metrics(scores, len(items) - 1)
            for metric, value in metrics.items():
                metric_values[metric].append(value)
            ties.append(int(diagnostics["Target_Tie_Size"]))
            unique_scores.append(int(diagnostics["Unique_Scores"]))
            score_sds.append(float(diagnostics["Score_SD"]))
            all_tied += int(diagnostics["All_Scores_Tied"])
            user_rows.append(
                {
                    "Model": "DCML (Ours)",
                    "Stage": stage,
                    "User_Index": str(user),
                    **metrics,
                    **diagnostics,
                }
            )

        valid = len(candidates) > 0 and all_tied / len(candidates) < 0.99
        row = {
            "Category": "Independent causal-informed multimodal recommender (Proposed)",
            "Model": "DCML (Ours)",
            "Resolution": "Recommendation model",
            "Stage": stage,
            "Candidate_Users": len(candidates),
            "Evaluated_Users": len(candidates),
            "Skipped_Users": 0,
            "Candidate_Set_Size": len(candidate_items(next(iter(candidates.values())))) if candidates else 0,
            "Independent_Recommender": True,
            "External_Backbone": "None",
            "Multi_Task_Training": as_bool(metadata.get("multi_task_training", False)),
            "Training_Split": metadata.get("training_split"),
            "Validation_Split": metadata.get("validation_split"),
            "Evaluation_Split": metadata.get("evaluation_split"),
            "Causal_Prior_Nuisance_Split": metadata.get("causal_prior_nuisance_split"),
            "Causal_Prior_Effect_Split": metadata.get("causal_prior_effect_split"),
            "Selected_Causal_Lambda": metadata.get("selected_causal_lambda"),
            "Validation_Selected_Causal_Lambda": metadata.get("validation_selected_causal_lambda"),
            "Validation_Prior_Stability_Gate_Passed": metadata.get("validation_prior_stability_gate_passed"),
            "Refit_Prior_Stability_Gate_Passed": metadata.get("refit_prior_stability_gate_passed"),
            "Candidate_H0_Users": metadata.get("candidate_h0_users"),
            "Relevance_Retention_Floor": metadata.get("relevance_retention_floor"),
            "Scoring_Protocol": metadata.get("architecture"),
            "Target_Tied_Users": int(sum(value > 1 for value in ties)),
            "All_Scores_Tied_Users": all_tied,
            "All_Scores_Tied_Rate": float(all_tied / len(candidates)),
            "Mean_Target_Tie_Size": float(np.mean(ties)),
            "Mean_Unique_Scores": float(np.mean(unique_scores)),
            "Mean_Within_User_Score_SD": float(np.mean(score_sds)),
            "Valid_Model_Output": valid,
            "Invalid_Reason": "" if valid else "degenerate_scores",
            "Tie_Handling": "expected metric under uniform random tie-breaking",
            "Candidate_SHA256": metadata["candidate_sha256"],
            "Prediction_SHA256": metadata["prediction_sha256"],
            "User_Map_SHA256": metadata["user_map_sha256"],
            "Item_Map_SHA256": metadata["item_map_sha256"],
        }
        for metric in METRICS:
            row.update(aggregate(metric_values[metric], metric))
        summary_rows.append(row)
        diagnostic_rows.append(
            {
                "Stage": stage,
                "Independent_Recommender": True,
                "External_Backbone": "None",
                "Training_Targets": "|".join(metadata.get("training_targets", [])),
                "Training_Edges_By_Stage": json.dumps(metadata.get("training_edges_by_stage", {}), sort_keys=True),
                "Selected_Causal_Lambda": metadata.get("selected_causal_lambda"),
                "Validation_Selected_Causal_Lambda": metadata.get("validation_selected_causal_lambda"),
                "Validation_Prior_Stability_Gate_Passed": metadata.get("validation_prior_stability_gate_passed"),
                "Refit_Prior_Stability_Gate_Passed": metadata.get("refit_prior_stability_gate_passed"),
                "Candidate_H0_Users": metadata.get("candidate_h0_users"),
                "Mean_Estimated_Causal_Score@10": metadata.get("mean_estimated_causal_score_at_10"),
                "Prediction_File": str(prediction_path),
                "Prediction_Metadata_File": str(metadata_path),
            }
        )
        logger.info("DCML/%s users=%d HR@10=%.6f NDCG@10=%.6f", stage, len(candidates), row["HR@10"], row["NDCG@10"])

    pd.DataFrame(summary_rows).to_csv(output / f"Performance_Results_Summary_Hybrid_{SCRIPT_VERSION}.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(output / f"Performance_Results_Hybrid_Diagnostics_{SCRIPT_VERSION}.csv", index=False)
    pd.DataFrame(user_rows).to_csv(output / f"DCML_User_Metrics_{SCRIPT_VERSION}.csv", index=False)
    write_json(
        {
            "script_version": SCRIPT_VERSION,
            "method_role": "independent causal-informed multimodal recommender",
            "independent_recommender": True,
            "external_backbone": None,
            "primary_causal_results_used_to_score_set_c": False,
            "set_c_table_read_during_training": False,
            "set_c_candidate_targets_used_for_training_or_selection": False,
            "causal_prior_support_guard_applied": True,
            "orthogonalization_scoring_basis": global_metadata.get("orthogonalization_scoring_basis"),
            "prior_stability_gate_uses_set_c": False,
            "set_c_candidate_h0_users": global_metadata.get("set_c_candidate_h0_users"),
            "stages": {
                row["Stage"]: {
                    "candidate_sha256": row["Candidate_SHA256"],
                    "prediction_sha256": row["Prediction_SHA256"],
                    "evaluated_users": row["Evaluated_Users"],
                    "selected_causal_lambda": row["Selected_Causal_Lambda"],
                }
                for row in summary_rows
            },
        },
        output / f"DCML_Evaluation_Metadata_{SCRIPT_VERSION}.json",
        indent=2,
    )


if __name__ == "__main__":
    main()
