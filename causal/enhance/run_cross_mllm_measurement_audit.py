#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare and evaluate a real cross-MLLM MCRE measurement audit.

This script creates a fixed item list and an output template for one alternative
MLLM, then evaluates the completed alternative-model scores against the Qwen
MCRE key and, optionally, the blinded human-annotation file. Empty templates are
rejected in evaluation mode so software checks cannot be mistaken for evidence.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "20260801"
ENHANCE_DIR = Path(__file__).resolve().parent
CAUSAL_ROOT = ENHANCE_DIR.parent
PROJECT_ROOT = CAUSAL_ROOT.parent
OUTPUT_DIR = _dcml_paths.project_path("enhance/output")

CONSTRUCTS = [
    "T_con_mkt",
    "T_int_vis",
    "T_int_fac",
    "atomic_a_pri",
    "atomic_a_gft",
    "atomic_a_sub",
    "atomic_a_urg",
    "atomic_a_spec",
    "atomic_a_str",
]

CONSTRUCT_LABELS = {
    "T_con_mkt": "Marketing salience",
    "T_int_vis": "Functional visual salience",
    "T_int_fac": "Factual density",
    "atomic_a_pri": "Price shock",
    "atomic_a_gft": "Gift appeal",
    "atomic_a_sub": "Subsidy authenticity",
    "atomic_a_urg": "Urgency",
    "atomic_a_spec": "Specification overlay",
    "atomic_a_str": "Internal-structure view",
}

DEFAULT_CONTEXT_CANDIDATES = [
    _dcml_paths.workspace_path(PROJECT_ROOT, "20260802_results/output/Cross_MLLM_Audit_Input_Items_20260801.csv"),
    _dcml_paths.workspace_path(PROJECT_ROOT, "20260802_results/output/Cross_MLLM_Audit_Input_Items_20260730.csv"),
    _dcml_paths.workspace_path(PROJECT_ROOT, "20260731_results/Cross_MLLM_Audit_Input_Items_20260730.csv"),
    _dcml_paths.project_path(str(_DCMLPath("enhance") / _DCMLPath("output/Cross_MLLM_Audit_Input_Items_20260801.csv"))),
    _dcml_paths.project_path(str(_DCMLPath("enhance") / _DCMLPath("output/Cross_MLLM_Audit_Input_Items_20260730.csv"))),
]

DEFAULT_QWEN_KEY_CANDIDATES = [
    _dcml_paths.workspace_path(CAUSAL_ROOT, "Unified_Visualization/output/MCRE_Human_Annotation_Blinded_Key_20260728.csv"),
    _dcml_paths.workspace_path(PROJECT_ROOT, "20260728_results/MCRE_Human_Annotation_Blinded_Key_20260728.csv"),
]

DEFAULT_QWEN_REFERENCE_CANDIDATES = [
    OUTPUT_DIR / f"Cross_MLLM_Audit_Qwen_Reference_{VERSION}.csv",
    OUTPUT_DIR / "Cross_MLLM_Audit_Agreement_Long_20260730.csv",
    _dcml_paths.workspace_path(PROJECT_ROOT, "20260731_results/Cross_MLLM_Audit_Agreement_Long_20260730.csv"),
    _dcml_paths.workspace_path(PROJECT_ROOT, "20260802_results/output/Cross_MLLM_Audit_Agreement_Long_20260730.csv"),
]


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def qwen_key_path(path: Path | None) -> Path:
    if path is not None:
        return path
    found = first_existing(DEFAULT_QWEN_KEY_CANDIDATES)
    if found is None:
        raise FileNotFoundError("No Qwen MCRE key file found. Provide --qwen-key.")
    return found


def qwen_key_to_long(path: Path) -> pd.DataFrame:
    key = read_csv(path)
    required = {"dataset", "item_id", "construct_code", "mllm_score_0_10"}
    missing = sorted(required.difference(key.columns))
    if missing:
        raise KeyError(f"Qwen key file missing columns: {missing}")
    key = key[key["construct_code"].isin(CONSTRUCTS)].copy()
    key["item_id"] = key["item_id"].astype(str)
    key["qwen_score_0_10"] = pd.to_numeric(key["mllm_score_0_10"], errors="coerce")
    return key[["dataset", "item_id", "construct_code", "qwen_score_0_10"]]


def qwen_reference_to_long(path: Path) -> pd.DataFrame:
    frame = read_csv(path)
    required = {"dataset", "item_id", "construct_code", "qwen_score_0_10"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Qwen reference file missing columns: {missing}")
    frame = frame[frame["construct_code"].isin(CONSTRUCTS)].copy()
    frame["item_id"] = frame["item_id"].astype(str)
    frame["qwen_score_0_10"] = pd.to_numeric(frame["qwen_score_0_10"], errors="coerce")
    return frame[["dataset", "item_id", "construct_code", "qwen_score_0_10"]]


def load_qwen_reference(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.qwen_reference_long is not None:
        path = args.qwen_reference_long
        return qwen_reference_to_long(path), str(path)
    found = first_existing(DEFAULT_QWEN_REFERENCE_CANDIDATES)
    if found is not None:
        return qwen_reference_to_long(found), str(found)
    path = qwen_key_path(args.qwen_key)
    return qwen_key_to_long(path), str(path)


def reference_missing_count(context_keys: pd.DataFrame, reference: pd.DataFrame) -> int:
    merged = context_keys.merge(reference, on=["dataset", "item_id"], how="left")
    expected = len(context_keys) * len(CONSTRUCTS)
    missing_construct_rows = max(0, expected - len(merged))
    missing_scores = int(merged["qwen_score_0_10"].isna().sum()) if "qwen_score_0_10" in merged.columns else expected
    return missing_construct_rows + missing_scores


def matched_qwen_reference(context_keys: pd.DataFrame, qwen_from_key: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    candidates: list[tuple[str, pd.DataFrame]] = [("qwen_key", qwen_from_key)]
    for path in DEFAULT_QWEN_REFERENCE_CANDIDATES[1:]:
        if path.exists():
            try:
                candidates.append((str(path), qwen_reference_to_long(path)))
            except Exception:
                continue
    best_name, best_ref = min(candidates, key=lambda pair: reference_missing_count(context_keys, pair[1]))
    missing = reference_missing_count(context_keys, best_ref)
    if missing:
        raise ValueError(
            f"The fixed audit sample has {missing} missing Qwen reference scores after checking all candidates. "
            "Use --qwen-key or --context-file with matching item coverage."
        )
    return context_keys.merge(best_ref, on=["dataset", "item_id"], how="left"), best_name


def load_context(qwen: pd.DataFrame, context_file: Path | None, items_per_dataset: int, seed: int) -> pd.DataFrame:
    if context_file is None:
        context_file = first_existing(DEFAULT_CONTEXT_CANDIDATES)
    if context_file is not None and context_file.exists():
        context = read_csv(context_file)
        required = {"dataset", "item_id"}
        missing = sorted(required.difference(context.columns))
        if missing:
            raise KeyError(f"Context file missing columns: {missing}")
        context["item_id"] = context["item_id"].astype(str)
    else:
        raise FileNotFoundError(
            "No fixed audit context file with image paths and text excerpts was found. "
            "Provide --context-file or copy Cross_MLLM_Audit_Input_Items_20260801.csv into causal/enhance/output."
        )

    rng = np.random.default_rng(seed)
    sampled = []
    for dataset, group in context.groupby("dataset", sort=True):
        if items_per_dataset > 0 and len(group) > items_per_dataset:
            group = group.sample(
                n=items_per_dataset,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
        sampled.append(group)
    return pd.concat(sampled, ignore_index=True).sort_values(["dataset", "item_id"]).reset_index(drop=True)


def prompt_for_row(row: pd.Series) -> str:
    schema = "\n".join(f"- {code}: {label}" for code, label in CONSTRUCT_LABELS.items())
    text = str(row.get("text_excerpt", "") or "")
    return (
        "You are auditing multimodal product content under a fixed MCRE rubric. "
        "Score each construct from 0 to 10. Use 0 when the cue is absent. "
        "Return only valid JSON with keys schema_valid and scores. "
        "The scores object must contain every construct code below.\n\n"
        f"Constructs:\n{schema}\n\n"
        f"Dataset: {row['dataset']}\n"
        f"Item ID: {row['item_id']}\n"
        f"Image path: {row.get('image_path', '')}\n"
        f"Text excerpt: {text[:1500]}"
    )


def prepare(args: argparse.Namespace) -> None:
    ensure_output_dir()
    qwen = qwen_key_to_long(qwen_key_path(args.qwen_key))
    context = load_context(qwen, args.context_file, args.items_per_dataset, args.seed)

    input_path = OUTPUT_DIR / f"Cross_MLLM_Audit_Input_Items_{VERSION}.csv"
    qwen_reference_path = OUTPUT_DIR / f"Cross_MLLM_Audit_Qwen_Reference_{VERSION}.csv"
    template_path = OUTPUT_DIR / f"Cross_MLLM_Audit_Output_Template_{VERSION}.csv"
    prompt_path = OUTPUT_DIR / f"Cross_MLLM_Audit_Prompts_{VERSION}.jsonl"
    manifest_path = OUTPUT_DIR / f"Cross_MLLM_Audit_Manifest_{VERSION}.json"

    context.to_csv(input_path, index=False)
    context_keys = context[["dataset", "item_id"]].copy()
    context_keys["item_id"] = context_keys["item_id"].astype(str)
    qwen_reference, qwen_reference_source = matched_qwen_reference(context_keys, qwen)
    qwen_reference.to_csv(qwen_reference_path, index=False)

    template = context[["dataset", "item_id"]].copy()
    template.insert(2, "model_id", args.model_id or "")
    template.insert(3, "schema_valid", "")
    for construct in CONSTRUCTS:
        template[f"{construct}_0_10"] = ""
    template["raw_response_or_file"] = ""
    template["notes"] = ""
    template.to_csv(template_path, index=False)

    with prompt_path.open("w", encoding="utf-8") as handle:
        for row in context.itertuples(index=False):
            record = row._asdict()
            series = pd.Series(record)
            handle.write(
                json.dumps(
                    {
                        "dataset": series["dataset"],
                        "item_id": str(series["item_id"]),
                        "model_id": args.model_id or "",
                        "image_path": str(series.get("image_path", "")),
                        "prompt": prompt_for_row(series),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "version": VERSION,
        "mode": "prepare",
        "constructs": CONSTRUCTS,
        "input_items": str(input_path),
        "qwen_reference": str(qwen_reference_path),
        "qwen_reference_source": qwen_reference_source,
        "output_template": str(template_path),
        "prompt_jsonl": str(prompt_path),
        "qwen_key": str(qwen_key_path(args.qwen_key)),
        "context_file": str(args.context_file) if args.context_file else None,
        "items_per_dataset": args.items_per_dataset,
        "model_id_hint": args.model_id,
        "next_step": (
            "Fill the output template with real alternative-MLLM scores, then run "
            "--mode evaluate --alternative-output <filled_csv_or_jsonl>."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Input items: {input_path}")
    print(f"Qwen reference: {qwen_reference_path}")
    print(f"Output template: {template_path}")
    print(f"Prompt JSONL: {prompt_path}")


def alt_csv_to_long(path: Path) -> pd.DataFrame:
    frame = read_csv(path)
    required = {"dataset", "item_id", "model_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Alternative output missing columns: {missing}")
    if "schema_valid" not in frame.columns:
        frame["schema_valid"] = True
    rows = []
    for construct in CONSTRUCTS:
        column = f"{construct}_0_10"
        if column not in frame.columns:
            raise KeyError(f"Alternative output missing score column: {column}")
        tmp = frame[["dataset", "item_id", "model_id", "schema_valid", column]].copy()
        tmp["construct_code"] = construct
        tmp["alternative_score_0_10"] = pd.to_numeric(tmp[column], errors="coerce")
        rows.append(tmp.drop(columns=[column]))
    out = pd.concat(rows, ignore_index=True)
    out["item_id"] = out["item_id"].astype(str)
    out["schema_valid"] = out["schema_valid"].astype(str).str.lower().isin(["true", "1", "yes", "y", "valid"])
    return out


def alt_jsonl_to_long(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            scores = obj.get("scores", obj.get("mcre_scores", {}))
            if not isinstance(scores, dict):
                raise ValueError(f"Line {line_no}: scores must be a JSON object")
            for construct in CONSTRUCTS:
                rows.append(
                    {
                        "dataset": obj.get("dataset"),
                        "item_id": str(obj.get("item_id")),
                        "model_id": obj.get("model_id", ""),
                        "schema_valid": bool(obj.get("schema_valid", True)),
                        "construct_code": construct,
                        "alternative_score_0_10": pd.to_numeric(scores.get(construct), errors="coerce"),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No JSONL records found in {path}")
    return out


def alternative_to_long(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        return alt_jsonl_to_long(path)
    return alt_csv_to_long(path)


def human_to_long(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["dataset", "item_id", "construct_code", "human_median_0_10"])
    frame = read_csv(path)
    frame["item_id"] = frame["item_id"].astype(str)
    if {"dataset", "item_id", "construct_code", "human_median_0_10"}.issubset(frame.columns):
        frame["human_median_0_10"] = pd.to_numeric(frame["human_median_0_10"], errors="coerce")
        return frame[["dataset", "item_id", "construct_code", "human_median_0_10"]]
    required = {"dataset", "item_id", "construct_code", "rater_id", "score_0_10"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(
            "Human annotation file must contain either human_median_0_10 or rater-level scores. "
            f"Missing columns: {missing}"
        )
    frame["score_0_10"] = pd.to_numeric(frame["score_0_10"], errors="coerce")
    return (
        frame.dropna(subset=["score_0_10"])
        .groupby(["dataset", "item_id", "construct_code"], as_index=False)["score_0_10"]
        .median()
        .rename(columns={"score_0_10": "human_median_0_10"})
    )


def spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 3 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return np.nan
    return float(pair["left"].rank(method="average").corr(pair["right"].rank(method="average")))


def pearson(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 3 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return np.nan
    return float(pair["left"].corr(pair["right"]))


def quadratic_weighted_kappa(left: pd.Series, right: pd.Series, levels: int = 11) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if pair.empty:
        return np.nan
    lval = np.clip(np.rint(pair["left"].to_numpy(dtype=float)).astype(int), 0, levels - 1)
    rval = np.clip(np.rint(pair["right"].to_numpy(dtype=float)).astype(int), 0, levels - 1)
    observed = np.zeros((levels, levels), dtype=float)
    np.add.at(observed, (lval, rval), 1.0)
    expected = np.outer(np.bincount(lval, minlength=levels), np.bincount(rval, minlength=levels))
    expected = expected / max(1, len(pair))
    grid = np.arange(levels, dtype=float)
    weights = np.square(grid[:, None] - grid[None, :]) / np.square(levels - 1)
    denominator = float(np.sum(weights * expected))
    return float(1.0 - np.sum(weights * observed) / denominator) if denominator > 0 else np.nan


def tertile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=object)
    finite = numeric.dropna()
    if finite.nunique() < 3:
        return out
    ranks = finite.rank(method="average", pct=True)
    out.loc[finite.index] = pd.cut(
        ranks,
        bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        labels=["low", "mid", "high"],
        include_lowest=True,
    ).astype(str)
    return out


def binary_kappa(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if pair.empty:
        return np.nan
    left_bool = pair["left"].astype(bool)
    right_bool = pair["right"].astype(bool)
    observed = float((left_bool == right_bool).mean())
    p_left = float(left_bool.mean())
    p_right = float(right_bool.mean())
    expected = p_left * p_right + (1.0 - p_left) * (1.0 - p_right)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else np.nan


def zero_preserved_rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    positive = numeric.gt(0) & numeric.notna()
    out.loc[numeric.eq(0)] = 0.0
    if positive.any():
        out.loc[positive] = numeric.loc[positive].rank(method="average", pct=True)
    return out


def add_zero_preserved_rank_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["alternative_score_zprn"] = np.nan
    frame["qwen_score_zprn"] = np.nan
    frame["human_median_zprn"] = np.nan
    group_cols = ["dataset", "construct_code"]
    for _, idx in frame.groupby(group_cols, dropna=False).groups.items():
        idx = list(idx)
        frame.loc[idx, "alternative_score_zprn"] = zero_preserved_rank(frame.loc[idx, "alternative_score_0_10"])
        frame.loc[idx, "qwen_score_zprn"] = zero_preserved_rank(frame.loc[idx, "qwen_score_0_10"])
        if "human_median_0_10" in frame.columns:
            frame.loc[idx, "human_median_zprn"] = zero_preserved_rank(frame.loc[idx, "human_median_0_10"])
    return frame


def agreement_metrics(frame: pd.DataFrame, left_col: str, right_col: str) -> dict:
    pair = frame[[left_col, right_col]].dropna()
    if pair.empty:
        return {
            "N_Compared": 0,
            "Spearman": np.nan,
            "Pearson": np.nan,
            "MAE": np.nan,
            "Weighted_Kappa": np.nan,
            "Tertile_Agreement": np.nan,
            "Large_Error_Rate_abs_ge_2_5": np.nan,
            "Cue_Presence_Agreement": np.nan,
            "Cue_Presence_Kappa": np.nan,
            "Left_Nonzero_Rate": np.nan,
            "Right_Nonzero_Rate": np.nan,
        }
    left_present = pair[left_col].gt(0)
    right_present = pair[right_col].gt(0)
    return {
        "N_Compared": int(len(pair)),
        "Spearman": spearman(pair[left_col], pair[right_col]),
        "Pearson": pearson(pair[left_col], pair[right_col]),
        "MAE": float((pair[left_col] - pair[right_col]).abs().mean()),
        "Weighted_Kappa": quadratic_weighted_kappa(pair[left_col], pair[right_col]),
        "Tertile_Agreement": float((tertile(pair[left_col]) == tertile(pair[right_col])).mean()),
        "Large_Error_Rate_abs_ge_2_5": float((pair[left_col] - pair[right_col]).abs().ge(2.5).mean()),
        "Cue_Presence_Agreement": float((left_present == right_present).mean()),
        "Cue_Presence_Kappa": binary_kappa(left_present, right_present),
        "Left_Nonzero_Rate": float(left_present.mean()),
        "Right_Nonzero_Rate": float(right_present.mean()),
    }


def evaluate(args: argparse.Namespace) -> None:
    ensure_output_dir()
    alt = alternative_to_long(args.alternative_output)
    alt["model_id"] = alt["model_id"].fillna("").astype(str)
    if alt["model_id"].str.strip().eq("").all():
        raise ValueError("Alternative output has blank model_id for all rows. Fill model_id before evaluation.")
    nonmissing = alt["alternative_score_0_10"].notna().sum()
    if nonmissing == 0:
        raise ValueError("Alternative output contains no numeric scores. Do not evaluate an empty template.")
    if args.require_schema_valid and not alt["schema_valid"].all():
        invalid = int((~alt["schema_valid"]).sum())
        total = int(len(alt))
        raise ValueError(
            f"{invalid} of {total} alternative-score rows are not schema_valid. "
            "Fix or rerun those rows before evaluation."
        )

    qwen, qwen_source = load_qwen_reference(args)
    human = human_to_long(args.human_annotation_long)
    merged = alt.merge(qwen, on=["dataset", "item_id", "construct_code"], how="left")
    if merged["qwen_score_0_10"].isna().any():
        missing = int(merged["qwen_score_0_10"].isna().sum())
        raise ValueError(
            f"Alternative scores and Qwen reference do not cover the same item--construct rows. "
            f"Missing Qwen scores: {missing}. Use --qwen-reference-long with the matched reference file."
        )
    if not human.empty:
        merged = merged.merge(human, on=["dataset", "item_id", "construct_code"], how="left")
    else:
        merged["human_median_0_10"] = np.nan
    merged = add_zero_preserved_rank_columns(merged)

    rows = []
    for (model_id, dataset, construct), group in merged.groupby(["model_id", "dataset", "construct_code"], dropna=False):
        base = {
            "model_id": model_id,
            "dataset": dataset,
            "construct_code": construct,
            "construct_name": CONSTRUCT_LABELS.get(construct, construct),
            "Rows": int(len(group)),
            "Schema_Valid_Rate": float(group["schema_valid"].mean()),
        }
        qwen_raw_metrics = agreement_metrics(group, "alternative_score_0_10", "qwen_score_0_10")
        qwen_zprn_metrics = agreement_metrics(group, "alternative_score_zprn", "qwen_score_zprn")
        human_raw_metrics = agreement_metrics(group, "alternative_score_0_10", "human_median_0_10")
        human_zprn_metrics = agreement_metrics(group, "alternative_score_zprn", "human_median_zprn")
        rows.append(
            {
                **base,
                **{f"Alt_vs_Qwen_Raw_{key}": value for key, value in qwen_raw_metrics.items()},
                **{f"Alt_vs_Qwen_ZPRN_{key}": value for key, value in qwen_zprn_metrics.items()},
                **{f"Alt_vs_Human_Raw_{key}": value for key, value in human_raw_metrics.items()},
                **{f"Alt_vs_Human_ZPRN_{key}": value for key, value in human_zprn_metrics.items()},
            }
        )

    summary = pd.DataFrame(rows)
    dataset_summary_rows = []
    for (model_id, dataset), group in merged.groupby(["model_id", "dataset"], dropna=False):
        qwen_raw_metrics = agreement_metrics(group, "alternative_score_0_10", "qwen_score_0_10")
        qwen_zprn_metrics = agreement_metrics(group, "alternative_score_zprn", "qwen_score_zprn")
        human_raw_metrics = agreement_metrics(group, "alternative_score_0_10", "human_median_0_10")
        human_zprn_metrics = agreement_metrics(group, "alternative_score_zprn", "human_median_zprn")
        dataset_summary_rows.append(
            {
                "model_id": model_id,
                "dataset": dataset,
                "Rows": int(len(group)),
                "Schema_Valid_Rate": float(group["schema_valid"].mean()),
                **{f"Alt_vs_Qwen_Raw_{key}": value for key, value in qwen_raw_metrics.items()},
                **{f"Alt_vs_Qwen_ZPRN_{key}": value for key, value in qwen_zprn_metrics.items()},
                **{f"Alt_vs_Human_Raw_{key}": value for key, value in human_raw_metrics.items()},
                **{f"Alt_vs_Human_ZPRN_{key}": value for key, value in human_zprn_metrics.items()},
            }
        )
    dataset_summary = pd.DataFrame(dataset_summary_rows)

    long_path = OUTPUT_DIR / f"Cross_MLLM_Audit_Agreement_Long_{VERSION}.csv"
    summary_path = OUTPUT_DIR / f"Cross_MLLM_Audit_Model_Summary_{VERSION}.csv"
    dataset_summary_path = OUTPUT_DIR / f"Cross_MLLM_Audit_Dataset_Summary_{VERSION}.csv"
    manifest_path = OUTPUT_DIR / f"Cross_MLLM_Audit_Manifest_{VERSION}.json"
    downloads_path = OUTPUT_DIR / f"CROSS_MLLM_RESULT_DOWNLOAD_PATHS_{VERSION}.txt"

    merged.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    dataset_summary.to_csv(dataset_summary_path, index=False)
    manifest = {
        "version": VERSION,
        "mode": "evaluate",
        "alternative_output": str(args.alternative_output),
        "qwen_reference": qwen_source,
        "human_annotation_long": str(args.human_annotation_long) if args.human_annotation_long else None,
        "outputs": [str(long_path), str(summary_path), str(dataset_summary_path)],
        "reporting_boundary": (
            "This audit checks schema portability and measurement agreement for the sampled items, "
            "construct schema, and alternative MLLM that generated the supplied scores. "
            "Raw 0-10 agreement and zero-preserved-rank-normalized agreement are reported separately."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with downloads_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        output_paths = [
            args.alternative_output,
            Path(qwen_source),
            long_path,
            summary_path,
            dataset_summary_path,
            manifest_path,
        ]
        for path in output_paths:
            path = path.resolve()
            if path.exists() and path.is_relative_to(PROJECT_ROOT):
                writer.writerow([str(path.relative_to(PROJECT_ROOT))])
    print(f"Agreement long file: {long_path}")
    print(f"Model summary: {summary_path}")
    print(f"Dataset summary: {dataset_summary_path}")
    print(f"Download list: {downloads_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "evaluate"], default="prepare")
    parser.add_argument("--qwen-key", type=Path, default=None)
    parser.add_argument("--qwen-reference-long", type=Path, default=None)
    parser.add_argument("--context-file", type=Path, default=None)
    parser.add_argument("--items-per-dataset", type=int, default=200)
    parser.add_argument("--model-id", type=str, default="")
    parser.add_argument("--alternative-output", type=Path, default=None)
    parser.add_argument("--human-annotation-long", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--require-schema-valid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        if args.alternative_output is None:
            raise ValueError("--mode evaluate requires --alternative-output")
        evaluate(args)


if __name__ == "__main__":
    main()
