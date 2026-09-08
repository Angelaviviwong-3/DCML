#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent utilities for the Suning stage-specific baseline rerun."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "20260717"
DATA_VERSION = "20260712"
CAUSAL_RESULT_VERSION = "20260717"
DATASET = "suning"
STAGES = ("click", "cart", "purchase")
BASE_SEED = 20260717


CAUSAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAUSAL_ROOT))

from temporal_certification import filter_certified_frame, load_certification


def training_seed_offset() -> int:
    raw = os.environ.get("SUNING_TRAINING_SEED_OFFSET", "0")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"SUNING_TRAINING_SEED_OFFSET must be an integer, found {raw!r}") from exc


def find_causal_root(anchor_file: str | os.PathLike | None = None) -> Path:
    env_root = os.environ.get("DCML_CAUSAL_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    anchors = [Path(anchor_file).resolve()] if anchor_file is not None else []
    anchors.append(Path.cwd().resolve())
    for anchor in anchors:
        for parent in [anchor.parent, *anchor.parents] if anchor.is_file() else [anchor, *anchor.parents]:
            if parent.name == "causal":
                return parent
            candidate = parent / "causal"
            if candidate.exists():
                return candidate.resolve()
    raise RuntimeError("Cannot locate causal root. Set DCML_CAUSAL_ROOT explicitly.")


def ensure_dir(path: str | os.PathLike) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_logger(root: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(f"{name}_{SCRIPT_VERSION}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
    for handler in (
        logging.FileHandler(ensure_dir(_dcml_paths.workspace_path(root, 'log')) / f"{name}_{SCRIPT_VERSION}.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def baseline_data_dir(root: Path) -> Path:
    return ensure_dir(_dcml_paths.workspace_path(root, 'results_of_comparison') / DATASET / "baseline_data")


def performance_dir(root: Path) -> Path:
    return ensure_dir(_dcml_paths.workspace_path(root, 'results_of_comparison') / DATASET / "dcml_performance")


def cache_dir(root: Path, cache_name: str) -> Path:
    return ensure_dir(baseline_data_dir(root) / f"{cache_name}_{SCRIPT_VERSION}")


def stage_cache_dir(root: Path, cache_name: str, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    return ensure_dir(cache_dir(root, cache_name) / f"stage_{stage}")


def result_input_paths(root: Path, filename: str, version: str = DATA_VERSION) -> list[Path]:
    return [
        _dcml_paths.workspace_path(root, 'processed_data') / DATASET / "build_dataset" / filename,
        _dcml_paths.workspace_path(root, 'processed_data') / DATASET / "DML_Results" / filename,
        _dcml_paths.workspace_path(root, 'processed_data') / DATASET / "Final_Causal_Output" / filename,
        _dcml_paths.workspace_path(root.parent, f"{version}_results", filename),
    ]


def read_table(paths: Iterable[Path]) -> pd.DataFrame:
    tried: list[str] = []
    expanded: list[Path] = []
    for path in paths:
        expanded.append(path)
        if path.suffix == ".parquet":
            expanded.append(path.with_suffix(".csv"))
        elif path.suffix == ".csv":
            expanded.append(path.with_suffix(".parquet"))
    seen: set[Path] = set()
    for path in expanded:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            tried.append(str(path))
            continue
        if path.suffix == ".parquet":
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                tried.append(f"{path}: {exc}")
                continue
        if path.suffix == ".csv":
            for encoding in ("utf-8", "utf-8-sig", "gb18030"):
                try:
                    return pd.read_csv(path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
                except Exception as exc:
                    tried.append(f"{path} [{encoding}]: {exc}")
                    break
    raise FileNotFoundError("Cannot read any input table. Tried: " + "; ".join(tried[:16]))


def load_train_table(root: Path) -> pd.DataFrame:
    frame = read_table(result_input_paths(root, f"DCML_B_{DATA_VERSION}.parquet"))
    certified, _, _ = load_certification(root, DATASET)
    return filter_certified_frame(frame, certified)


def load_test_table(root: Path) -> pd.DataFrame:
    frame = read_table(result_input_paths(root, f"DCML_C_{DATA_VERSION}.parquet"))
    certified, audit, _ = load_certification(root, DATASET)
    filtered = filter_certified_frame(frame, certified)
    expected = int(audit["set_c_rows_on_temporally_certified_items"])
    if len(filtered) != expected:
        raise RuntimeError(f"Certified Set C row mismatch: {len(filtered)} != {expected}")
    return filtered


def load_master_table(root: Path) -> pd.DataFrame:
    frame = read_table(result_input_paths(root, f"DCML_Table_{DATA_VERSION}.parquet"))
    certified, _, _ = load_certification(root, DATASET)
    return filter_certified_frame(frame, certified)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(obj, path: Path, *, indent: int | None = None) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=indent)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_map_paths(root: Path) -> tuple[Path, Path]:
    base = baseline_data_dir(root)
    return base / f"user2id_{SCRIPT_VERSION}.json", base / f"item2id_{SCRIPT_VERSION}.json"


def load_id_maps(root: Path) -> tuple[dict[str, int], dict[str, int]]:
    user_path, item_path = id_map_paths(root)
    if not user_path.exists() or not item_path.exists():
        raise FileNotFoundError("Run step1_prepare_baseline_data.py before model preparation.")
    return read_json(user_path), read_json(item_path)


def candidate_path(root: Path, stage: str) -> Path:
    path = baseline_data_dir(root) / f"test_candidates_{stage}_{SCRIPT_VERSION}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {stage} candidate file: {path}")
    return path


def load_candidates(root: Path, stage: str) -> dict:
    return read_json(candidate_path(root, stage))


def candidate_items(data: dict) -> list[int]:
    return [int(value) for value in data["negatives"]] + [int(data["target"])]


def prediction_path(root: Path, model: str, stage: str) -> Path:
    return baseline_data_dir(root) / f"{model}_{stage}_predictions_{SCRIPT_VERSION}.json"


def prediction_metadata_path(root: Path, model: str, stage: str) -> Path:
    return baseline_data_dir(root) / f"{model}_{stage}_prediction_metadata_{SCRIPT_VERSION}.json"


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stage_seed(stage: str, offset: int = 0) -> int:
    return BASE_SEED + training_seed_offset() + STAGES.index(stage) * 1000 + offset


def map_positive_pairs(
    frame: pd.DataFrame,
    user2id: dict[str, int],
    item2id: dict[str, int],
    stage: str,
) -> np.ndarray:
    mask = frame[f"y_{stage}"].fillna(0).astype(int) == 1
    pairs = []
    for row in frame.loc[mask, ["user_id", "item_id"]].itertuples(index=False):
        user = str(row.user_id)
        item = str(row.item_id)
        if user in user2id and item in item2id:
            pairs.append((int(user2id[user]), int(item2id[item])))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def unique_pairs(pairs: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted({(int(u), int(i)) for u, i in pairs}), dtype=np.int64)


def write_pair_file(path: Path, pairs: np.ndarray, sep: str = " ") -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        for user, item in pairs:
            handle.write(f"{int(user)}{sep}{int(item)}\n")


def find_first_col(frame: pd.DataFrame, prefix_or_name: str) -> str | None:
    if prefix_or_name in frame.columns:
        return prefix_or_name
    return next((col for col in frame.columns if col.startswith(prefix_or_name)), None)


def feature_matrix(
    master: pd.DataFrame,
    item2id: dict[str, int],
    columns: Iterable[str | None],
) -> np.ndarray:
    clean = [col for col in columns if col is not None and col in master.columns]
    if not clean:
        return np.zeros((len(item2id), 1), dtype=np.float32)
    item_frame = master[["item_id", *clean]].drop_duplicates("item_id")
    matrix = np.zeros((len(item2id), len(clean)), dtype=np.float32)
    for row in item_frame.itertuples(index=False):
        raw_item = str(row[0])
        if raw_item not in item2id:
            continue
        values = np.nan_to_num(np.asarray(row[1:], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        matrix[int(item2id[raw_item])] = values
    mean = matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True)
    scale[scale <= 1e-12] = 1.0
    matrix = ((matrix - mean) / scale).astype(np.float32)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def suning_feature_columns(master: pd.DataFrame) -> dict[str, list[str | None]]:
    return {
        "visual": [find_first_col(master, "T_int_vis_calib"), find_first_col(master, "atomic_a_spec_calib"), find_first_col(master, "atomic_a_str_calib")],
        "text": [find_first_col(master, "T_int_fac_calib"), find_first_col(master, "T_int_sem")],
        "audio": [find_first_col(master, "T_con_mkt_calib")],
        "image": [find_first_col(master, "T_int_vis_calib"), find_first_col(master, "atomic_a_spec_calib"), find_first_col(master, "atomic_a_str_calib")],
    }


def export_prediction_bundle(
    root: Path,
    model: str,
    stage: str,
    predictions: dict[str, dict[str, float]],
    *,
    training_edges: int,
    training_protocol: str,
    extra_metadata: dict | None = None,
) -> dict:
    candidates = load_candidates(root, stage)
    _, certification_summary, _ = load_certification(root, DATASET)
    if set(predictions) != set(candidates):
        missing = len(set(candidates) - set(predictions))
        extra = len(set(predictions) - set(candidates))
        raise ValueError(f"{model}/{stage} user mismatch: missing={missing}, extra={extra}")
    all_tied = 0
    score_stds: list[float] = []
    unique_scores: list[int] = []
    for user, data in candidates.items():
        expected_items = {str(item) for item in candidate_items(data)}
        actual_items = set(predictions[user])
        if actual_items != expected_items:
            raise ValueError(f"{model}/{stage}/{user} candidate-score mismatch")
        values = np.asarray([float(predictions[user][item]) for item in expected_items], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{model}/{stage}/{user} contains non-finite scores")
        score_stds.append(float(np.std(values)))
        unique_scores.append(int(len(np.unique(values))))
        all_tied += int(np.ptp(values) <= 1e-12)
    if predictions and all_tied == len(predictions):
        raise RuntimeError(f"{model}/{stage} produced all-tied predictions")
    output = prediction_path(root, model, stage)
    write_json(predictions, output)
    user_path, item_path = id_map_paths(root)
    metadata = {
        "script_version": SCRIPT_VERSION,
        "data_version": DATA_VERSION,
        "model": model,
        "stage": stage,
        "stage_specific_training": True,
        "training_target": f"y_{stage}",
        "training_edges": int(training_edges),
        "training_protocol": training_protocol,
        "candidate_users": int(len(candidates)),
        "candidate_file": str(candidate_path(root, stage)),
        "candidate_sha256": sha256_file(candidate_path(root, stage)),
        "user_map_sha256": sha256_file(user_path),
        "item_map_sha256": sha256_file(item_path),
        "prediction_file": str(output),
        "prediction_sha256": sha256_file(output),
        "all_tied_users": int(all_tied),
        "all_tied_rate": float(all_tied / max(len(predictions), 1)),
        "mean_unique_scores": float(np.mean(unique_scores)) if unique_scores else None,
        "mean_within_user_score_sd": float(np.mean(score_stds)) if score_stds else None,
        "analysis_temporal_status": "PASS_BY_CONSTRUCTION_CERTIFIED_ITEMS_ONLY",
        "analysis_population": "items with all selected MCRE reviews strictly before item-specific Set C entry",
        "full_sample_audit_status": certification_summary.get("audit_status"),
        "set_c_item_certification_rate": certification_summary.get("set_c_item_certification_rate"),
        "set_c_row_certification_rate": certification_summary.get("set_c_row_certification_rate"),
        "temporal_certification_item_file": certification_summary.get("item_certification_file"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    write_json(metadata, prediction_metadata_path(root, model, stage), indent=2)
    return metadata
