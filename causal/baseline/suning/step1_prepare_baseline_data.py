#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build independent stage-specific Suning recommendation candidates."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CAUSAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAUSAL_ROOT))

from temporal_certification import restrict_analysis_splits

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable


SCRIPT_VERSION = "20260717"
DATA_VERSION = "20260712"
STAGES = ("click", "cart", "purchase")
BASE_SEED = 20260717


def find_causal_root(anchor_file: str | os.PathLike | None = None) -> Path:
    if os.environ.get("DCML_CAUSAL_ROOT"):
        return Path(os.environ["DCML_CAUSAL_ROOT"]).expanduser().resolve()
    anchors = [Path(anchor_file).resolve()] if anchor_file is not None else []
    anchors.append(Path.cwd().resolve())
    for anchor in anchors:
        for parent in [anchor.parent, *anchor.parents] if anchor.is_file() else [anchor, *anchor.parents]:
            if parent.name == "causal":
                return parent
            if (parent / "causal").exists():
                return (parent / "causal").resolve()
    raise RuntimeError("Cannot locate causal root. Set DCML_CAUSAL_ROOT.")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(root: Path) -> logging.Logger:
    logger = logging.getLogger(f"suning_baseline_candidates_{SCRIPT_VERSION}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
    for handler in (
        logging.FileHandler(ensure_dir(_dcml_paths.workspace_path(root, 'log')) / f"suning_baseline_candidates_{SCRIPT_VERSION}.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def input_paths(root: Path, filename: str) -> list[Path]:
    return [
        _dcml_paths.workspace_path(root, 'processed_data') / "suning" / "build_dataset" / filename,
        _dcml_paths.workspace_path(root.parent, f"{DATA_VERSION}_results", filename),
    ]


def read_table(paths: list[Path]) -> pd.DataFrame:
    attempted = []
    for original in paths:
        candidates = [original]
        candidates.append(original.with_suffix(".csv") if original.suffix == ".parquet" else original.with_suffix(".parquet"))
        for path in candidates:
            if not path.exists():
                attempted.append(str(path))
                continue
            if path.suffix == ".parquet":
                try:
                    return pd.read_parquet(path)
                except Exception as exc:
                    attempted.append(f"{path}: {exc}")
            else:
                for encoding in ("utf-8", "utf-8-sig", "gb18030"):
                    try:
                        return pd.read_csv(path, encoding=encoding)
                    except UnicodeDecodeError:
                        continue
    raise FileNotFoundError("Cannot read input table: " + "; ".join(attempted[:12]))


def write_json(obj, path: Path, indent: int | None = None) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=indent)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in ("timestamp", "user_id", "item_id") if col in frame.columns]
    return frame.sort_values(columns, kind="mergesort") if columns else frame


def positive_pairs(frame: pd.DataFrame, stage: str, user2id: dict[str, int], item2id: dict[str, int]) -> np.ndarray:
    positive = frame[frame[f"y_{stage}"].fillna(0).astype(int) == 1]
    pairs = {
        (int(user2id[str(row.user_id)]), int(item2id[str(row.item_id)]))
        for row in positive[["user_id", "item_id"]].itertuples(index=False)
    }
    return np.asarray(sorted(pairs), dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)


def export_interactions(frame: pd.DataFrame, user2id: dict[str, int], item2id: dict[str, int], out_dir: Path, split: str) -> None:
    for stage in STAGES:
        grouped: dict[int, list[int]] = defaultdict(list)
        for user, item in positive_pairs(frame, stage, user2id, item2id):
            grouped[int(user)].append(int(item))
        with open(out_dir / f"{split}_{stage}_{SCRIPT_VERSION}.txt", "w", encoding="utf-8") as handle:
            for user in sorted(grouped):
                handle.write(f"{user} {' '.join(str(item) for item in grouped[user])}\n")


def all_observed_positives(frame: pd.DataFrame, user2id: dict[str, int], item2id: dict[str, int]) -> dict[int, set[int]]:
    interacted: dict[int, set[int]] = defaultdict(set)
    mask = frame[[f"y_{stage}" for stage in STAGES]].fillna(0).astype(int).sum(axis=1) > 0
    for row in frame.loc[mask, ["user_id", "item_id"]].itertuples(index=False):
        interacted[int(user2id[str(row.user_id)])].add(int(item2id[str(row.item_id)]))
    return interacted


def build_stage_candidates(
    test: pd.DataFrame,
    stage: str,
    user2id: dict[str, int],
    item2id: dict[str, int],
    stage_train_pairs: np.ndarray,
    global_interacted: dict[int, set[int]],
    max_users: int,
    negatives: int,
    seed: int,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    train_users = {int(user) for user, _ in stage_train_pairs}
    train_items = np.asarray(sorted({int(item) for _, item in stage_train_pairs}), dtype=int)
    train_item_set = set(train_items.tolist())
    if len(train_items) <= negatives:
        raise ValueError(f"{stage}: stage-specific warm item pool is too small")

    positive = sort_frame(test[test[f"y_{stage}"].fillna(0).astype(int) == 1].copy())
    positive["_user"] = positive["user_id"].astype(str)
    positive["_item"] = positive["item_id"].astype(str)
    positive["_uid"] = positive["_user"].map(user2id)
    positive["_iid"] = positive["_item"].map(item2id)
    all_positive_users = set(positive["_user"])
    prior_stage_user = positive["_uid"].map(lambda value: pd.notna(value) and int(value) in train_users)
    prior_stage_item = positive["_iid"].map(lambda value: pd.notna(value) and int(value) in train_item_set)
    warm = positive[prior_stage_user & prior_stage_item].copy()
    latest = warm.drop_duplicates("_user", keep="last").set_index("_user")["_iid"].astype(int).to_dict()
    eligible_users = np.asarray(list(latest), dtype=object)
    if max_users > 0 and len(eligible_users) > max_users:
        selected = rng.choice(len(eligible_users), size=max_users, replace=False)
        chosen_users = eligible_users[selected].tolist()
        sampling = "uniform_without_replacement"
    else:
        chosen_users = eligible_users.tolist()
        sampling = "all_stage_warm_start_users"

    candidates = {}
    for raw_user in tqdm(chosen_users, desc=f"{stage} candidates"):
        user = int(user2id[str(raw_user)])
        target = int(latest[str(raw_user)])
        forbidden = global_interacted.get(user, set()) | {target}
        available = np.asarray([item for item in train_items if int(item) not in forbidden], dtype=int)
        if len(available) < negatives:
            continue
        sampled = rng.choice(available, size=negatives, replace=False).astype(int).tolist()
        candidates[str(user)] = {"target": target, "negatives": sampled}

    prior_users_raw = {
        str(raw_user)
        for raw_user, uid in zip(positive["_user"], positive["_uid"])
        if pd.notna(uid) and int(uid) in train_users
    }
    metadata = {
        "positive_test_users": int(len(all_positive_users)),
        "stage_warm_start_eligible_users": int(len(latest)),
        "excluded_no_prior_stage_interaction": int(len(all_positive_users - prior_users_raw)),
        "excluded_no_stage_warm_target": int(len(prior_users_raw - set(latest))),
        "requested_users": int(len(chosen_users)),
        "evaluated_users": int(len(candidates)),
        "stage_train_users": int(len(train_users)),
        "stage_train_items": int(len(train_items)),
        "stage_train_edges": int(len(stage_train_pairs)),
        "negatives_per_user": int(negatives),
        "user_sampling_method": sampling,
        "stage_seed": int(seed),
        "training_target": f"y_{stage}",
        "target_definition": "latest stage-positive item in Set C observed for the same stage in Set B",
        "negative_definition": "uniform stage-warm item with no observed positive interaction for the user",
        "evaluation_population": "stage-specific warm-start users and items",
    }
    return candidates, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-users-per-stage", type=int, default=0, help="Use <=0 for every stage-specific warm-start user.")
    parser.add_argument("--negatives", type=int, default=99)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_causal_root(__file__)
    logger = setup_logger(root)
    out_dir = ensure_dir(_dcml_paths.workspace_path(root, 'results_of_comparison') / "suning" / "baseline_data")
    train = sort_frame(read_table(input_paths(root, f"DCML_B_{DATA_VERSION}.parquet")))
    test = sort_frame(read_table(input_paths(root, f"DCML_C_{DATA_VERSION}.parquet")))
    certified, certification_manifest = restrict_analysis_splits(
        root,
        "suning",
        {"Set_B": train, "Set_C": test},
        out_dir,
        "suning_prediction_baseline",
        outcomes=[f"y_{stage}" for stage in STAGES],
        balance_columns=["H_level", "M_norm", *[f"y_{stage}" for stage in STAGES]],
        balance_roles={
            "H_level": "moderator",
            "M_norm": "moderator_proxy",
            **{f"y_{stage}": "outcome_descriptive_only" for stage in STAGES},
        },
    )
    train, test = sort_frame(certified["Set_B"]), sort_frame(certified["Set_C"])
    required = {"user_id", "item_id", *[f"y_{stage}" for stage in STAGES]}
    missing = sorted((required - set(train.columns)) | (required - set(test.columns)))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    combined = pd.concat([train, test], ignore_index=True)
    users = list(dict.fromkeys(combined["user_id"].astype(str)))
    items = list(dict.fromkeys(combined["item_id"].astype(str)))
    user2id = {value: index for index, value in enumerate(users)}
    item2id = {value: index for index, value in enumerate(items)}
    user_path = out_dir / f"user2id_{SCRIPT_VERSION}.json"
    item_path = out_dir / f"item2id_{SCRIPT_VERSION}.json"
    write_json(user2id, user_path)
    write_json(item2id, item_path)
    export_interactions(train, user2id, item2id, out_dir, "train")
    export_interactions(test, user2id, item2id, out_dir, "test")

    global_interacted = all_observed_positives(combined, user2id, item2id)
    metadata = {
        "script_version": SCRIPT_VERSION,
        "data_version": DATA_VERSION,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_users": int(len(user2id)),
        "n_items": int(len(item2id)),
        "max_users_per_stage": int(args.max_users_per_stage),
        "negatives_per_user": int(args.negatives),
        "base_seed": int(args.seed),
        "training_split": "Set B",
        "evaluation_split": "Set C",
        "evaluation_protocol": "stage-specific warm-start leave-one-positive-out candidate evaluation",
        "analysis_temporal_status": certification_manifest["analysis_temporal_status"],
        "analysis_population": certification_manifest["analysis_population"],
        "temporal_certification_manifest": certification_manifest["manifest_file"],
        "full_sample_audit_status": certification_manifest["full_sample_audit_status"],
        "user_map_sha256": sha256_file(user_path),
        "item_map_sha256": sha256_file(item_path),
        "stages": {},
    }
    for index, stage in enumerate(STAGES):
        pairs = positive_pairs(train, stage, user2id, item2id)
        candidates, stage_meta = build_stage_candidates(
            test,
            stage,
            user2id,
            item2id,
            pairs,
            global_interacted,
            args.max_users_per_stage,
            args.negatives,
            args.seed + index * 1000,
        )
        candidate_file = out_dir / f"test_candidates_{stage}_{SCRIPT_VERSION}.json"
        write_json(candidates, candidate_file, indent=2)
        stage_meta["candidate_sha256"] = sha256_file(candidate_file)
        metadata["stages"][stage] = stage_meta
        logger.info("%s metadata: %s", stage, stage_meta)

    write_json(metadata, out_dir / f"candidate_metadata_{SCRIPT_VERSION}.json", indent=2)
    pd.DataFrame(metadata["stages"]).T.to_csv(out_dir / f"candidate_stage_summary_{SCRIPT_VERSION}.csv")
    logger.info("Saved independent %s candidate bundle to %s", SCRIPT_VERSION, out_dir)


if __name__ == "__main__":
    main()
