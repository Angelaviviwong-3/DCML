#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare one Amazon dataset for the purchase-only prediction benchmark."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_CORE = Path(__file__).resolve().parent / "A_base"
sys.path.insert(0, str(BASELINE_CORE))

from amazon_baseline_utils import (  # noqa: E402
    BASE_SEED,
    SCRIPT_VERSION,
    VALID_DATASETS,
    active_dataset,
    baseline_data_dir,
    candidate_path,
    ensure_dir,
    find_causal_root,
    id_map_paths,
    load_split_table,
    setup_logger,
    sha256_file,
    write_json,
    write_pair_file,
)

CAUSAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAUSAL_ROOT))

from temporal_certification import load_certification  # noqa: E402


def sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ("timestamp", "user_id", "item_id") if column in frame.columns]
    return frame.sort_values(columns, kind="mergesort").reset_index(drop=True)


def positive_pairs(frame: pd.DataFrame, user2id: dict[str, int], item2id: dict[str, int]) -> np.ndarray:
    positive = frame[pd.to_numeric(frame["y_purchase"], errors="coerce").fillna(0).astype(int).eq(1)]
    pairs = {
        (int(user2id[str(row.user_id)]), int(item2id[str(row.item_id)]))
        for row in positive[["user_id", "item_id"]].itertuples(index=False)
        if str(row.user_id) in user2id and str(row.item_id) in item2id
    }
    return np.asarray(sorted(pairs), dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)


def observed_positive_items(
    frames: list[pd.DataFrame],
    user2id: dict[str, int],
    item2id: dict[str, int],
) -> dict[int, set[int]]:
    output: dict[int, set[int]] = defaultdict(set)
    for frame in frames:
        mask = pd.to_numeric(frame["y_purchase"], errors="coerce").fillna(0).astype(int).eq(1)
        for row in frame.loc[mask, ["user_id", "item_id"]].itertuples(index=False):
            raw_user, raw_item = str(row.user_id), str(row.item_id)
            if raw_user not in user2id or raw_item not in item2id:
                continue
            output[int(user2id[raw_user])].add(int(item2id[raw_item]))
    return output


def build_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    user2id: dict[str, int],
    item2id: dict[str, int],
    negatives: int,
    max_users: int,
    seed: int,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    train_pairs = positive_pairs(train, user2id, item2id)
    train_users = {int(user) for user, _ in train_pairs}
    train_items = np.asarray(sorted({int(item) for _, item in train_pairs}), dtype=np.int64)
    train_item_set = set(train_items.tolist())
    if len(train_items) <= negatives:
        raise RuntimeError(f"Purchase training item pool has only {len(train_items)} items; need > {negatives}")

    positive = sort_frame(
        test[pd.to_numeric(test["y_purchase"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
    )
    positive["_user"] = positive["user_id"].astype(str)
    positive["_item"] = positive["item_id"].astype(str)
    positive["_uid"] = positive["_user"].map(user2id)
    positive["_iid"] = positive["_item"].map(item2id)
    all_positive_users = set(positive["_user"])
    warm_user = positive["_uid"].map(lambda value: pd.notna(value) and int(value) in train_users)
    warm_item = positive["_iid"].map(lambda value: pd.notna(value) and int(value) in train_item_set)
    warm = positive[warm_user & warm_item].copy()
    latest = warm.drop_duplicates("_user", keep="last").set_index("_user")["_iid"].astype(int).to_dict()
    eligible_users = np.asarray(sorted(latest), dtype=object)
    if max_users > 0 and len(eligible_users) > max_users:
        selected = np.sort(rng.choice(len(eligible_users), size=max_users, replace=False))
        chosen_users = eligible_users[selected].tolist()
        user_sampling = "uniform_without_replacement_from_purchase_warm_users"
    else:
        chosen_users = eligible_users.tolist()
        user_sampling = "all_purchase_warm_users"

    interacted = observed_positive_items([train, test], user2id, item2id)
    candidates = {}
    excluded_insufficient_negatives = 0
    for raw_user in chosen_users:
        user = int(user2id[str(raw_user)])
        target = int(latest[str(raw_user)])
        forbidden = interacted.get(user, set()) | {target}
        available = np.asarray([item for item in train_items if int(item) not in forbidden], dtype=np.int64)
        if len(available) < negatives:
            excluded_insufficient_negatives += 1
            continue
        sampled = rng.choice(available, size=negatives, replace=False).astype(int).tolist()
        candidates[str(user)] = {"target": target, "negatives": sampled}

    prior_user_raw = {
        str(raw_user)
        for raw_user, uid in zip(positive["_user"], positive["_uid"])
        if pd.notna(uid) and int(uid) in train_users
    }
    metadata = {
        "stage": "purchase",
        "positive_test_users": int(len(all_positive_users)),
        "purchase_warm_start_eligible_users": int(len(latest)),
        "excluded_no_prior_purchase": int(len(all_positive_users - prior_user_raw)),
        "excluded_nonwarm_target_item": int(len(prior_user_raw - set(latest))),
        "requested_users": int(len(chosen_users)),
        "evaluated_users": int(len(candidates)),
        "excluded_insufficient_negative_pool": int(excluded_insufficient_negatives),
        "purchase_train_users": int(len(train_users)),
        "purchase_train_items": int(len(train_items)),
        "purchase_train_edges": int(len(train_pairs)),
        "negatives_per_user": int(negatives),
        "user_sampling_method": user_sampling,
        "candidate_seed": int(seed),
        "training_target": "y_purchase",
        "target_definition": "latest certified Set C Purchase-positive item that is warm in certified Set B",
        "negative_definition": (
            "uniform certified Set-B Purchase-warm item with no observed Purchase-positive interaction for the user"
        ),
        "evaluation_population": "Purchase-stage warm-start users and items among temporally certified items",
    }
    return candidates, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=VALID_DATASETS, default=os.environ.get("AMAZON_BASELINE_DATASET"))
    parser.add_argument("--negatives", type=int, default=99)
    parser.add_argument("--max-users", type=int, default=10000, help="Use <=0 for every eligible warm-start user")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    args = parser.parse_args()
    if args.dataset not in VALID_DATASETS:
        parser.error("--dataset is required")
    if args.negatives <= 0:
        parser.error("--negatives must be positive")
    return args


def main() -> None:
    args = parse_args()
    os.environ["AMAZON_BASELINE_DATASET"] = args.dataset
    dataset = active_dataset()
    root = find_causal_root(__file__)
    logger = setup_logger(root, "amazon_prediction_baseline_prep")
    output = baseline_data_dir(root)
    set_b = sort_frame(load_split_table(root, "B"))
    set_c = sort_frame(load_split_table(root, "C"))
    required = {"user_id", "item_id", "timestamp", "y_purchase"}
    missing = sorted((required - set(set_b.columns)) | (required - set(set_c.columns)))
    if missing:
        raise ValueError(f"{dataset}: missing required columns: {missing}")

    users = list(dict.fromkeys(set_b["user_id"].astype(str)))
    items = list(dict.fromkeys(set_b["item_id"].astype(str)))
    user2id = {value: index for index, value in enumerate(users)}
    item2id = {value: index for index, value in enumerate(items)}
    user_path, item_path = id_map_paths(root)
    write_json(user2id, user_path)
    write_json(item2id, item_path)

    train_pairs = positive_pairs(set_b, user2id, item2id)
    test_pairs = positive_pairs(set_c, user2id, item2id)
    write_pair_file(output / f"train_purchase_{SCRIPT_VERSION}.txt", train_pairs)
    write_pair_file(output / f"test_purchase_{SCRIPT_VERSION}.txt", test_pairs)
    candidates, stage_metadata = build_candidates(
        set_b,
        set_c,
        user2id,
        item2id,
        args.negatives,
        args.max_users,
        args.seed + (0 if dataset == "amazon_appliances" else 100_000),
    )
    write_json(candidates, candidate_path(root), indent=2)
    stage_metadata.update(
        {
            "stage_train_edges": stage_metadata["purchase_train_edges"],
            "stage_warm_start_eligible_users": stage_metadata["purchase_warm_start_eligible_users"],
            "candidate_file": str(candidate_path(root)),
            "candidate_sha256": sha256_file(candidate_path(root)),
        }
    )

    certified_items, certification, _ = load_certification(root, dataset)
    split_rows = []
    for split, frame in (("Set_B", set_b), ("Set_C", set_c)):
        split_rows.append(
            {
                "Dataset": dataset,
                "Split": split,
                "Certified_Rows": int(len(frame)),
                "Certified_Items": int(frame["item_id"].astype(str).nunique()),
                "Users": int(frame["user_id"].astype(str).nunique()),
                "Purchase_Positive_Rows": int(pd.to_numeric(frame["y_purchase"], errors="coerce").fillna(0).eq(1).sum()),
                "Purchase_Rate": float(pd.to_numeric(frame["y_purchase"], errors="coerce").fillna(0).mean()),
                "Min_Timestamp": float(pd.to_numeric(frame["timestamp"], errors="coerce").min()),
                "Max_Timestamp": float(pd.to_numeric(frame["timestamp"], errors="coerce").max()),
            }
        )
    diagnostics_path = output / f"{dataset}_prediction_split_diagnostics_{SCRIPT_VERSION}.csv"
    pd.DataFrame(split_rows).to_csv(diagnostics_path, index=False)
    stage_summary_path = output / f"{dataset}_candidate_stage_summary_{SCRIPT_VERSION}.csv"
    pd.DataFrame([stage_metadata]).to_csv(stage_summary_path, index=False)
    candidate_file = candidate_path(root)
    manifest = {
        "script_version": SCRIPT_VERSION,
        "data_version": "20260717",
        "dataset": dataset,
        "outcome": "Purchase",
        "amazon_click_available": False,
        "amazon_cart_available": False,
        "training_split": "temporally certified Set B",
        "evaluation_split": "temporally certified Set C",
        "set_c_outcomes_used_for_training_or_model_selection": False,
        "candidate_protocol": "one observed Purchase-positive target plus uniformly sampled Purchase-warm negatives",
        "claim_boundary": "sampled-choice Purchase-stage recommendation only; no full-funnel or OPE claim",
        "certified_item_count": int(len(certified_items)),
        "full_sample_audit_status": certification.get("audit_status"),
        "set_c_item_certification_rate": certification.get("set_c_item_certification_rate"),
        "set_c_row_certification_rate": certification.get("set_c_row_certification_rate"),
        "certification_item_file": certification.get("item_certification_file"),
        "certification_item_sha256": certification.get("item_certification_sha256"),
        "candidate_file": str(candidate_file),
        "candidate_sha256": sha256_file(candidate_file),
        "user_map_file": str(user_path),
        "user_map_sha256": sha256_file(user_path),
        "item_map_file": str(item_path),
        "item_map_sha256": sha256_file(item_path),
        "split_diagnostics_file": str(diagnostics_path),
        "candidate_stage_summary_file": str(stage_summary_path),
        "stage": stage_metadata,
        "stages": {"purchase": stage_metadata},
    }
    write_json(manifest, output / f"{dataset}_prediction_baseline_manifest_{SCRIPT_VERSION}.json", indent=2)
    write_json(manifest, output / f"candidate_metadata_{SCRIPT_VERSION}.json", indent=2)
    logger.info(
        "Prepared %s Purchase-only benchmark | Set B rows=%d Set C rows=%d candidates=%d",
        dataset,
        len(set_b),
        len(set_c),
        len(candidates),
    )


if __name__ == "__main__":
    main()
