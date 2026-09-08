#!/usr/bin/env python3
"""Train the independent Amazon Purchase-only DCML recommender."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for the DCML recommender") from exc

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required to estimate the training-only DML prior") from exc

BASELINE_CORE = Path(__file__).resolve().parent / "A_base"
sys.path.insert(0, str(BASELINE_CORE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
CAUSAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAUSAL_ROOT))

from temporal_certification import filter_certified_frame, load_certification  # noqa: E402
from dcml_utils import extract_category, normalize_ids  # noqa: E402

from amazon_baseline_utils import (  # noqa: E402
    BASE_SEED,
    DATA_VERSION,
    SCRIPT_VERSION,
    STAGES,
    active_dataset,
    candidate_items,
    candidate_path,
    find_causal_root,
    id_map_paths,
    load_candidates,
    load_id_maps,
    performance_dir,
    read_json,
    read_table,
    result_input_paths,
    set_reproducible_seed,
    setup_logger,
    sha256_file,
    split_filename,
    write_json,
)
from dcml_recommender_core_amazon import (  # noqa: E402
    DCMLRecommender,
    bpr_loss,
    funnel_order_loss,
)

OUTCOMES = tuple(f"y_{stage}" for stage in STAGES)
MACRO_COMPONENTS = ("T_con_mkt", "T_con_soc", "T_con_rat", "T_int_fac", "T_int_vis", "T_int_sem")
MACRO_COLUMNS = (
    "T_con_mkt_calib",
    "T_con_soc",
    "T_con_rat",
    "T_int_fac_calib",
    "T_int_vis_calib",
    "T_int_sem",
)
DEFAULT_LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)
METRICS = ("HR@10", "NDCG@10", "HR@20", "NDCG@20")
SUPPORT_MIN_N = 500
SUPPORT_MIN_SD = 0.02
SUPPORT_MIN_IQR = 0.02
SUPPORT_MIN_BINS = 4


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("None of these paths exist: " + "; ".join(map(str, paths)))


def split_path(root: Path, split: str) -> Path:
    filename = split_filename(split)
    return first_existing(result_input_paths(root, filename) + result_input_paths(root, filename.replace(".parquet", ".csv")))


def load_split(root: Path, split: str) -> tuple[pd.DataFrame, Path]:
    path = split_path(root, split)
    frame = read_table([path])
    certified_items, _, _ = load_certification(root, active_dataset())
    filtered = filter_certified_frame(frame, certified_items)
    if filtered.empty:
        raise RuntimeError(f"Set {split} is empty after temporal item certification")
    return filtered, path


def parse_float_grid(text: str) -> tuple[float, ...]:
    values = tuple(sorted({float(part.strip()) for part in text.split(",") if part.strip()}))
    if not values or values[0] != 0.0 or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("the lambda grid must contain 0 and nonnegative values")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--funnel-loss-weight",
        type=float,
        default=0.0,
        help="Compatibility option; must remain zero because Amazon has no native Click/Cart outcomes.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--validation-users", type=int, default=5000)
    parser.add_argument("--validation-negatives", type=int, default=99)
    parser.add_argument("--relevance-retention-floor", type=float, default=0.95)
    parser.add_argument("--causal-lambdas", type=parse_float_grid, default=DEFAULT_LAMBDAS)
    parser.add_argument("--stability-min-sign-agreement", type=float, default=2.0 / 3.0)
    parser.add_argument("--stability-min-correlation", type=float, default=0.50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Audit the saved splits, schema, item categories, temporal validation, and candidates, then stop.",
    )
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be in (0, 0.5)")
    if not 0 < args.relevance_retention_floor <= 1:
        parser.error("--relevance-retention-floor must be in (0, 1]")
    if not 0 <= args.stability_min_sign_agreement <= 1:
        parser.error("--stability-min-sign-agreement must be in [0, 1]")
    if not -1 <= args.stability_min_correlation <= 1:
        parser.error("--stability-min-correlation must be in [-1, 1]")
    if args.funnel_loss_weight != 0:
        parser.error("--funnel-loss-weight must be 0 for Amazon Purchase-only experiments")
    return args


def device_from_arg(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ("timestamp", "user_id", "item_id") if column in frame.columns]
    return frame.sort_values(columns, kind="mergesort").reset_index(drop=True)


def temporal_train_validation(frame: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    tagged = frame.copy()
    tagged["_effect_row_id"] = np.arange(len(tagged), dtype=np.int64)
    ordered = sort_frame(tagged)
    split = max(1, min(len(ordered) - 1, int(math.floor(len(ordered) * (1.0 - validation_fraction)))))
    boundary = ordered.loc[split - 1, "timestamp"]
    while split < len(ordered) and ordered.loc[split, "timestamp"] == boundary:
        split += 1
    if split >= len(ordered):
        raise RuntimeError("Set B cannot be divided into non-overlapping train and validation periods")
    return ordered.iloc[:split].copy(), ordered.iloc[split:].copy()


def map_pairs(frame: pd.DataFrame, user2id: dict[str, int], item2id: dict[str, int], stage: str) -> np.ndarray:
    positive = frame.loc[frame[f"y_{stage}"].fillna(0).astype(int) == 1, ["user_id", "item_id"]]
    pairs = {
        (int(user2id[str(row.user_id)]), int(item2id[str(row.item_id)]))
        for row in positive.itertuples(index=False)
        if str(row.user_id) in user2id and str(row.item_id) in item2id
    }
    return np.asarray(sorted(pairs), dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)


def stage_positive_sets(pairs: np.ndarray) -> dict[int, set[int]]:
    output: dict[int, set[int]] = defaultdict(set)
    for user, item in pairs:
        output[int(user)].add(int(item))
    return output


def legacy_latest_user_levels(frame: pd.DataFrame, user2id: dict[str, int]) -> np.ndarray:
    """Reproduce the superseded last-row state only for the state audit."""
    levels = np.zeros(len(user2id), dtype=np.int64)
    latest = sort_frame(frame).drop_duplicates("user_id", keep="last")
    for row in latest[["user_id", "H_level"]].itertuples(index=False):
        raw = str(row.user_id)
        if raw in user2id:
            value = pd.to_numeric(row.H_level, errors="coerce")
            levels[int(user2id[raw])] = int(np.clip(value if pd.notna(value) else 0, 0, 3))
    return levels


def history_level_cuts(reference: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(reference["user_purchases_pre_log"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    positive = values[values > 0].dropna().to_numpy(float)
    if len(positive) == 0:
        return np.asarray([], dtype=float)
    return np.unique(np.quantile(positive, [1.0 / 3.0, 2.0 / 3.0])).astype(float)


def assign_history_levels(values: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    levels = np.zeros(len(values), dtype=np.int64)
    positive = np.isfinite(values) & (values > 0)
    if not positive.any():
        return levels
    if len(cuts) == 0:
        levels[positive] = 1
    elif len(cuts) == 1:
        levels[positive] = 1 + (values[positive] > cuts[0]).astype(np.int64)
    else:
        levels[positive] = 1 + np.searchsorted(cuts, values[positive], side="right")
    return np.clip(levels, 0, 3)


def event_history_levels(frame: pd.DataFrame, cuts: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(frame["user_purchases_pre_log"], errors="coerce").fillna(0.0).to_numpy(float)
    return assign_history_levels(values, cuts)


def end_of_window_user_levels(
    history: pd.DataFrame,
    user2id: dict[str, int],
    cuts: np.ndarray,
) -> np.ndarray:
    """Return the involvement state after Purchase positives observed through a cutoff."""
    levels = np.zeros(len(user2id), dtype=np.int64)
    positive = history[pd.to_numeric(history["y_purchase"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
    positive["user_id"] = positive["user_id"].astype(str)
    counts = positive.groupby("user_id", sort=False).size()
    values = np.log1p(counts.to_numpy(float))
    assigned = assign_history_levels(values, cuts)
    for raw_user, level in zip(counts.index, assigned):
        if raw_user in user2id:
            levels[int(user2id[raw_user])] = int(level)
    return levels


def user_state_audit_rows(
    context: str,
    candidates_by_stage: dict[str, dict],
    current_levels: np.ndarray,
    legacy_levels: np.ndarray,
    cuts: np.ndarray,
) -> list[dict]:
    rows = []
    for stage, candidates in candidates_by_stage.items():
        users = np.asarray([int(user) for user in candidates], dtype=np.int64)
        current = current_levels[users]
        legacy = legacy_levels[users]
        mismatch = current != legacy
        row = {
            "Context": context,
            "Stage": stage,
            "Candidate_Users": int(len(users)),
            "Legacy_H0_Users": int(np.sum(legacy == 0)),
            "Corrected_H0_Users": int(np.sum(current == 0)),
            "State_Mismatch_Users": int(mismatch.sum()),
            "State_Mismatch_Rate": float(mismatch.mean()) if len(mismatch) else 0.0,
            "H_Cut_1": float(cuts[0]) if len(cuts) else np.nan,
            "H_Cut_2": float(cuts[-1]) if len(cuts) > 1 else np.nan,
            "State_Definition": "log1p(Purchase positives through cutoff), discretized by pre-purchase-history tertiles",
        }
        for level in range(4):
            row[f"Corrected_H{level}_Users"] = int(np.sum(current == level))
        rows.append(row)
    return rows


def item_profiles(
    frame: pd.DataFrame,
    item2id: dict[str, int],
    content_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    ordered = sort_frame(frame).copy()
    ordered["item_id"] = ordered["item_id"].astype(str)
    latest = ordered.drop_duplicates("item_id", keep="last").set_index("item_id")
    medians = {
        column: float(pd.to_numeric(ordered[column], errors="coerce").median())
        if column in ordered and pd.to_numeric(ordered[column], errors="coerce").notna().any()
        else 0.0
        for column in content_columns
    }
    raw = np.zeros((len(item2id), len(content_columns)), dtype=np.float32)
    observed = np.zeros(len(item2id), dtype=bool)
    for raw_item, index in item2id.items():
        if raw_item not in latest.index:
            continue
        row = latest.loc[raw_item]
        raw[int(index)] = [
            float(pd.to_numeric(row.get(column, medians[column]), errors="coerce"))
            if pd.notna(pd.to_numeric(row.get(column, medians[column]), errors="coerce"))
            else medians[column]
            for column in content_columns
        ]
        observed[int(index)] = True
    reference = raw[observed] if observed.any() else raw
    mean = reference.mean(axis=0, keepdims=True)
    scale = reference.std(axis=0, keepdims=True)
    scale[scale <= 1e-8] = 1.0
    standardized = ((raw - mean) / scale).astype(np.float32)
    standardized[~observed] = 0.0
    return standardized, raw, medians


def load_item_categories(root: Path, item2id: dict[str, int]) -> tuple[np.ndarray, dict]:
    dataset = active_dataset()
    path = first_existing(
        [
            _dcml_paths.workspace_path(root, 'processed_data') / dataset / "item_feature&Confounder_price" / "item_text_price_matrix_20260713.parquet",
            _dcml_paths.workspace_path(root, 'processed_data') / dataset / "item_feature&Confounder_price" / "item_text_price_matrix.parquet",
            _dcml_paths.workspace_path(root, 'processed_data') / dataset / "item_feature&Confounder_price" / "item_text_price_matrix_20260713.csv",
            _dcml_paths.workspace_path(root, 'processed_data') / dataset / "item_feature&Confounder_price" / "item_text_price_matrix.csv",
        ]
    )
    source_path = path
    if path.suffix == ".parquet":
        try:
            items = pd.read_parquet(path)
        except Exception as parquet_error:
            csv_path = path.with_suffix(".csv")
            if not csv_path.is_file():
                raise RuntimeError(f"Could not read item-category source {path}: {parquet_error}") from parquet_error
            items = pd.read_csv(csv_path, encoding="utf-8-sig", encoding_errors="replace")
            source_path = csv_path
    else:
        items = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace")
    if "item_id" not in items.columns:
        raise ValueError(f"Item-category source has no item_id column: {source_path}")

    # Reuse the exact canonicalization and category extraction applied by the
    # 20260717 Amazon builder. A separate normalization rule changes ASIN keys.
    items = normalize_ids(items.copy(), ["item_id"])
    canonical_duplicate_rows = int(items["item_id"].duplicated(keep=False).sum())
    category_frame = extract_category(items, out_col="_category")
    category_frame["_category"] = (
        category_frame["_category"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    )
    lookup = dict(zip(category_frame["item_id"].astype(str), category_frame["_category"].astype(str)))
    output = np.asarray(["Unknown"] * len(item2id), dtype=object)
    missing_items = []
    raw_required_items = list(item2id)
    normalized_required = normalize_ids(pd.DataFrame({"item_id": raw_required_items}), ["item_id"])
    for raw_item, normalized in zip(raw_required_items, normalized_required["item_id"].astype(str)):
        index = int(item2id[raw_item])
        if normalized not in lookup:
            missing_items.append(raw_item)
            continue
        output[index] = lookup[normalized]
    audit = {
        "script_version": SCRIPT_VERSION,
        "category_source_file": str(source_path),
        "category_source_sha256": sha256_file(source_path),
        "category_source_rows": int(len(items)),
        "category_source_unique_canonical_items": int(items["item_id"].nunique()),
        "category_source_duplicate_canonical_rows": canonical_duplicate_rows,
        "mapped_items": int(len(item2id) - len(missing_items)),
        "required_items": int(len(item2id)),
        "mapping_rate": float((len(item2id) - len(missing_items)) / max(len(item2id), 1)),
        "missing_items": int(len(missing_items)),
        "missing_item_examples": list(map(str, missing_items[:20])),
        "unknown_category_items": int(np.sum(output == "Unknown")),
        "item_id_normalization": "dcml_utils_20260713.normalize_ids; identical to the Amazon 20260717 builder",
        "category_extraction": "dcml_utils_20260713.extract_category; identical to the Amazon 20260717 builder",
        "required_item_scope": "all items in the Set-B-trained item map; Set-C evaluation is warm-start only",
        "requirement": "every mapped Set B item must resolve through the builder-identical item-category contract",
    }
    return output, audit


def split_contract_summary(frame: pd.DataFrame) -> dict:
    timestamps = pd.to_numeric(frame.get("timestamp"), errors="coerce")
    purchase = pd.to_numeric(frame.get("y_purchase"), errors="coerce").fillna(0).astype(int)
    return {
        "rows": int(len(frame)),
        "unique_users": int(frame["user_id"].astype(str).nunique()),
        "unique_items": int(frame["item_id"].astype(str).nunique()),
        "purchase_positive_rows": int(purchase.eq(1).sum()),
        "purchase_negative_rows": int(purchase.eq(0).sum()),
        "timestamp_min": float(timestamps.min()) if timestamps.notna().any() else None,
        "timestamp_max": float(timestamps.max()) if timestamps.notna().any() else None,
    }


def candidate_contract_summary(candidates: dict, user2id: dict[str, int], item2id: dict[str, int]) -> dict:
    valid_users = {int(value) for value in user2id.values()}
    valid_items = {int(value) for value in item2id.values()}
    invalid_users = 0
    invalid_items = 0
    duplicate_candidate_lists = 0
    candidate_sizes = []
    for raw_user, data in candidates.items():
        user = int(raw_user)
        items = candidate_items(data)
        candidate_sizes.append(len(items))
        invalid_users += int(user not in valid_users)
        invalid_items += sum(int(item not in valid_items) for item in items)
        duplicate_candidate_lists += int(len(items) != len(set(items)))
    return {
        "users": int(len(candidates)),
        "candidate_size_min": int(min(candidate_sizes)) if candidate_sizes else 0,
        "candidate_size_max": int(max(candidate_sizes)) if candidate_sizes else 0,
        "invalid_user_ids": int(invalid_users),
        "invalid_item_ids": int(invalid_items),
        "duplicate_candidate_lists": int(duplicate_candidate_lists),
    }


def semantic_history(
    frame: pd.DataFrame,
    user2id: dict[str, int],
    item2id: dict[str, int],
    item_categories: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[int, str], int]]:
    totals = np.zeros(len(user2id), dtype=np.int64)
    category_purchases: dict[tuple[int, str], int] = defaultdict(int)
    positive = frame.loc[
        pd.to_numeric(frame["y_purchase"], errors="coerce").fillna(0).astype(int).eq(1),
        ["user_id", "item_id"],
    ]
    for row in positive.itertuples(index=False):
        raw_user, raw_item = str(row.user_id), str(row.item_id)
        if raw_user not in user2id or raw_item not in item2id:
            continue
        user, item = int(user2id[raw_user]), int(item2id[raw_item])
        totals[user] += 1
        category_purchases[(user, str(item_categories[item]))] += 1
    return totals, category_purchases


def semantic_alignment(
    user: int,
    item: int,
    item_categories: np.ndarray,
    totals: np.ndarray,
    category_purchases: dict[tuple[int, str], int],
) -> float:
    total = int(totals[user])
    if total <= 0:
        return 0.0
    return float(np.clip(category_purchases.get((user, str(item_categories[item])), 0) / total, 0.0, 1.0))


def fit_training_residuals(
    nuisance_frame: pd.DataFrame,
    effect_frame: pd.DataFrame,
    confounders: list[str],
    seed: int,
) -> dict[str, np.ndarray]:
    x_train = nuisance_frame.reindex(columns=confounders).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    x_effect = effect_frame.reindex(columns=confounders).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    residuals: dict[str, np.ndarray] = {}
    for index, (component, column) in enumerate(zip(MACRO_COMPONENTS, MACRO_COLUMNS)):
        model = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=seed + index,
        )
        target = pd.to_numeric(nuisance_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        train_observed = target.notna().to_numpy()
        if int(train_observed.sum()) < 500:
            raise RuntimeError(f"Set A has insufficient observed values for {column}: {int(train_observed.sum())}")
        model.fit(x_train[train_observed], target.loc[train_observed].to_numpy(float))
        observed = pd.to_numeric(effect_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        effect_observed = observed.notna().to_numpy()
        values = np.full(len(effect_frame), np.nan, dtype=float)
        values[effect_observed] = (
            observed.loc[effect_observed].to_numpy(float) - model.predict(x_effect[effect_observed])
        )
        residuals[f"T::{component}"] = values
    for index, stage in enumerate(STAGES):
        model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=seed + 100 + index,
        )
        target = pd.to_numeric(nuisance_frame[f"y_{stage}"], errors="coerce").fillna(0).astype(int).to_numpy()
        if len(np.unique(target)) < 2:
            raise RuntimeError(f"Set A has only one class for y_{stage}")
        model.fit(x_train, target)
        observed = pd.to_numeric(effect_frame[f"y_{stage}"], errors="coerce").fillna(0).to_numpy(float)
        residuals[f"Y::{stage}"] = observed - model.predict_proba(x_effect)[:, 1]
    return residuals


def density_stats(values: np.ndarray, bins: int = 10) -> tuple[int, int, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0, 0, np.nan
    counts, _ = np.histogram(finite, bins=bins)
    nonzero = counts[counts > 0]
    return (
        int(len(nonzero)),
        int(nonzero.min()) if len(nonzero) else 0,
        float(counts.max() / len(finite)),
    )


def support_diagnostics(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {
            "Estimable": False,
            "Support_Flag": "empty_treatment_residual",
            "Support_N": 0,
            "Residual_SD": np.nan,
            "Residual_IQR": np.nan,
            "Residual_Support_1_99": np.nan,
            "Effective_Bins_10": 0,
            "Min_Bin_Count_10": 0,
            "Max_Bin_Share_10": np.nan,
        }
    sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    iqr = float(np.percentile(finite, 75) - np.percentile(finite, 25))
    support = float(np.percentile(finite, 99) - np.percentile(finite, 1))
    effective_bins, min_bin, max_share = density_stats(finite)
    reasons = []
    if len(finite) < SUPPORT_MIN_N:
        reasons.append("small_n")
    if sd < SUPPORT_MIN_SD:
        reasons.append("low_residual_sd")
    if iqr < SUPPORT_MIN_IQR:
        reasons.append("low_residual_iqr")
    if effective_bins < SUPPORT_MIN_BINS:
        reasons.append("sparse_effective_bins")
    return {
        "Estimable": not reasons,
        "Support_Flag": "ok" if not reasons else "|".join(reasons),
        "Support_N": int(len(finite)),
        "Residual_SD": sd,
        "Residual_IQR": iqr,
        "Residual_Support_1_99": support,
        "Effective_Bins_10": effective_bins,
        "Min_Bin_Count_10": min_bin,
        "Max_Bin_Share_10": max_share,
    }


def original_basis_coefficients(pure_coefficients: np.ndarray, projection_slopes: np.ndarray) -> np.ndarray:
    """Convert [conformity, pure-interest] coefficients to the original treatment basis."""
    conformity = np.asarray(pure_coefficients[:3], dtype=float)
    interest = np.asarray(pure_coefficients[3:], dtype=float)
    return np.concatenate([conformity - projection_slopes @ interest, interest])


def estimate_causal_prior(
    residuals: dict[str, np.ndarray],
    mask: np.ndarray,
    h_values_all: np.ndarray,
    label: str,
    include_cate: bool = True,
) -> tuple[dict, list[dict], list[dict]]:
    requested = np.asarray(mask, dtype=bool)
    treatment_all = np.column_stack([residuals[f"T::{component}"] for component in MACRO_COMPONENTS])
    complete = requested & np.isfinite(treatment_all).all(axis=1)
    if int(complete.sum()) < SUPPORT_MIN_N:
        raise RuntimeError(
            f"{label}: only {int(complete.sum())} rows have complete six-variable treatment history; "
            f"at least {SUPPORT_MIN_N} are required"
        )
    conformity = treatment_all[complete, :3]
    interest = treatment_all[complete, 3:]
    projection_design = np.column_stack([np.ones(int(complete.sum())), conformity])
    pure_interest = np.empty_like(interest)
    projection_parameters = np.empty((projection_design.shape[1], interest.shape[1]), dtype=float)
    for column in range(interest.shape[1]):
        coefficient, *_ = np.linalg.lstsq(projection_design, interest[:, column], rcond=None)
        projection_parameters[:, column] = coefficient
        pure_interest[:, column] = interest[:, column] - projection_design @ coefficient
    treatment = np.column_stack([conformity, pure_interest])
    projection_slopes = projection_parameters[1:, :]
    h_values = np.asarray(h_values_all, dtype=int)[complete]
    prior: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    rows = []
    projection_rows = []
    for interest_index, interest_component in enumerate(MACRO_COMPONENTS[3:]):
        projection_rows.append(
            {
                "Prior_Sample": label,
                "Interest_Component": interest_component,
                "Projection_Term": "Intercept",
                "Projection_Coefficient": float(projection_parameters[0, interest_index]),
            }
        )
        for conformity_index, conformity_component in enumerate(MACRO_COMPONENTS[:3]):
            projection_rows.append(
                {
                    "Prior_Sample": label,
                    "Interest_Component": interest_component,
                    "Projection_Term": conformity_component,
                    "Projection_Coefficient": float(projection_slopes[conformity_index, interest_index]),
                }
            )
    groups = [("ATE", np.ones(int(complete.sum()), dtype=bool))]
    if include_cate:
        groups.extend((str(level), h_values == level) for level in range(4))
    for stage in STAGES:
        y = residuals[f"Y::{stage}"][complete]
        ate_scoring = None
        for h_level, group in groups:
            if int(group.sum()) < 500:
                continue
            diagnostics = [support_diagnostics(treatment[group, index]) for index in range(treatment.shape[1])]
            all_supported = all(item["Estimable"] for item in diagnostics)
            fallback_to_ate = h_level != "ATE" and not all_supported
            pure_values = np.full(len(MACRO_COMPONENTS), np.nan, dtype=float)
            if fallback_to_ate:
                if ate_scoring is None:
                    raise RuntimeError("ATE must be estimated before support-aware CATE fallback")
                scoring_values = ate_scoring.copy()
            else:
                supported = np.asarray([item["Estimable"] for item in diagnostics], dtype=bool)
                if not supported.any():
                    scoring_values = np.zeros(len(MACRO_COMPONENTS), dtype=float)
                else:
                    design = np.column_stack([np.ones(group.sum()), treatment[group][:, supported]])
                    coefficient, *_ = np.linalg.lstsq(design, y[group], rcond=None)
                    pure_values[supported] = coefficient[1:]
                    pure_for_conversion = np.nan_to_num(pure_values, nan=0.0)
                    scoring_values = original_basis_coefficients(pure_for_conversion, projection_slopes)
                if h_level == "ATE":
                    ate_scoring = scoring_values.copy()
            values = {component: float(value) for component, value in zip(MACRO_COMPONENTS, scoring_values)}
            prior[stage][h_level] = values
            for index, (component, value) in enumerate(values.items()):
                diagnostic = diagnostics[index]
                rows.append(
                    {
                        "Prior_Sample": label,
                        "Stage": stage,
                        "H_level": h_level,
                        "Component": component,
                        "Coefficient": value,
                        "Pure_Basis_Coefficient": (
                            float(pure_values[index]) if np.isfinite(pure_values[index]) else np.nan
                        ),
                        "N": int(group.sum()),
                        **diagnostic,
                        "Fallback_to_ATE": fallback_to_ate,
                        "Fallback_Rule": "whole H group falls back to ATE when any component lacks residual support",
                        "Estimation_Basis": "conformity residuals + Gram-Schmidt pure-interest residuals",
                        "Scoring_Basis": "algebraically equivalent coefficients on original treatment values",
                        "Nuisance_Split": "Set A",
                        "Effect_Split": "Set B only",
                        "Use": "recommendation training prior; not a reported causal estimand",
                    }
                )
    prior["__scoring_basis__"] = "original_treatment_equivalent"
    prior["__projection_slopes__"] = projection_slopes.tolist()
    return dict(prior), rows, projection_rows


def temporal_half_masks(frame: pd.DataFrame, base_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(np.asarray(base_mask, dtype=bool))
    selected = frame.iloc[indices].copy()
    selected["_original_index"] = indices
    selected = sort_frame(selected)
    split = max(1, min(len(selected) - 1, len(selected) // 2))
    boundary = selected.iloc[split - 1]["timestamp"]
    while split < len(selected) and selected.iloc[split]["timestamp"] == boundary:
        split += 1
    if split >= len(selected):
        raise RuntimeError("Effect sample cannot be split into non-overlapping early/late periods")
    early = np.zeros(len(frame), dtype=bool)
    late = np.zeros(len(frame), dtype=bool)
    early[selected.iloc[:split]["_original_index"].to_numpy(int)] = True
    late[selected.iloc[split:]["_original_index"].to_numpy(int)] = True
    return early, late


def prior_stability_audit(
    residuals: dict[str, np.ndarray],
    effect_frame: pd.DataFrame,
    base_mask: np.ndarray,
    h_values: np.ndarray,
    label: str,
    min_sign_agreement: float,
    min_correlation: float,
) -> tuple[dict[str, bool], list[dict]]:
    early_mask, late_mask = temporal_half_masks(effect_frame, base_mask)
    early_prior, early_rows, _ = estimate_causal_prior(
        residuals, early_mask, h_values, f"{label} early half", include_cate=False
    )
    late_prior, late_rows, _ = estimate_causal_prior(
        residuals, late_mask, h_values, f"{label} late half", include_cate=False
    )
    early_support = pd.DataFrame(early_rows)
    late_support = pd.DataFrame(late_rows)
    gates, rows = {}, []
    for stage in STAGES:
        early = np.asarray([early_prior[stage]["ATE"][component] for component in MACRO_COMPONENTS], dtype=float)
        late = np.asarray([late_prior[stage]["ATE"][component] for component in MACRO_COMPONENTS], dtype=float)
        sign_match = np.sign(early) == np.sign(late)
        sign_agreement = float(sign_match.mean())
        correlation = float(np.corrcoef(early, late)[0, 1]) if np.std(early) > 0 and np.std(late) > 0 else np.nan
        early_ok = bool(
            early_support[(early_support["Stage"] == stage) & (early_support["H_level"] == "ATE")]["Estimable"].all()
        )
        late_ok = bool(
            late_support[(late_support["Stage"] == stage) & (late_support["H_level"] == "ATE")]["Estimable"].all()
        )
        passed = bool(
            early_ok
            and late_ok
            and sign_agreement + 1e-12 >= min_sign_agreement
            and np.isfinite(correlation)
            and correlation + 1e-12 >= min_correlation
        )
        gates[stage] = passed
        for index, component in enumerate(MACRO_COMPONENTS):
            rows.append(
                {
                    "Prior_Sample": label,
                    "Stage": stage,
                    "Component": component,
                    "Early_Coefficient": float(early[index]),
                    "Late_Coefficient": float(late[index]),
                    "Sign_Match": bool(sign_match[index]),
                    "Stage_Sign_Agreement": sign_agreement,
                    "Stage_Coefficient_Correlation": correlation,
                    "Early_ATE_Support_Passed": early_ok,
                    "Late_ATE_Support_Passed": late_ok,
                    "Min_Sign_Agreement": min_sign_agreement,
                    "Min_Correlation": min_correlation,
                    "Stability_Gate_Passed": passed,
                    "Gate_Use": "pre-C recommendation prior gate; Set C outcomes and estimates are not used",
                }
            )
    return gates, rows


def build_validation_candidates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    user2id: dict[str, int],
    item2id: dict[str, int],
    stage: str,
    negatives: int,
    max_users: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    train_pairs = map_pairs(train, user2id, item2id, stage)
    train_users = {int(user) for user, _ in train_pairs}
    train_items = np.asarray(sorted({int(item) for _, item in train_pairs}), dtype=int)
    train_item_set = set(train_items.tolist())
    positive = sort_frame(validation.loc[validation[f"y_{stage}"].fillna(0).astype(int) == 1].copy())
    positive["_uid"] = positive["user_id"].astype(str).map(user2id)
    positive["_iid"] = positive["item_id"].astype(str).map(item2id)
    warm = positive[
        positive["_uid"].map(lambda value: pd.notna(value) and int(value) in train_users)
        & positive["_iid"].map(lambda value: pd.notna(value) and int(value) in train_item_set)
    ]
    latest = warm.drop_duplicates("_uid", keep="last").set_index("_uid")["_iid"].astype(int).to_dict()
    users = np.asarray(sorted(latest), dtype=int)
    if max_users > 0 and len(users) > max_users:
        users = np.sort(rng.choice(users, size=max_users, replace=False))
    observed: dict[int, set[int]] = defaultdict(set)
    all_frame = pd.concat([train, validation], ignore_index=True)
    mask = all_frame[list(OUTCOMES)].fillna(0).astype(int).sum(axis=1) > 0
    for row in all_frame.loc[mask, ["user_id", "item_id"]].itertuples(index=False):
        raw_user, raw_item = str(row.user_id), str(row.item_id)
        if raw_user in user2id and raw_item in item2id:
            observed[int(user2id[raw_user])].add(int(item2id[raw_item]))
    output = {}
    for user in users:
        target = int(latest[int(user)])
        available = np.asarray([item for item in train_items if item not in observed[user] and item != target], dtype=int)
        if len(available) < negatives:
            continue
        sampled = rng.choice(available, size=negatives, replace=False).astype(int).tolist()
        output[str(int(user))] = {"target": target, "negatives": sampled}
    return output


def sample_negative_items(
    users: np.ndarray,
    item_pool: np.ndarray,
    positive_sets: dict[int, set[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = rng.choice(item_pool, size=len(users), replace=True).astype(np.int64)
    invalid = np.asarray([int(item) in positive_sets[int(user)] for user, item in zip(users, negatives)], dtype=bool)
    attempts = 0
    while invalid.any():
        negatives[invalid] = rng.choice(item_pool, size=int(invalid.sum()), replace=True)
        invalid = np.asarray([int(item) in positive_sets[int(user)] for user, item in zip(users, negatives)], dtype=bool)
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Unable to sample stage-specific negative items")
    return negatives


def train_model(
    frame: pd.DataFrame,
    user2id: dict[str, int],
    item2id: dict[str, int],
    content_matrix: np.ndarray,
    h_levels: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[DCMLRecommender, list[dict]]:
    set_reproducible_seed(seed)
    rng = np.random.default_rng(seed)
    model = DCMLRecommender(len(user2id), len(item2id), content_matrix.shape[1], args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    content_tensor = torch.as_tensor(content_matrix, dtype=torch.float32, device=device)
    level_tensor = torch.as_tensor(h_levels, dtype=torch.long, device=device)
    pairs = {stage: map_pairs(frame, user2id, item2id, stage) for stage in STAGES}
    positives = {stage: stage_positive_sets(pairs[stage]) for stage in STAGES}
    pools = {stage: np.asarray(sorted({int(item) for _, item in pairs[stage]}), dtype=np.int64) for stage in STAGES}
    if any(len(pairs[stage]) == 0 or len(pools[stage]) < 2 for stage in STAGES):
        raise RuntimeError("Every stage requires positive training pairs and at least two items")
    steps = max(int(math.ceil(len(pairs[stage]) / args.batch_size)) for stage in STAGES)
    diagnostics = []
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        order_losses = []
        for _ in range(steps):
            for stage_index in rng.permutation(len(STAGES)):
                stage = STAGES[int(stage_index)]
                selected = rng.integers(0, len(pairs[stage]), size=args.batch_size)
                batch = pairs[stage][selected]
                users, positive_items = batch[:, 0], batch[:, 1]
                negative_items = sample_negative_items(users, pools[stage], positives[stage], rng)
                user_t = torch.as_tensor(users, dtype=torch.long, device=device)
                positive_t = torch.as_tensor(positive_items, dtype=torch.long, device=device)
                negative_t = torch.as_tensor(negative_items, dtype=torch.long, device=device)
                stage_t = torch.full((len(users),), int(stage_index), dtype=torch.long, device=device)
                h_t = level_tensor[user_t]
                positive_score = model(user_t, positive_t, stage_t, h_t, content_tensor[positive_t])
                negative_score = model(user_t, negative_t, stage_t, h_t, content_tensor[negative_t])
                loss = bpr_loss(positive_score, negative_score)
                if args.funnel_loss_weight > 0:
                    all_stage_scores = [
                        model(
                            user_t,
                            positive_t,
                            torch.full_like(stage_t, index),
                            h_t,
                            content_tensor[positive_t],
                        )
                        for index in range(len(STAGES))
                    ]
                    order_loss = funnel_order_loss(all_stage_scores)
                    loss = loss + args.funnel_loss_weight * order_loss
                    order_losses.append(float(order_loss.detach().cpu()))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
        diagnostics.append(
            {
                "Epoch": epoch + 1,
                "Loss": float(np.mean(epoch_losses)),
                "Funnel_Order_Loss": float(np.mean(order_losses)) if order_losses else 0.0,
                "Steps_Per_Stage": steps,
            }
        )
    return model, diagnostics


def candidate_causal_scores(
    user: int,
    items: list[int],
    stage: str,
    h_level: int,
    prior: dict,
    macro_item_values: np.ndarray,
    item_categories: np.ndarray,
    semantic_totals: np.ndarray,
    semantic_category_purchases: dict[tuple[int, str], int],
    content_columns: list[str],
) -> np.ndarray:
    if prior.get("__scoring_basis__") != "original_treatment_equivalent":
        raise RuntimeError("Recommendation prior is not expressed on the candidate treatment basis")
    coefficient = prior.get(stage, {}).get(str(h_level)) or prior.get(stage, {}).get("ATE", {})
    index = {column: position for position, column in enumerate(content_columns)}
    values = []
    for item in items:
        treatment = {
            "T_con_mkt": macro_item_values[item, index["T_con_mkt_calib"]],
            "T_con_soc": macro_item_values[item, index["T_con_soc"]],
            "T_con_rat": macro_item_values[item, index["T_con_rat"]],
            "T_int_fac": macro_item_values[item, index["T_int_fac_calib"]],
            "T_int_vis": macro_item_values[item, index["T_int_vis_calib"]],
            "T_int_sem": semantic_alignment(user, item, item_categories, semantic_totals, semantic_category_purchases),
        }
        values.append(sum(float(coefficient.get(component, 0.0)) * float(treatment[component]) for component in MACRO_COMPONENTS))
    return np.asarray(values, dtype=float)


def expected_metrics(scores: np.ndarray, target_index: int) -> tuple[dict[str, float], int]:
    target = float(scores[target_index])
    tied = np.isclose(scores, target, rtol=1e-12, atol=1e-12)
    better = int(np.sum((scores > target) & ~tied))
    tie_size = int(tied.sum())
    positions = np.arange(better + 1, better + tie_size + 1, dtype=float)
    output = {}
    for k in (10, 20):
        inside = positions <= k
        output[f"HR@{k}"] = float(np.mean(inside))
        output[f"NDCG@{k}"] = float(np.mean(np.where(inside, 1.0 / np.log2(positions + 1.0), 0.0)))
    return output, tie_size


def discounted_causal_score(scores: np.ndarray, causal_scores: np.ndarray, k: int = 10) -> float:
    order = np.lexsort((np.arange(len(scores)), -scores))[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2, dtype=float))
    return float(np.dot(causal_scores[order], discounts) / discounts.sum())


def raw_candidate_scores(
    model: DCMLRecommender,
    user: int,
    items: list[int],
    stage: str,
    content_matrix: np.ndarray,
    macro_item_values: np.ndarray,
    h_levels: np.ndarray,
    prior: dict,
    item_categories: np.ndarray,
    semantic_totals: np.ndarray,
    semantic_category_purchases: dict[tuple[int, str], int],
    content_columns: list[str],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    stage_index = STAGES.index(stage)
    users_t = torch.full((len(items),), user, dtype=torch.long, device=device)
    items_t = torch.as_tensor(items, dtype=torch.long, device=device)
    stages_t = torch.full((len(items),), stage_index, dtype=torch.long, device=device)
    h_level = int(h_levels[user])
    h_t = torch.full((len(items),), h_level, dtype=torch.long, device=device)
    content_t = torch.as_tensor(content_matrix[items], dtype=torch.float32, device=device)
    relevance = model(users_t, items_t, stages_t, h_t, content_t).detach().cpu().numpy().astype(float)
    causal = candidate_causal_scores(
        user,
        items,
        stage,
        h_level,
        prior,
        macro_item_values,
        item_categories,
        semantic_totals,
        semantic_category_purchases,
        content_columns,
    )
    return relevance, causal


def estimate_score_calibration(
    model: DCMLRecommender,
    candidates: dict,
    stage: str,
    content_matrix: np.ndarray,
    macro_item_values: np.ndarray,
    h_levels: np.ndarray,
    prior: dict,
    item_categories: np.ndarray,
    semantic_totals: np.ndarray,
    semantic_category_purchases: dict[tuple[int, str], int],
    content_columns: list[str],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    relevance_values, causal_values = [], []
    with torch.no_grad():
        for user_key, candidate in candidates.items():
            user = int(user_key)
            relevance, causal = raw_candidate_scores(
                model,
                user,
                candidate_items(candidate),
                stage,
                content_matrix,
                macro_item_values,
                h_levels,
                prior,
                item_categories,
                semantic_totals,
                semantic_category_purchases,
                content_columns,
                device,
            )
            relevance_values.append(relevance)
            causal_values.append(causal)
    relevance_all = np.concatenate(relevance_values)
    causal_all = np.concatenate(causal_values)
    relevance_sd = float(relevance_all.std())
    causal_sd = float(causal_all.std())
    return {
        "relevance_mean": float(relevance_all.mean()),
        "relevance_sd": relevance_sd if relevance_sd > 1e-12 else 1.0,
        "causal_mean": float(causal_all.mean()),
        "causal_sd": causal_sd if causal_sd > 1e-12 else 1.0,
        "calibration_pairs": int(len(relevance_all)),
        "calibration_users": int(len(candidates)),
    }


def apply_score_calibration(values: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return (np.asarray(values, dtype=float) - float(mean)) / float(sd)


def score_candidates(
    model: DCMLRecommender,
    candidates: dict,
    stage: str,
    content_matrix: np.ndarray,
    macro_item_values: np.ndarray,
    h_levels: np.ndarray,
    prior: dict,
    item_categories: np.ndarray,
    semantic_totals: np.ndarray,
    semantic_category_purchases: dict[tuple[int, str], int],
    content_columns: list[str],
    calibration: dict[str, float],
    causal_lambda: float,
    device: torch.device,
    evaluate_targets: bool = True,
) -> tuple[dict, dict[str, list[float]], list[float], list[int]]:
    model.eval()
    predictions = {}
    metric_values: dict[str, list[float]] = defaultdict(list)
    causal_values, tie_sizes = [], []
    with torch.no_grad():
        for user_key, candidate in candidates.items():
            user = int(user_key)
            items = candidate_items(candidate)
            relevance, causal = raw_candidate_scores(
                model,
                user,
                items,
                stage,
                content_matrix,
                macro_item_values,
                h_levels,
                prior,
                item_categories,
                semantic_totals,
                semantic_category_purchases,
                content_columns,
                device,
            )
            relevance_scaled = apply_score_calibration(
                relevance,
                calibration["relevance_mean"],
                calibration["relevance_sd"],
            )
            causal_scaled = apply_score_calibration(
                causal,
                calibration["causal_mean"],
                calibration["causal_sd"],
            )
            final_scores = relevance_scaled + causal_lambda * causal_scaled
            if evaluate_targets:
                metrics, tie_size = expected_metrics(final_scores, len(items) - 1)
                for metric, value in metrics.items():
                    metric_values[metric].append(value)
                tie_sizes.append(tie_size)
            causal_values.append(discounted_causal_score(final_scores, causal, 10))
            predictions[user_key] = {str(item): float(score) for item, score in zip(items, final_scores)}
    return predictions, metric_values, causal_values, tie_sizes


def select_causal_lambdas(
    model: DCMLRecommender,
    validation_candidates: dict[str, dict],
    content_matrix: np.ndarray,
    macro_item_values: np.ndarray,
    h_levels: np.ndarray,
    prior: dict,
    item_categories: np.ndarray,
    semantic_totals: np.ndarray,
    semantic_category_purchases: dict[tuple[int, str], int],
    content_columns: list[str],
    calibrations: dict[str, dict[str, float]],
    stability_gates: dict[str, bool],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], list[dict]]:
    rows, selected = [], {}
    for stage in STAGES:
        candidates = validation_candidates[stage]
        stage_rows = []
        ndcg_by_lambda: dict[float, np.ndarray] = {}
        for causal_lambda in args.causal_lambdas:
            _, metric_values, causal_values, _ = score_candidates(
                model,
                candidates,
                stage,
                content_matrix,
                macro_item_values,
                h_levels,
                prior,
                item_categories,
                semantic_totals,
                semantic_category_purchases,
                content_columns,
                calibrations[stage],
                causal_lambda,
                device,
            )
            row = {
                "Stage": stage,
                "Causal_Lambda": causal_lambda,
                "Validation_Users": len(candidates),
                "Mean_Estimated_Causal_Score@10": float(np.mean(causal_values)),
                "Causal_Score_Interpretation": "model-implied utility; not realized policy uplift or OPE",
                **{f"Calibration_{key}": value for key, value in calibrations[stage].items()},
            }
            for metric in METRICS:
                row[metric] = float(np.mean(metric_values[metric]))
            ndcg_by_lambda[float(causal_lambda)] = np.asarray(metric_values["NDCG@10"], dtype=float)
            stage_rows.append(row)
        base = next(row for row in stage_rows if row["Causal_Lambda"] == 0.0)
        base_values = ndcg_by_lambda[0.0]
        allowed_degradation = (1.0 - args.relevance_retention_floor) * float(base["NDCG@10"])
        eligible = []
        stability_passed = bool(stability_gates.get(stage, False))
        for row in stage_rows:
            difference = ndcg_by_lambda[float(row["Causal_Lambda"])] - base_values
            mean_difference = float(difference.mean())
            paired_se = float(difference.std(ddof=1) / math.sqrt(len(difference))) if len(difference) > 1 else 0.0
            lower_bound = mean_difference - 1.645 * paired_se
            mean_retention = row["NDCG@10"] + 1e-12 >= args.relevance_retention_floor * base["NDCG@10"]
            paired_noninferior = lower_bound + 1e-12 >= -allowed_degradation
            row["NDCG@10_Difference_vs_Lambda0"] = mean_difference
            row["NDCG@10_Difference_Paired_SE"] = paired_se
            row["NDCG@10_Difference_OneSided95_Lower"] = lower_bound
            row["Allowed_NDCG@10_Degradation"] = allowed_degradation
            row["Meets_Mean_Retention_Floor"] = mean_retention
            row["Passes_Paired_Noninferiority"] = paired_noninferior
            row["Prior_Stability_Gate_Passed"] = stability_passed
            row["Eligible_After_Stability_Gate"] = bool(
                mean_retention and paired_noninferior and (stability_passed or row["Causal_Lambda"] == 0.0)
            )
            if row["Eligible_After_Stability_Gate"]:
                eligible.append(row)
        choice = sorted(
            eligible,
            key=lambda row: (-row["Mean_Estimated_Causal_Score@10"], -row["NDCG@10"], row["Causal_Lambda"]),
        )[0]
        selected[stage] = float(choice["Causal_Lambda"])
        for row in stage_rows:
            row["Baseline_NDCG@10"] = base["NDCG@10"]
            row["Retention_Floor"] = args.relevance_retention_floor
            row["Meets_Retention_Floor"] = row in eligible
            row["Selected"] = row is choice
            row["Selection_Rule"] = (
                "maximize validation model-implied causal score subject to B-only prior-stability gating, "
                "mean retention, and paired one-sided NDCG@10 noninferiority"
            )
            rows.append(row)
    return selected, rows


def main() -> None:
    args = parse_args()
    root = find_causal_root(__file__)
    dataset = active_dataset()
    logger = setup_logger(root, "independent_dcml_recommender_training")
    output = performance_dir(root)
    _, certification_summary, _ = load_certification(root, dataset)
    device = device_from_arg(args.device)
    seed_offset = int(os.environ.get("AMAZON_TRAINING_SEED_OFFSET", "0"))
    seed = BASE_SEED + seed_offset + 901
    set_reproducible_seed(seed)

    set_a, set_a_path = load_split(root, "A")
    set_b, set_b_path = load_split(root, "B")
    set_a, set_b = sort_frame(set_a), sort_frame(set_b)
    user2id, item2id = load_id_maps(root)
    user_map_path, item_map_path = id_map_paths(root)
    schema = read_json(
        first_existing(
            result_input_paths(
                root,
                f"{dataset}_DML_feature_schema_20260717.json",
                version="20260717",
            )
        )
    )
    treatments = list(schema.get("treatments", []))
    confounders = list(schema.get("confounder_whitelist", schema.get("confounders", [])))
    content_columns = [column for column in treatments if column != "T_int_sem"]
    required_columns = {"user_id", "item_id", "timestamp", "y_purchase", *treatments, *confounders}
    missing_a = sorted(required_columns - set(set_a.columns))
    missing_b = sorted(required_columns - set(set_b.columns))
    missing_macro = sorted((set(MACRO_COLUMNS) - set(set_a.columns)) | (set(MACRO_COLUMNS) - set(set_b.columns)))

    item_categories, category_audit = load_item_categories(root, item2id)
    write_json(category_audit, output / f"DCML_Item_Category_Audit_{SCRIPT_VERSION}.json", indent=2)

    b_train, b_validation = temporal_train_validation(set_b, args.validation_fraction)
    train_row_ids = set(b_train["_effect_row_id"].astype(int))
    train_mask = np.asarray([index in train_row_ids for index in range(len(set_b))], dtype=bool)
    b_train = b_train.drop(columns="_effect_row_id")
    b_validation = b_validation.drop(columns="_effect_row_id")
    validation_candidates = {}
    for index, stage in enumerate(STAGES):
        validation_candidates[stage] = build_validation_candidates(
            b_train,
            b_validation,
            user2id,
            item2id,
            stage,
            args.validation_negatives,
            args.validation_users,
            seed + index * 1000,
        )
        write_json(
            validation_candidates[stage],
            output / f"DCML_validation_candidates_{stage}_{SCRIPT_VERSION}.json",
            indent=2,
        )
    final_candidates = {stage: load_candidates(root, stage) for stage in STAGES}

    preflight_failures = []
    preflight_warnings = []
    if missing_a:
        preflight_failures.append(f"Set A missing required columns: {missing_a}")
    if missing_b:
        preflight_failures.append(f"Set B missing required columns: {missing_b}")
    if missing_macro:
        preflight_failures.append(f"Aggregate DCML specification missing columns: {missing_macro}")
    if category_audit["missing_items"]:
        preflight_failures.append(
            f"{category_audit['missing_items']} Set-B item IDs have no builder-consistent category mapping"
        )
    if category_audit["unknown_category_items"]:
        preflight_warnings.append(
            f"{category_audit['unknown_category_items']} Set-B items resolve only to Unknown category"
        )
    if category_audit["category_source_duplicate_canonical_rows"]:
        preflight_warnings.append(
            f"{category_audit['category_source_duplicate_canonical_rows']} category-source rows share canonical item IDs; "
            "the builder-identical first-record rule is applied"
        )
    split_summaries = {
        "set_a": split_contract_summary(set_a),
        "set_b": split_contract_summary(set_b),
        "set_b_temporal_train": split_contract_summary(b_train),
        "set_b_temporal_validation": split_contract_summary(b_validation),
    }
    for split_name, summary in split_summaries.items():
        if summary["purchase_positive_rows"] == 0 or summary["purchase_negative_rows"] == 0:
            preflight_failures.append(f"{split_name} does not contain both Purchase outcome classes")
    train_timestamp_max = split_summaries["set_b_temporal_train"]["timestamp_max"]
    validation_timestamp_min = split_summaries["set_b_temporal_validation"]["timestamp_min"]
    strict_temporal_validation = bool(
        train_timestamp_max is not None
        and validation_timestamp_min is not None
        and train_timestamp_max < validation_timestamp_min
    )
    if not strict_temporal_validation:
        preflight_failures.append("Set B train/validation timestamps are not strictly separated")
    validation_candidate_summaries = {
        stage: candidate_contract_summary(candidates, user2id, item2id)
        for stage, candidates in validation_candidates.items()
    }
    final_candidate_summaries = {
        stage: candidate_contract_summary(candidates, user2id, item2id)
        for stage, candidates in final_candidates.items()
    }
    for scope, summaries in (
        ("Set-B validation", validation_candidate_summaries),
        ("Set-C evaluation", final_candidate_summaries),
    ):
        for stage, summary in summaries.items():
            if summary["users"] == 0:
                preflight_failures.append(f"{scope} has no warm-start candidates for {stage}")
            if summary["invalid_user_ids"] or summary["invalid_item_ids"] or summary["duplicate_candidate_lists"]:
                preflight_failures.append(f"{scope} candidate contract failed for {stage}: {summary}")

    preflight = {
        "script_version": SCRIPT_VERSION,
        "data_version": DATA_VERSION,
        "dataset": dataset,
        "status": "PASS" if not preflight_failures else "FAIL",
        "failures": preflight_failures,
        "warnings": preflight_warnings,
        "set_a_file": str(set_a_path),
        "set_b_file": str(set_b_path),
        "feature_schema_treatments": treatments,
        "feature_schema_confounders": confounders,
        "missing_set_a_columns": missing_a,
        "missing_set_b_columns": missing_b,
        "missing_aggregate_columns": missing_macro,
        "split_summaries": split_summaries,
        "strict_temporal_validation": strict_temporal_validation,
        "item_category_audit": category_audit,
        "validation_candidates": validation_candidate_summaries,
        "evaluation_candidates": final_candidate_summaries,
        "candidate_protocol": "Purchase-only, 1 positive plus sampled negatives, warm users and warm target items",
    }
    preflight_path = output / f"DCML_Recommendation_Preflight_{SCRIPT_VERSION}.json"
    write_json(preflight, preflight_path, indent=2)
    if preflight_failures:
        raise RuntimeError(
            "Amazon DCML recommendation preflight failed: "
            + "; ".join(preflight_failures)
            + f". See {preflight_path}."
        )
    if args.preflight_only:
        logger.info(
            "DCML recommendation preflight passed | validation users=%s evaluation users=%s",
            {stage: len(value) for stage, value in validation_candidates.items()},
            {stage: len(value) for stage, value in final_candidates.items()},
        )
        return

    validation_history = sort_frame(pd.concat([set_a, b_train], ignore_index=True))
    final_history = sort_frame(pd.concat([set_a, set_b], ignore_index=True))
    validation_cuts = history_level_cuts(validation_history)
    final_cuts = history_level_cuts(final_history)
    validation_effect_h = event_history_levels(set_b, validation_cuts)
    final_effect_h = event_history_levels(set_b, final_cuts)
    residuals = fit_training_residuals(
        set_a,
        set_b,
        confounders,
        seed,
    )
    validation_prior, validation_prior_rows, validation_projection_rows = estimate_causal_prior(
        residuals,
        train_mask,
        validation_effect_h,
        "Set B training partition",
    )
    full_b_mask = np.ones(len(set_b), dtype=bool)
    final_prior, final_prior_rows, final_projection_rows = estimate_causal_prior(
        residuals,
        full_b_mask,
        final_effect_h,
        "Full Set B",
    )
    validation_stability_gates, validation_stability_rows = prior_stability_audit(
        residuals,
        set_b,
        train_mask,
        validation_effect_h,
        "Set B training partition",
        args.stability_min_sign_agreement,
        args.stability_min_correlation,
    )
    final_stability_gates, final_stability_rows = prior_stability_audit(
        residuals,
        set_b,
        full_b_mask,
        final_effect_h,
        "Full Set B",
        args.stability_min_sign_agreement,
        args.stability_min_correlation,
    )

    train_content, train_macro, _ = item_profiles(validation_history, item2id, content_columns)
    train_h = end_of_window_user_levels(validation_history, user2id, validation_cuts)
    train_sem_total, train_sem_category = semantic_history(
        validation_history, user2id, item2id, item_categories
    )
    state_audit_rows = user_state_audit_rows(
        "B validation start",
        validation_candidates,
        train_h,
        legacy_latest_user_levels(b_train, user2id),
        validation_cuts,
    )
    full_h = end_of_window_user_levels(final_history, user2id, final_cuts)
    state_audit_rows.extend(
        user_state_audit_rows(
            "Set C start",
            final_candidates,
            full_h,
            legacy_latest_user_levels(set_b, user2id),
            final_cuts,
        )
    )
    state_audit = pd.DataFrame(state_audit_rows)
    if (state_audit["Corrected_H0_Users"] > 0).any():
        failed = state_audit.loc[state_audit["Corrected_H0_Users"] > 0, ["Context", "Stage"]].to_dict("records")
        raise RuntimeError(f"Warm-start candidate users still have H=0 after cutoff-state reconstruction: {failed}")

    validation_model, validation_training_rows = train_model(
        b_train,
        user2id,
        item2id,
        train_content,
        train_h,
        args,
        seed,
        device,
    )
    validation_calibrations = {
        stage: estimate_score_calibration(
            validation_model,
            validation_candidates[stage],
            stage,
            train_content,
            train_macro,
            train_h,
            validation_prior,
            item_categories,
            train_sem_total,
            train_sem_category,
            content_columns,
            device,
        )
        for stage in STAGES
    }
    selected_lambdas, tuning_rows = select_causal_lambdas(
        validation_model,
        validation_candidates,
        train_content,
        train_macro,
        train_h,
        validation_prior,
        item_categories,
        train_sem_total,
        train_sem_category,
        content_columns,
        validation_calibrations,
        validation_stability_gates,
        args,
        device,
    )

    deployed_lambdas = {
        stage: float(selected_lambdas[stage]) if final_stability_gates.get(stage, False) else 0.0
        for stage in STAGES
    }
    for row in tuning_rows:
        stage = str(row["Stage"])
        row["Refit_Stability_Gate_Passed"] = bool(final_stability_gates.get(stage, False))
        row["Deployed_Causal_Lambda"] = deployed_lambdas[stage]

    full_content, full_macro, _ = item_profiles(final_history, item2id, content_columns)
    full_sem_total, full_sem_category = semantic_history(
        final_history, user2id, item2id, item_categories
    )
    final_model, final_training_rows = train_model(
        set_b,
        user2id,
        item2id,
        full_content,
        full_h,
        args,
        seed + 10_000,
        device,
    )
    final_calibrations = {
        stage: estimate_score_calibration(
            final_model,
            validation_candidates[stage],
            stage,
            full_content,
            full_macro,
            full_h,
            final_prior,
            item_categories,
            full_sem_total,
            full_sem_category,
            content_columns,
            device,
        )
        for stage in STAGES
    }
    checkpoint_path = output / f"DCML_Recommender_State_{SCRIPT_VERSION}.pt"
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "users": len(user2id),
            "items": len(item2id),
            "content_columns": content_columns,
            "embedding_dim": args.embedding_dim,
            "validation_selected_causal_lambdas": selected_lambdas,
            "selected_causal_lambdas": deployed_lambdas,
            "score_calibrations": final_calibrations,
            "seed": seed + 10_000,
        },
        checkpoint_path,
    )

    for stage in STAGES:
        candidates = final_candidates[stage]
        predictions, metric_values, causal_values, tie_sizes = score_candidates(
            final_model,
            candidates,
            stage,
            full_content,
            full_macro,
            full_h,
            final_prior,
            item_categories,
            full_sem_total,
            full_sem_category,
            content_columns,
            final_calibrations[stage],
            deployed_lambdas[stage],
            device,
            evaluate_targets=False,
        )
        prediction_path = output / f"DCML_{stage}_predictions_{SCRIPT_VERSION}.json"
        write_json(predictions, prediction_path)
        metadata = {
            "script_version": SCRIPT_VERSION,
            "data_version": DATA_VERSION,
            "model": "DCML (Ours)",
            "stage": stage,
            "independent_recommender": True,
            "analysis_temporal_status": "PASS_BY_CONSTRUCTION_CERTIFIED_ITEMS_ONLY",
            "analysis_population": "items with all selected MCRE reviews strictly before item-specific Set C entry",
            "full_sample_audit_status": certification_summary.get("audit_status"),
            "set_c_item_certification_rate": certification_summary.get("set_c_item_certification_rate"),
            "set_c_row_certification_rate": certification_summary.get("set_c_row_certification_rate"),
            "temporal_certification_item_file": certification_summary.get("item_certification_file"),
            "external_backbone": None,
            "architecture": (
                "Purchase-specific collaborative representation + MCRE content encoder + support-aware "
                "Set-A/Set-B-only DML causal-informed score"
            ),
            "training_split": "Set B",
            "validation_split": "chronological tail of Set B",
            "evaluation_split": "Set C",
            "set_c_table_read_during_training": False,
            "set_c_candidate_targets_used_for_training_or_selection": False,
            "test_outcomes_used_for_training_or_selection": False,
            "causal_prior_nuisance_split": "Set A",
            "causal_prior_effect_split": "Set B",
            "reported_causal_estimand_source": "none; recommendation prior is separate from the primary Set C DML estimates",
            "validation_selected_causal_lambda": selected_lambdas[stage],
            "selected_causal_lambda": deployed_lambdas[stage],
            "validation_prior_stability_gate_passed": bool(validation_stability_gates.get(stage, False)),
            "refit_prior_stability_gate_passed": bool(final_stability_gates.get(stage, False)),
            "lambda_selection_rule": (
                "maximize validation model-implied causal score subject to B-only prior-stability gating, "
                "mean retention, and paired one-sided NDCG@10 noninferiority"
            ),
            "causal_score_interpretation": "model-implied causal utility; not an off-policy estimate of realized uplift",
            "causal_prior_support_rule": (
                "N>=500, residual SD>=0.02, residual IQR>=0.02, and at least four occupied histogram bins; "
                "unsupported H groups fall back to ATE"
            ),
            "orthogonalization_scoring_basis": "algebraically equivalent coefficients on original treatment values",
            "user_state_definition": "Set A+B Purchase positives observed through the Set C cutoff; no Set C table access",
            "candidate_h0_users": int(
                state_audit.loc[
                    (state_audit["Context"] == "Set C start") & (state_audit["Stage"] == stage),
                    "Corrected_H0_Users",
                ].iloc[0]
            ),
            "relevance_retention_floor": args.relevance_retention_floor,
            "score_calibration": final_calibrations[stage],
            "score_calibration_split": "Set B chronological validation candidates; candidate labels not used",
            "training_targets": list(OUTCOMES),
            "multi_task_training": False,
            "training_edges_by_stage": {name: int(len(map_pairs(set_b, user2id, item2id, name))) for name in STAGES},
            "candidate_users": len(candidates),
            "candidate_sha256": sha256_file(candidate_path(root, stage)),
            "prediction_file": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "checkpoint_file": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "user_map_sha256": sha256_file(user_map_path),
            "item_map_sha256": sha256_file(item_map_path),
            "set_a_sha256": sha256_file(set_a_path),
            "set_b_sha256": sha256_file(set_b_path),
            "item_category_audit": category_audit,
            "seed": seed + 10_000,
            "mean_estimated_causal_score_at_10": float(np.mean(causal_values)),
        }
        write_json(metadata, output / f"DCML_{stage}_prediction_metadata_{SCRIPT_VERSION}.json", indent=2)
        logger.info(
            "DCML/%s scored users=%d lambda=%.3f; Set C outcomes were not evaluated in training",
            stage,
            len(candidates),
            deployed_lambdas[stage],
        )

    prior_rows = pd.DataFrame(validation_prior_rows + final_prior_rows)
    prior_rows.to_csv(
        output / f"DCML_Recommendation_Causal_Prior_{SCRIPT_VERSION}.csv",
        index=False,
    )
    pd.DataFrame(validation_projection_rows + final_projection_rows).to_csv(
        output / f"DCML_Recommendation_Orthogonalization_{SCRIPT_VERSION}.csv",
        index=False,
    )
    pd.DataFrame(validation_stability_rows + final_stability_rows).to_csv(
        output / f"DCML_Recommendation_Prior_Stability_{SCRIPT_VERSION}.csv",
        index=False,
    )
    state_audit.to_csv(output / f"DCML_User_State_Audit_{SCRIPT_VERSION}.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(output / f"DCML_Recommendation_Validation_Tuning_{SCRIPT_VERSION}.csv", index=False)
    validation_diagnostics = pd.DataFrame(validation_training_rows).assign(Training_Run="B_train_for_validation")
    final_diagnostics = pd.DataFrame(final_training_rows).assign(Training_Run="full_B_final_model")
    pd.concat([validation_diagnostics, final_diagnostics], ignore_index=True).to_csv(
        output / f"DCML_Recommender_Training_Diagnostics_{SCRIPT_VERSION}.csv",
        index=False,
    )
    write_json(
        {
            "script_version": SCRIPT_VERSION,
            "independent_recommender": True,
            "external_backbone": None,
            "analysis_temporal_status": "PASS_BY_CONSTRUCTION_CERTIFIED_ITEMS_ONLY",
            "analysis_population": "items with all selected MCRE reviews strictly before item-specific Set C entry",
            "full_sample_audit_status": certification_summary.get("audit_status"),
            "set_c_item_certification_rate": certification_summary.get("set_c_item_certification_rate"),
            "set_c_row_certification_rate": certification_summary.get("set_c_row_certification_rate"),
            "temporal_certification_item_file": certification_summary.get("item_certification_file"),
            "validation_selected_causal_lambdas": selected_lambdas,
            "selected_causal_lambdas": deployed_lambdas,
            "validation_prior_stability_gates": validation_stability_gates,
            "refit_prior_stability_gates": final_stability_gates,
            "prior_stability_gate_uses_set_c": False,
            "causal_prior_support_guard_applied": True,
            "causal_prior_unsupported_group_fallback": "whole H group to ATE",
            "prediction_semantic_missing_encoding": (
                "zero is a prediction-only encoding when no purchase history exists; it is not an estimable causal T_int_sem value"
            ),
            "orthogonalization_scoring_basis": "original_treatment_equivalent",
            "validation_h_level_cuts": validation_cuts.tolist(),
            "set_c_start_h_level_cuts": final_cuts.tolist(),
            "user_state_definition": "Purchase positives observed through the applicable cutoff, including Set A history",
            "set_c_candidate_h0_users": {
                stage: int(
                    state_audit.loc[
                        (state_audit["Context"] == "Set C start") & (state_audit["Stage"] == stage),
                        "Corrected_H0_Users",
                    ].iloc[0]
                )
                for stage in STAGES
            },
            "causal_identification_outputs_modified": False,
            "primary_causal_results_used_to_score_set_c": False,
            "set_c_table_read_during_training": False,
            "set_c_candidate_targets_used_for_training_or_selection": False,
            "recommendation_prior": (
                "nuisance models fitted on Set A; coefficients estimated on Set B only; support-aware CATE fallback; "
                "early/late Set B stability gate"
            ),
            "content_columns": content_columns,
            "score_calibrations": final_calibrations,
            "item_category_audit": category_audit,
            "hyperparameters": vars(args),
            "seed_offset": seed_offset,
            "device": str(device),
        },
        output / f"DCML_Recommender_Metadata_{SCRIPT_VERSION}.json",
        indent=2,
    )

    for stale_name in (
        f"DCML_Reranking_Sensitivity_{SCRIPT_VERSION}.csv",
        f"DCML_Reranking_Quality_Audit_{SCRIPT_VERSION}.csv",
        *(f"DCML_{stage}_candidate_scores_{SCRIPT_VERSION}.json" for stage in STAGES),
        f"Performance_Results_Summary_Hybrid_{SCRIPT_VERSION}.csv",
        f"Performance_Results_Hybrid_Diagnostics_{SCRIPT_VERSION}.csv",
        f"DCML_User_Metrics_{SCRIPT_VERSION}.csv",
        f"DCML_Evaluation_Metadata_{SCRIPT_VERSION}.json",
        f"Recommendation_Candidate_Sensitivity_{SCRIPT_VERSION}.csv",
        f"Recommendation_Candidate_Sensitivity_Summary_{SCRIPT_VERSION}.csv",
        f"Recommendation_Candidate_Sensitivity_Metadata_{SCRIPT_VERSION}.json",
        f"Table4_Baseline_Comparison_Full_{SCRIPT_VERSION}.csv",
        f"Recommendation_User_Metrics_{SCRIPT_VERSION}.csv",
        f"Recommendation_User_Metrics_Metadata_{SCRIPT_VERSION}.json",
        f"Paired_DCML_vs_Baselines_{SCRIPT_VERSION}.csv",
        f"DCML_Independent_Model_Audit_{SCRIPT_VERSION}.csv",
        f"Prediction_Bundle_Audit_{SCRIPT_VERSION}.csv",
        f"Comparison_Quality_Gate_{SCRIPT_VERSION}.json",
        f"Missing_Baseline_Predictions_{SCRIPT_VERSION}.csv",
        f"Invalid_Baseline_Outputs_{SCRIPT_VERSION}.csv",
        *(
            (
                f"Recommendation_Training_Seed_Runs_{SCRIPT_VERSION}.csv",
                f"Recommendation_Training_Seed_Summary_{SCRIPT_VERSION}.csv",
            )
            if seed_offset == 0
            else ()
        ),
    ):
        stale = output / stale_name
        if stale.is_file():
            stale.unlink()
    for stale_pattern in (
        f"DCML_Backbone_*_{SCRIPT_VERSION}.*",
        f"DCML_*_backbone_*_{SCRIPT_VERSION}.*",
        f"DCML_*_candidate_scores_{SCRIPT_VERSION}.json",
    ):
        for stale in output.glob(stale_pattern):
            stale.unlink()
    logger.info("Independent DCML recommender training completed; outputs: %s", output)


if __name__ == "__main__":
    main()
