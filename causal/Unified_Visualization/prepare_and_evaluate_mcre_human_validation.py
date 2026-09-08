#!/usr/bin/env python3
"""Prepare and evaluate blinded human validation for the full MCRE inventory.

The default workflow never generates human scores. It creates a blinded annotation
template for three independent raters and evaluates completed real annotations.
``--synthetic-pipeline-test`` is only a software test; its files are explicitly
labelled SYNTHETIC_DO_NOT_REPORT and must not be used as research evidence.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "20260728"
CAUSAL_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = _dcml_paths.workspace_path(CAUSAL_ROOT, "Unified_Visualization/output")

DATASETS = {
    "suning": {
        "scalar_candidates": [
            "processed_data/suning/item_multimodal_scalars/checkpoint_scalars_full_3.csv",
            "processed_data/suning/item_multimodal_scalars/item_multimodal_scalars_full_3.parquet",
        ],
        "image_dir": "processed_data/suning/picture",
        "text": "processed_data/suning/item_feature&Confounder_price/item_text_price_matrix.csv",
    },
    "amazon_appliances": {
        "scalar_candidates": [
            "processed_data/amazon_appliances/item_multimodal_scalars/amazon_appliances_item_multimodal_scalars_merged_full_2.csv",
            "processed_data/amazon_appliances/item_multimodal_scalars/amazon_appliances_item_multimodal_scalars_merged_full_2.parquet",
        ],
        "image_dir": "processed_data/amazon_appliances/pictures",
        "text": "processed_data/amazon_appliances/item_feature&Confounder_price/item_text_price_matrix.csv",
    },
    "amazon_beauty": {
        "scalar_candidates": [
            "processed_data/amazon_beauty/item_multimodal_scalars/amazon_beauty_item_multimodal_scalars_merged_full.csv",
            "processed_data/amazon_beauty/item_multimodal_scalars/amazon_beauty_item_multimodal_scalars_merged_full.parquet",
        ],
        "image_dir": "processed_data/amazon_beauty/pictures",
        "text": "processed_data/amazon_beauty/item_feature&Confounder_price/item_text_price_matrix.csv",
    },
}

# Only MLLM-derived measurements are annotated. Marketplace signals and
# user-item category alignment are constructed from observed records.
CONSTRUCTS = {
    "T_con_mkt": ("Marketing salience", "MNC", "composite"),
    "T_int_vis": ("Functional visual salience", "MAI", "composite"),
    "T_int_fac": ("Factual density", "MAI", "shared"),
    "atomic_a_pri": ("Price shock", "MNC", "component"),
    "atomic_a_gft": ("Gift appeal", "MNC", "component"),
    "atomic_a_sub": ("Subsidy authenticity", "MNC", "component"),
    "atomic_a_urg": ("Urgency", "MNC", "component"),
    "atomic_a_spec": ("Specification overlay", "MAI", "component"),
    "atomic_a_str": ("Internal-structure view", "MAI", "component"),
}

RATERS = {
    "A": "multimodal measurement specialist (NLP, knowledge graphs, and multimodal modeling)",
    "B": "product-domain specialist (product R&D and technical specifications)",
    "C": "e-commerce operations specialist (promotion and marketplace operations)",
}

SPEARMAN_PERMUTATIONS = 1999


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def first_existing(candidates: list[str]) -> Path:
    for relative in candidates:
        path = _dcml_paths.workspace_path(CAUSAL_ROOT, relative)
        if path.exists():
            return path
    raise FileNotFoundError("No scalar source found:\n" + "\n".join(candidates))


def raw_score_column(frame: pd.DataFrame, construct: str) -> str:
    candidates = [construct, f"{construct}_raw"]
    for column in candidates:
        if column in frame.columns:
            return column
    raise KeyError(f"Missing raw MCRE measurement for {construct}")


def resolve_image(image_dir: Path, item_id: str) -> str:
    patterns = [f"{item_id}_MAIN.*", f"{item_id}_*", f"{item_id}.*"]
    for pattern in patterns:
        matches = sorted(image_dir.glob(pattern))
        if matches:
            return str(matches[0])
    return ""


def text_lookup(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, low_memory=False)
    except (UnicodeDecodeError, pd.errors.ParserError, OSError):
        # Text is annotation context only; scalar measurements and item IDs remain valid.
        return {}
    item_col = next((c for c in ["item_id", "parent_asin"] if c in frame.columns), None)
    text_col = next((c for c in ["text_W", "text", "aggregated_text"] if c in frame.columns), None)
    if item_col is None or text_col is None:
        return {}
    values = frame[[item_col, text_col]].drop_duplicates(item_col)
    return dict(zip(values[item_col].astype(str), values[text_col].fillna("").astype(str)))


def stratified_sample(frame: pd.DataFrame, columns: list[str], n: int, seed: int) -> pd.DataFrame:
    work = frame.copy()
    values = work[columns].apply(pd.to_numeric, errors="coerce").clip(0, 1)
    work["_mean_intensity"] = values.mean(axis=1)
    work["_absence_band"] = pd.cut(
        values.eq(0).sum(axis=1), bins=[-1, 0, 3, len(columns)], labels=["none", "some", "many"]
    ).astype(str)
    try:
        work["_intensity_band"] = pd.qcut(
            work["_mean_intensity"], q=4, labels=False, duplicates="drop"
        ).astype(str)
    except ValueError:
        work["_intensity_band"] = "0"
    work["_stratum"] = work["_absence_band"] + "_" + work["_intensity_band"]
    rng = np.random.default_rng(seed)
    groups = list(work.groupby("_stratum", dropna=False))
    target = max(1, int(np.ceil(n / max(1, len(groups)))))
    selected = []
    for _, group in groups:
        take = min(target, len(group))
        selected.extend(rng.choice(group.index.to_numpy(), size=take, replace=False).tolist())
    remaining = work.index.difference(pd.Index(selected)).to_numpy()
    if len(selected) < min(n, len(work)):
        extra_n = min(n, len(work)) - len(selected)
        selected.extend(rng.choice(remaining, size=extra_n, replace=False).tolist())
    return work.loc[selected[: min(n, len(work))]].drop(
        columns=["_mean_intensity", "_absence_band", "_intensity_band", "_stratum"]
    )


def prepare(samples_per_dataset: int, seed: int) -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotation_rows: list[dict] = []
    key_rows: list[dict] = []
    availability_rows: list[dict] = []
    for dataset, contract in DATASETS.items():
        source = first_existing(contract["scalar_candidates"])
        frame = read_table(source)
        if "item_id" not in frame.columns:
            raise KeyError(f"{source} does not contain item_id")
        columns = [raw_score_column(frame, construct) for construct in CONSTRUCTS]
        frame = frame.drop_duplicates("item_id")
        sample = stratified_sample(frame, columns, samples_per_dataset, seed)
        texts = text_lookup(_dcml_paths.workspace_path(CAUSAL_ROOT, contract["text"]))
        image_dir = _dcml_paths.workspace_path(CAUSAL_ROOT, contract["image_dir"])
        for construct, (name, pathway, level) in CONSTRUCTS.items():
            column = raw_score_column(frame, construct)
            observed = pd.to_numeric(frame[column], errors="coerce")
            availability_rows.append(
                {
                    "dataset": dataset,
                    "construct_code": construct,
                    "source_rows": int(len(frame)),
                    "measurement_available_rate": float(observed.notna().mean()),
                    "exact_zero_rate": float(observed.fillna(np.nan).eq(0).mean()),
                    "schema_valid_rate": np.nan,
                    "schema_valid_note": "raw extraction-log path status is not present in the scalar table",
                }
            )
            for row in sample[["item_id", column]].itertuples(index=False, name=None):
                item_id, raw_value = str(row[0]), float(row[1]) if pd.notna(row[1]) else np.nan
                key_rows.append(
                    {
                        "dataset": dataset,
                        "item_id": item_id,
                        "construct_code": construct,
                        "mllm_score_0_10": raw_value * 10 if np.isfinite(raw_value) else np.nan,
                        "source_file": str(source),
                    }
                )
                context = {
                    "dataset": dataset,
                    "item_id": item_id,
                    "image_path": resolve_image(image_dir, item_id),
                    "text_excerpt": texts.get(item_id, "")[:500],
                    "construct_code": construct,
                    "construct_name": name,
                    "pathway": pathway,
                    "measurement_level": level,
                }
                for rater_id, role in RATERS.items():
                    annotation_rows.append(
                        {
                            **context,
                            "rater_id": rater_id,
                            "rater_role": role,
                            "score_0_10": "",
                            "confidence_1_5": "",
                            "comment": "",
                        }
                    )
    annotation = pd.DataFrame(annotation_rows)
    key = pd.DataFrame(key_rows)
    availability = pd.DataFrame(availability_rows)
    template_path = OUTPUT_DIR / f"MCRE_Human_Annotation_Template_{VERSION}.csv"
    key_path = OUTPUT_DIR / f"MCRE_Human_Annotation_Blinded_Key_{VERSION}.csv"
    availability_path = OUTPUT_DIR / f"MCRE_Measurement_Availability_{VERSION}.csv"
    annotation.to_csv(template_path, index=False)
    for rater_id in RATERS:
        annotation.loc[annotation["rater_id"].eq(rater_id)].to_csv(
            OUTPUT_DIR / f"MCRE_Human_Annotation_Rater_{rater_id}_{VERSION}.csv",
            index=False,
        )
    key.to_csv(key_path, index=False)
    availability.to_csv(availability_path, index=False)
    return template_path, key_path, availability_path


def icc2k(matrix: np.ndarray) -> float:
    """Two-way random-effects, absolute-agreement ICC for mean ratings."""
    n, k = matrix.shape
    grand = matrix.mean()
    row_mean = matrix.mean(axis=1)
    col_mean = matrix.mean(axis=0)
    msr = k * np.square(row_mean - grand).sum() / max(1, n - 1)
    msc = n * np.square(col_mean - grand).sum() / max(1, k - 1)
    residual = matrix - row_mean[:, None] - col_mean[None, :] + grand
    mse = np.square(residual).sum() / max(1, (n - 1) * (k - 1))
    denominator = msr + (msc - mse) / max(1, n)
    return float((msr - mse) / denominator) if denominator else np.nan


def quadratic_weighted_kappa(left: np.ndarray, right: np.ndarray, levels: int = 11) -> float:
    left = np.clip(np.rint(left).astype(int), 0, levels - 1)
    right = np.clip(np.rint(right).astype(int), 0, levels - 1)
    observed = np.zeros((levels, levels), dtype=float)
    np.add.at(observed, (left, right), 1.0)
    expected = np.outer(np.bincount(left, minlength=levels), np.bincount(right, minlength=levels))
    expected = expected / max(1, len(left))
    indices = np.arange(levels, dtype=float)
    weights = np.square(indices[:, None] - indices[None, :]) / np.square(levels - 1)
    denominator = float(np.sum(weights * expected))
    return float(1.0 - np.sum(weights * observed) / denominator) if denominator > 0 else np.nan


def spearman_permutation(
    left: np.ndarray,
    right: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0:
        return np.nan, np.nan
    rho = float(np.dot(left_centered, right_centered) / denominator)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted = rng.permutation(right_centered)
        permuted_rho = float(np.dot(left_centered, permuted) / denominator)
        exceedances += abs(permuted_rho) >= abs(rho)
    p_value = (1 + exceedances) / (permutations + 1)
    return rho, float(p_value)


def evaluate(annotation_path: Path, key_path: Path, evidence_status: str = "REAL_HUMAN_ANNOTATION") -> None:
    if evidence_status == "REAL_HUMAN_ANNOTATION" and "SYNTHETIC" in annotation_path.name.upper():
        raise ValueError("A synthetic annotation file cannot be evaluated as real human evidence")
    annotations = pd.read_csv(annotation_path, low_memory=False)
    key = pd.read_csv(key_path, low_memory=False)
    required = {"dataset", "item_id", "construct_code", "rater_id", "score_0_10"}
    if missing := sorted(required.difference(annotations.columns)):
        raise KeyError(f"Annotation file missing columns: {missing}")
    unexpected_raters = sorted(set(annotations["rater_id"].astype(str)).difference(RATERS))
    if unexpected_raters:
        raise ValueError(f"Unexpected rater IDs: {unexpected_raters}")
    if "rater_role" in annotations.columns:
        expected_roles = annotations["rater_id"].astype(str).map(RATERS)
        mismatched_roles = annotations["rater_role"].astype(str).ne(expected_roles)
        if mismatched_roles.any():
            raise ValueError("Rater-role labels do not match the prespecified expert panel")
    annotations["score_0_10"] = pd.to_numeric(annotations["score_0_10"], errors="coerce")
    if evidence_status == "REAL_HUMAN_ANNOTATION" and annotations["score_0_10"].isna().any():
        missing_scores = int(annotations["score_0_10"].isna().sum())
        raise ValueError(
            f"Real human annotation is incomplete: {missing_scores} score_0_10 values are missing"
        )
    invalid = annotations["score_0_10"].notna() & ~annotations["score_0_10"].between(0, 10)
    if invalid.any():
        raise ValueError("Human scores must lie in [0, 10]")
    rows = []
    failures = []
    reliability = []
    group_cols = ["dataset", "construct_code"]
    for (dataset, construct), group in annotations.groupby(group_cols):
        pivot = group.pivot_table(
            index="item_id", columns="rater_id", values="score_0_10", aggfunc="first"
        ).reindex(columns=list(RATERS))
        complete = pivot.dropna()
        if complete.empty:
            continue
        values = complete.to_numpy(dtype=float)
        kappas = []
        for left in range(values.shape[1]):
            for right in range(left + 1, values.shape[1]):
                kappas.append(
                    quadratic_weighted_kappa(values[:, left], values[:, right])
                )
        construct_name, pathway, measurement_level = CONSTRUCTS[construct]
        reliability.append(
            {
                "dataset": dataset,
                "construct_code": construct,
                "construct_name": construct_name,
                "pathway": pathway,
                "measurement_level": measurement_level,
                "complete_items": int(len(complete)),
                "ICC_2k_absolute_agreement": icc2k(values),
                "mean_pairwise_quadratic_kappa": float(np.nanmean(kappas)),
                "evidence_status": evidence_status,
            }
        )
        consensus = complete.median(axis=1).rename("human_median_0_10").reset_index()
        merged = consensus.merge(
            key[(key["dataset"] == dataset) & (key["construct_code"] == construct)],
            on="item_id",
            how="inner",
            validate="one_to_one",
        ).dropna(subset=["mllm_score_0_10"])
        group_seed = 20260728 + zlib.crc32(f"{dataset}|{construct}".encode("utf-8"))
        rho, p_value = spearman_permutation(
            merged["human_median_0_10"].to_numpy(dtype=float),
            merged["mllm_score_0_10"].to_numpy(dtype=float),
            SPEARMAN_PERMUTATIONS,
            group_seed,
        )
        qwk = quadratic_weighted_kappa(
            merged["human_median_0_10"].to_numpy(dtype=float),
            merged["mllm_score_0_10"].to_numpy(dtype=float),
        )
        error = (merged["human_median_0_10"] - merged["mllm_score_0_10"]).abs()
        rows.append(
            {
                "dataset": dataset,
                "construct_code": construct,
                "construct_name": construct_name,
                "pathway": pathway,
                "measurement_level": measurement_level,
                "items": int(len(merged)),
                "spearman_rho": float(rho),
                "spearman_p": float(p_value),
                "quadratic_weighted_kappa": float(qwk),
                "MAE_0_10": float(error.mean()),
                "failure_rate_abs_error_ge_2_5": float(error.ge(2.5).mean()),
                "evidence_status": evidence_status,
            }
        )
        bad = merged.loc[error.ge(2.5)].copy()
        bad["absolute_error"] = error.loc[bad.index]
        bad["dataset"] = dataset
        bad["construct_code"] = construct
        failures.extend(bad.to_dict("records"))
    prefix = "SYNTHETIC_DO_NOT_REPORT_" if evidence_status != "REAL_HUMAN_ANNOTATION" else ""
    validation_frame = pd.DataFrame(rows)
    reliability_frame = pd.DataFrame(reliability)
    validation_frame.to_csv(OUTPUT_DIR / f"{prefix}MCRE_Human_Validation_Results_{VERSION}.csv", index=False)
    reliability_frame.to_csv(
        OUTPUT_DIR / f"{prefix}MCRE_Human_Interrater_Reliability_{VERSION}.csv", index=False
    )
    pd.DataFrame(failures).to_csv(
        OUTPUT_DIR / f"{prefix}MCRE_Human_Validation_Failure_Cases_{VERSION}.csv", index=False
    )
    if not validation_frame.empty and not reliability_frame.empty:
        summary = validation_frame.merge(
            reliability_frame[
                [
                    "dataset",
                    "construct_code",
                    "complete_items",
                    "ICC_2k_absolute_agreement",
                    "mean_pairwise_quadratic_kappa",
                ]
            ],
            on=["dataset", "construct_code"],
            how="left",
            validate="one_to_one",
        )
        summary.to_csv(
            OUTPUT_DIR / f"{prefix}Table_4_MCRE_Human_Validation_Summary_{VERSION}.csv",
            index=False,
        )
        dataset_summary = (
            summary.groupby("dataset", as_index=False)
            .agg(
                items_per_treatment_min=("items", "min"),
                items_per_treatment_max=("items", "max"),
                validated_treatments=("construct_code", "nunique"),
                ICC_2k_min=("ICC_2k_absolute_agreement", "min"),
                ICC_2k_max=("ICC_2k_absolute_agreement", "max"),
                spearman_rho_min=("spearman_rho", "min"),
                spearman_rho_max=("spearman_rho", "max"),
                quadratic_kappa_min=("quadratic_weighted_kappa", "min"),
                quadratic_kappa_max=("quadratic_weighted_kappa", "max"),
                max_large_error_rate=("failure_rate_abs_error_ge_2_5", "max"),
            )
            .sort_values("dataset")
        )
        dataset_summary.to_csv(
            OUTPUT_DIR
            / f"{prefix}Table_4_MCRE_Human_Validation_Dataset_Summary_{VERSION}.csv",
            index=False,
        )
    manifest = {
        "version": VERSION,
        "evidence_status": evidence_status,
        "annotation_file": str(annotation_path),
        "blinded_key_file": str(key_path),
        "constructs": list(CONSTRUCTS),
        "excluded_from_annotation": ["T_con_soc", "T_con_rat", "T_int_sem"],
        "reason": "these variables are constructed from observed marketplace or behavioral records, not MLLM extraction",
        "spearman_inference": f"two-sided plus-one permutation p-value with {SPEARMAN_PERMUTATIONS} permutations",
    }
    (OUTPUT_DIR / f"{prefix}MCRE_Human_Validation_Manifest_{VERSION}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def combine_rater_files(paths: list[Path]) -> Path:
    if len(paths) != len(RATERS):
        raise ValueError(f"Expected {len(RATERS)} completed rater files")
    frames = []
    observed_ids = set()
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        ids = set(frame["rater_id"].astype(str).unique())
        if len(ids) != 1:
            raise ValueError(f"{path} must contain exactly one rater_id")
        rater_id = next(iter(ids))
        if rater_id not in RATERS or rater_id in observed_ids:
            raise ValueError(f"Unexpected or repeated rater_id in {path}: {rater_id}")
        observed_ids.add(rater_id)
        frames.append(frame)
    if observed_ids != set(RATERS):
        raise ValueError(f"Missing rater files: {sorted(set(RATERS).difference(observed_ids))}")
    combined = pd.concat(frames, ignore_index=True)
    score = pd.to_numeric(combined["score_0_10"], errors="coerce")
    if score.isna().any():
        raise ValueError(f"Completed rater files contain {int(score.isna().sum())} missing scores")
    output = OUTPUT_DIR / f"MCRE_Human_Annotations_{VERSION}.csv"
    combined.to_csv(output, index=False)
    return output


def synthetic_pipeline_test(template_path: Path, key_path: Path, seed: int) -> Path:
    template = pd.read_csv(template_path, low_memory=False)
    key = pd.read_csv(key_path, low_memory=False)
    merged = template.merge(key, on=["dataset", "item_id", "construct_code"], how="left")
    rng = np.random.default_rng(seed)
    # These role-specific settings test the workflow only. They encode plausible
    # differences in professional precision without creating research evidence.
    role_sd = {"A": 0.72, "B": 0.92, "C": 0.90}
    role_bias = {"A": 0.00, "B": 0.08, "C": -0.06}
    technical_constructs = {"T_int_vis", "T_int_fac", "atomic_a_spec", "atomic_a_str"}
    operations_constructs = {
        "T_con_mkt",
        "atomic_a_pri",
        "atomic_a_gft",
        "atomic_a_sub",
        "atomic_a_urg",
    }
    construct_bias = {
        "atomic_a_spec": {"B": 0.25},
        "atomic_a_str": {"B": 0.20},
        "atomic_a_urg": {"C": 0.25},
        "T_con_mkt": {"C": 0.20},
    }
    scores = []
    for row in merged.itertuples(index=False):
        base = float(row.mllm_score_0_10)
        if not np.isfinite(base):
            scores.append(np.nan)
            continue
        bias = role_bias[row.rater_id] + construct_bias.get(row.construct_code, {}).get(row.rater_id, 0.0)
        dispersion = role_sd[row.rater_id]
        if row.rater_id == "B" and row.construct_code in technical_constructs:
            dispersion = 0.62
        if row.rater_id == "C" and row.construct_code in operations_constructs:
            dispersion = 0.60
        # Larger disagreement near mid-scale reflects harder, ambiguous cases.
        ambiguity = 1.0 + 0.35 * (1.0 - abs(base - 5.0) / 5.0)
        score = np.clip(np.rint(base + bias + rng.normal(0, dispersion * ambiguity)), 0, 10)
        scores.append(float(score))
    merged["score_0_10"] = scores
    confidence = np.clip(
        np.rint(5 - np.abs(merged["score_0_10"] - merged["mllm_score_0_10"]) / 2), 1, 5
    )
    merged["confidence_1_5"] = pd.Series(confidence, dtype="Int64")
    output = OUTPUT_DIR / f"SYNTHETIC_DO_NOT_REPORT_MCRE_Human_Annotations_{VERSION}.csv"
    merged[template.columns].to_csv(output, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "combine", "evaluate", "all"], default="prepare")
    parser.add_argument("--samples-per-dataset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--annotation-file", type=Path)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--rater-files", type=Path, nargs=3)
    parser.add_argument("--synthetic-pipeline-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = args.annotation_file
    key = args.key_file
    if args.mode == "combine":
        if not args.rater_files:
            raise ValueError("Combine mode requires --rater-files A.csv B.csv C.csv")
        combined = combine_rater_files(args.rater_files)
        print(f"combined real annotations: {combined}")
        return
    if args.mode in {"prepare", "all"}:
        template, key, availability = prepare(args.samples_per_dataset, args.seed)
        print(f"annotation template: {template}")
        print(f"blinded key: {key}")
        print(f"availability audit: {availability}")
    if args.synthetic_pipeline_test:
        if template is None or key is None:
            raise ValueError("Synthetic test requires prepared template and key files")
        synthetic = synthetic_pipeline_test(template, key, args.seed)
        evaluate(synthetic, key, evidence_status="SYNTHETIC_PIPELINE_TEST_ONLY")
        print(f"synthetic pipeline test: {synthetic}")
    elif args.mode in {"evaluate", "all"}:
        if template is None or key is None:
            raise ValueError("Evaluation requires --annotation-file and --key-file")
        evaluate(template, key)


if __name__ == "__main__":
    main()
