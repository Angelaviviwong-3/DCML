#!/usr/bin/env python3
"""RQ1: cue-absence anchors and MCRE measurement-calibration ECDFs.

This figure describes marginal measurement behavior. Conditional treatment
support for DML is reported separately by the 20260722 positivity audit.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATASETS = {
    "Suning": "suning/item_multimodal_scalars/item_multimodal_scalars_full_3.parquet",
    "Amazon Appliances": (
        "amazon_appliances/item_multimodal_scalars/"
        "amazon_appliances_item_multimodal_scalars_merged_full_2.parquet"
    ),
    "Amazon Beauty": (
        "amazon_beauty/item_multimodal_scalars/"
        "amazon_beauty_item_multimodal_scalars_merged_full.parquet"
    ),
}

FEATURES = [
    ("T_int_fac", "Factual text density"),
    ("T_int_vis", "Functional visual saliency"),
    ("T_con_mkt", "Marketing visual saliency"),
]

COLORS = {
    "Suning": "#1976A3",
    "Amazon Appliances": "#D07A32",
    "Amazon Beauty": "#6B8E4E",
}


def calibrated_column(columns: set[str], raw: str) -> str:
    for candidate in (f"{raw}_calibrated", f"{raw}_calib"):
        if candidate in columns:
            return candidate
    raise KeyError(f"Missing calibrated column for {raw}")


def load_data(processed_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    zero_rows: list[dict] = []
    for dataset, relative in DATASETS.items():
        path = processed_root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        source = pd.read_parquet(path)
        available = set(source.columns)
        calibrated = {raw: calibrated_column(available, raw) for raw, _ in FEATURES}
        usecols = [raw for raw, _ in FEATURES] + list(calibrated.values())
        frame = source[usecols]
        for raw, title in FEATURES:
            values = pd.to_numeric(frame[raw], errors="coerce")
            observed = values.notna()
            zero_rows.append(
                {
                    "Dataset": dataset,
                    "Component": raw,
                    "Component_Label": title,
                    "Observed_Items": int(observed.sum()),
                    "Zero_Items": int(values.loc[observed].eq(0).sum()),
                    "Zero_Percentage": float(values.loc[observed].eq(0).mean() * 100),
                }
            )
            frames.append(
                pd.DataFrame(
                    {
                        "Dataset": dataset,
                        "Component": raw,
                        "Raw": values,
                        "Calibrated": pd.to_numeric(frame[calibrated[raw]], errors="coerce"),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(zero_rows)


def plot_ecdfs(long: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"font.family": "serif", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.6), sharey=True)
    for col_index, (component, title) in enumerate(FEATURES):
        subset = long.loc[long["Component"].eq(component)]
        for row_index, value_col in enumerate(("Raw", "Calibrated")):
            ax = axes[row_index, col_index]
            for dataset in DATASETS:
                values = subset.loc[subset["Dataset"].eq(dataset), value_col].dropna()
                sns.ecdfplot(values, ax=ax, color=COLORS[dataset], linewidth=1.8, label=dataset)
            ax.axvline(0, color="#444444", linestyle=":", linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("Raw MLLM score" if row_index == 0 else "Zero-preserved rank")
            ax.set_ylabel("Empirical cumulative probability" if col_index == 0 else "")
            if row_index == 1:
                ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0, 1.01)
    axes[0, 0].legend(frameon=False, title="Dataset", loc="lower right")
    for ax in [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.text(0.01, 0.965, "a", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.49, "b", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "Figure_6_rq1_semantic_support_20260722.pdf", bbox_inches="tight")
    fig.savefig(output / "Figure_6_rq1_semantic_support_20260722.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(str(_dcml_paths.project_path('processed_data'))),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(str(_dcml_paths.project_path('Unified_Visualization/output'))),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long, zero = load_data(args.processed_root)
    zero.to_csv(args.output_dir / "Table_4_cue_absence_anchors_20260722.csv", index=False)
    plot_ecdfs(long, args.output_dir)
    print(zero.to_string(index=False))


if __name__ == "__main__":
    main()
