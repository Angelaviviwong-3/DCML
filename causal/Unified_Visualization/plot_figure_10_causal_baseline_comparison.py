#!/usr/bin/env python3
"""RQ4 Figure 10: complete comparison with alternative causal estimators."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PRIMARY_ROOT = _dcml_paths.workspace_path(PROJECT_ROOT, "20260718_results/processed_data")
LOCAL_ALTERNATIVE_ROOT = (
    _dcml_paths.workspace_path(PROJECT_ROOT, "20260719_results/xzhe162/wh_workspace/casual/causal/processed_data")
)
LOCAL_BEAUTY_ALTERNATIVE = (
    _dcml_paths.workspace_path(PROJECT_ROOT, "20260720_results/processed_data/amazon_beauty/Final_Causal_Output", "amazon_beauty_Causal_Baseline_Comparison_Support_Consistent_20260720.csv")
)
SERVER_ROOT = Path(str(_dcml_paths.project_path('processed_data')))
DEFAULT_OUTPUT_DIR = _dcml_paths.project_path("Unified_Visualization/output")
PDF_NAME = "Figure_10_causal_baseline_comparison_20260727.pdf"
PNG_NAME = "Figure_10_causal_baseline_comparison_20260727.png"

DATASETS = ("suning", "amazon_appliances", "amazon_beauty")
DATASET_LABELS = {
    "suning": "Suning",
    "amazon_appliances": "Amazon Appliances",
    "amazon_beauty": "Amazon Beauty",
}
ESTIMATOR_LABELS = {
    "Standard DML (No GS)": "DML, no projection",
    "Naive OLS": "Controlled OLS",
    "Marginal GPS-IPW": "Marginal GPS-IPW",
}
COLORS = {
    "Standard DML (No GS)": "#2B6F6D",
    "Naive OLS": "#C8573C",
    "Marginal GPS-IPW": "#65758B",
}
MARKERS = {"Aggregate": "o", "Disaggregate": "s"}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def default_primary_root() -> Path:
    return LOCAL_PRIMARY_ROOT if LOCAL_PRIMARY_ROOT.is_dir() else SERVER_ROOT


def default_alternative_root() -> Path:
    return LOCAL_ALTERNATIVE_ROOT if LOCAL_ALTERNATIVE_ROOT.is_dir() else SERVER_ROOT


def primary_path(root: Path, dataset: str) -> Path:
    return _dcml_paths.workspace_path(root, dataset, "Final_Causal_Output", f"{dataset}_GS_order_sensitivity_20260717.csv")


def alternative_path(root: Path, dataset: str) -> Path:
    if dataset == "amazon_beauty":
        corrected = (
            _dcml_paths.workspace_path(root, dataset, "Final_Causal_Output", "amazon_beauty_Causal_Baseline_Comparison_Support_Consistent_20260720.csv")
        )
        if corrected.is_file():
            return corrected
        if LOCAL_BEAUTY_ALTERNATIVE.is_file():
            return LOCAL_BEAUTY_ALTERNATIVE
    return _dcml_paths.workspace_path(root, dataset, "Final_Causal_Output", f"{dataset}_Causal_Baseline_Comparison_20260719.csv")


def load_comparisons(primary_root: Path, alternative_root: Path, dataset: str) -> pd.DataFrame:
    primary_file = primary_path(primary_root, dataset)
    alternative_file = alternative_path(alternative_root, dataset)
    if not primary_file.is_file():
        raise FileNotFoundError(f"Missing primary result: {primary_file}")
    if not alternative_file.is_file():
        raise FileNotFoundError(f"Missing causal baseline result: {alternative_file}")

    primary = pd.read_csv(primary_file, low_memory=False)
    primary = primary[primary["GS_Variant"].eq("standard_heuristic_to_analytical")].copy()
    if "H_level" in primary.columns:
        primary = primary[primary["H_level"].astype(str).eq("All (ATE)")]
    primary["Coefficient"] = pd.to_numeric(primary["Coefficient"], errors="coerce")
    primary = primary.dropna(subset=["Coefficient"])
    primary = primary[["Resolution", "Stage", "Component", "Coefficient"]].rename(
        columns={"Coefficient": "DCML_ATE"}
    )

    alternatives = pd.read_csv(alternative_file, low_memory=False)
    alternatives["Coefficient"] = pd.to_numeric(alternatives["Coefficient"], errors="coerce")
    blocks = []
    for estimator in ESTIMATOR_LABELS:
        subset = alternatives[alternatives["Estimator"].eq(estimator)].copy()
        if estimator != "Marginal GPS-IPW" and "Support_Consistent_20260720" not in alternative_file.name:
            if dataset == "amazon_beauty":
                specifications = {
                    "Full_certified_5_without_T_int_sem",
                    "Full_certified_9_without_T_int_sem",
                }
            else:
                specifications = {"History_complete_6", "History_complete_10"}
            subset = subset[subset["Model_Specification"].isin(specifications)]
        subset = subset[["Resolution", "Stage", "Component", "Coefficient"]].rename(
            columns={"Coefficient": "Alternative_ATE"}
        )
        merged = primary.merge(subset, on=["Resolution", "Stage", "Component"], how="inner")
        merged = merged.dropna(subset=["DCML_ATE", "Alternative_ATE"])
        merged = merged.drop_duplicates(["Resolution", "Stage", "Component"])
        merged["Estimator"] = estimator
        blocks.append(merged)
    if not blocks:
        raise RuntimeError(f"No matched causal estimates for {dataset}")
    output = pd.concat(blocks, ignore_index=True)
    output.insert(0, "Dataset", dataset)
    return output


def panel_summary(data: pd.DataFrame) -> str:
    lines = []
    for estimator in ESTIMATOR_LABELS:
        subset = data[data["Estimator"].eq(estimator)]
        concordance = (
            np.sign(subset["DCML_ATE"]) == np.sign(subset["Alternative_ATE"])
        ).mean()
        correlation = subset[["DCML_ATE", "Alternative_ATE"]].corr().iloc[0, 1]
        short = {
            "Standard DML (No GS)": "No-proj. DML",
            "Naive OLS": "OLS",
            "Marginal GPS-IPW": "GPS-IPW",
        }[estimator]
        lines.append(f"{short}: r={correlation:.2f}, sign={concordance:.0%}")
    return "\n".join(lines)


def scatter_estimates(ax: plt.Axes, data: pd.DataFrame, *, with_labels: bool, size: float) -> None:
    for estimator in ESTIMATOR_LABELS:
        for resolution in MARKERS:
            subset = data[
                data["Estimator"].eq(estimator) & data["Resolution"].eq(resolution)
            ]
            if subset.empty:
                continue
            ax.scatter(
                subset["DCML_ATE"],
                subset["Alternative_ATE"],
                s=size,
                alpha=0.72,
                marker=MARKERS[resolution],
                color=COLORS[estimator],
                edgecolors="white",
                linewidths=0.35,
                label=(
                    ESTIMATOR_LABELS[estimator]
                    if with_labels and resolution == "Aggregate"
                    else None
                ),
            )


def set_equal_bounds(ax: plt.Axes, data: pd.DataFrame, minimum_pad: float) -> tuple[float, float]:
    values = pd.concat([data["DCML_ATE"], data["Alternative_ATE"]], ignore_index=True)
    lower, upper = float(values.min()), float(values.max())
    pad = max((upper - lower) * 0.08, minimum_pad)
    bounds = (lower - pad, upper + pad)
    ax.plot(bounds, bounds, color="#555555", linewidth=0.8, linestyle=":")
    ax.axhline(0, color="#BBBBBB", linewidth=0.6)
    ax.axvline(0, color="#BBBBBB", linewidth=0.6)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    return bounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=default_primary_root())
    parser.add_argument("--alternative-root", type=Path, default=default_alternative_root())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    primary_root = args.primary_root.expanduser().resolve()
    alternative_root = args.alternative_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    all_data = {
        dataset: load_comparisons(primary_root, alternative_root, dataset)
        for dataset in DATASETS
    }
    comparison_data = pd.concat(all_data.values(), ignore_index=True)
    comparison_data.to_csv(
        output_dir / "Figure_10_causal_baseline_comparison_data_20260727.csv",
        index=False,
    )
    summary_rows = []
    for (dataset, estimator), subset in comparison_data.groupby(["Dataset", "Estimator"]):
        summary_rows.append(
            {
                "Dataset": dataset,
                "Estimator": ESTIMATOR_LABELS[estimator],
                "Compared_Estimates": len(subset),
                "Pearson_Correlation": subset[["DCML_ATE", "Alternative_ATE"]].corr().iloc[0, 1],
                "Sign_Agreement": (
                    np.sign(subset["DCML_ATE"]) == np.sign(subset["Alternative_ATE"])
                ).mean(),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "Figure_10_causal_baseline_comparison_summary_20260727.csv",
        index=False,
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), constrained_layout=True)
    for ax, dataset, letter in zip(axes, DATASETS, ("a", "b", "c")):
        data = all_data[dataset]
        scatter_estimates(ax, data, with_labels=True, size=25)
        set_equal_bounds(ax, data, minimum_pad=0.002)
        ax.set_xlabel("DCML ATE")
        ax.set_ylabel("Alternative-estimator effect")
        ax.set_title(f"{letter}  {DATASET_LABELS[dataset]}", loc="left", fontweight="bold")
        ax.text(
            0.03,
            0.97,
            panel_summary(data),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.1,
            bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.92, "pad": 3},
        )
        if dataset != "suning":
            zoom_data = data[~data["Component"].eq("T_con_soc")].copy()
            zoom_ax = ax.inset_axes([0.51, 0.08, 0.46, 0.43])
            scatter_estimates(zoom_ax, zoom_data, with_labels=False, size=15)
            set_equal_bounds(zoom_ax, zoom_data, minimum_pad=0.0007)
            zoom_ax.set_title("Other cues", fontsize=7, pad=1.5)
            zoom_ax.tick_params(axis="both", labelsize=6, pad=1)
            zoom_ax.set_xlabel("")
            zoom_ax.set_ylabel("")
            ax.indicate_inset_zoom(zoom_ax, edgecolor="#888888", alpha=0.55, linewidth=0.6)

    estimator_handles, estimator_labels = axes[0].get_legend_handles_labels()
    marker_handles = [
        plt.Line2D([], [], color="#555555", marker="o", linestyle="None", label="Composite-level"),
        plt.Line2D([], [], color="#555555", marker="s", linestyle="None", label="Component-level"),
    ]
    fig.legend(
        estimator_handles + marker_handles,
        estimator_labels + [h.get_label() for h in marker_handles],
        loc="outside lower center",
        ncol=5,
        fontsize=8,
    )

    pdf_path = output_dir / PDF_NAME
    png_path = output_dir / PNG_NAME
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, facecolor="white")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
