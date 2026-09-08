#!/usr/bin/env python3
"""RQ5 Figure 11: composite-level ATE sensitivity with Amazon detail insets."""

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
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PROCESSED_ROOT = _dcml_paths.workspace_path(_dcml_paths.workspace_path(PROJECT_ROOT, "20260718_results"), 'processed_data')
SERVER_PROCESSED_ROOT = Path(str(_dcml_paths.project_path('processed_data')))
DEFAULT_OUTPUT_DIR = _dcml_paths.project_path("Unified_Visualization/output")
PDF_NAME = "Figure_11_ate_specification_sensitivity_20260728.pdf"
PNG_NAME = "Figure_11_ate_specification_sensitivity_20260728.png"

DATASETS = ("suning", "amazon_appliances", "amazon_beauty")
DATASET_LABELS = {
    "suning": "Suning",
    "amazon_appliances": "Amazon Appliances",
    "amazon_beauty": "Amazon Beauty",
}
VARIANTS = {
    "no_gs": "No projection",
    "reverse_analytical_to_heuristic": "Reverse order",
}
COLORS = {
    "no_gs": "#2B7A78",
    "reverse_analytical_to_heuristic": "#E07A5F",
}
MARKERS = {
    "no_gs": "o",
    "reverse_analytical_to_heuristic": "s",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
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


def default_processed_root() -> Path:
    return LOCAL_PROCESSED_ROOT if LOCAL_PROCESSED_ROOT.is_dir() else SERVER_PROCESSED_ROOT


def gs_path(processed_root: Path, dataset: str) -> Path:
    return (
        processed_root
        / dataset
        / "Final_Causal_Output"
        / f"{dataset}_GS_order_sensitivity_20260717.csv"
    )


def load_comparisons(processed_root: Path, dataset: str) -> pd.DataFrame:
    path = gs_path(processed_root, dataset)
    if not path.is_file():
        raise FileNotFoundError(f"Missing GS sensitivity result: {path}")
    data = pd.read_csv(path, low_memory=False)
    data = data[data["Resolution"].eq("Aggregate")].copy()
    if "H_level" in data.columns:
        data = data[data["H_level"].astype(str).eq("All (ATE)")]
    data["Coefficient"] = pd.to_numeric(data["Coefficient"], errors="coerce")

    keys = ["Stage", "Component"]
    primary = data[data["GS_Variant"].eq("standard_heuristic_to_analytical")][
        keys + ["Coefficient"]
    ].rename(columns={"Coefficient": "Primary_ATE"})
    frames = []
    for variant, label in VARIANTS.items():
        alternative = data[data["GS_Variant"].eq(variant)][
            keys + ["Coefficient"]
        ].rename(columns={"Coefficient": "Sensitivity_ATE"})
        merged = primary.merge(alternative, on=keys, how="inner").dropna()
        merged["Variant"] = variant
        merged["Variant_Label"] = label
        frames.append(merged)
    if not frames:
        raise RuntimeError(f"No comparable ATE estimates for {dataset}")
    result = pd.concat(frames, ignore_index=True)
    result["Dataset"] = dataset
    return result


def draw_scatter(ax: plt.Axes, data: pd.DataFrame, size: float, labels: bool) -> None:
    for variant, label in VARIANTS.items():
        subset = data[data["Variant"].eq(variant)]
        ax.scatter(
            subset["Primary_ATE"],
            subset["Sensitivity_ATE"],
            s=size,
            alpha=0.86,
            marker=MARKERS[variant],
            color=COLORS[variant],
            edgecolor="white",
            linewidth=0.35,
            label=label if labels else None,
            zorder=3,
        )


def set_equal_bounds(ax: plt.Axes, data: pd.DataFrame, minimum_pad: float) -> None:
    values = pd.concat([data["Primary_ATE"], data["Sensitivity_ATE"]], ignore_index=True)
    lower = min(float(values.min()), 0.0)
    upper = max(float(values.max()), 0.0)
    pad = max((upper - lower) * 0.08, minimum_pad)
    bounds = (lower - pad, upper + pad)
    ax.plot(bounds, bounds, color="#666666", linewidth=0.8, linestyle=":", zorder=1)
    ax.axhline(0, color="#C5C5C5", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#C5C5C5", linewidth=0.6, zorder=0)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=default_processed_root())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    processed_root = args.processed_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    panels = {dataset: load_comparisons(processed_root, dataset) for dataset in DATASETS}
    pd.concat(panels.values(), ignore_index=True).to_csv(
        output_dir / "Figure_11_ate_specification_sensitivity_data_20260728.csv",
        index=False,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.05), constrained_layout=True)
    for ax, dataset, letter in zip(axes, DATASETS, ("a", "b", "c")):
        data = panels[dataset]
        draw_scatter(ax, data, size=31, labels=dataset == "suning")
        set_equal_bounds(ax, data, minimum_pad=0.002)
        ax.set_xlabel("Primary-specification ATE")
        ax.set_ylabel("Sensitivity-specification ATE")
        ax.set_title(f"{letter}  {DATASET_LABELS[dataset]}", loc="left", fontweight="bold")

        if dataset != "suning":
            detail = data[~data["Component"].eq("T_con_soc")].copy()
            inset = ax.inset_axes([0.49, 0.08, 0.48, 0.44])
            draw_scatter(inset, detail, size=19, labels=False)
            set_equal_bounds(inset, detail, minimum_pad=0.00055)
            inset.set_title("Other cues", fontsize=7.2, pad=1.5)
            inset.tick_params(axis="both", labelsize=6, pad=1)
            inset.set_xlabel("")
            inset.set_ylabel("")
            ax.indicate_inset_zoom(inset, edgecolor="#888888", alpha=0.55, linewidth=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)

    pdf_path = output_dir / PDF_NAME
    png_path = output_dir / PNG_NAME
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, facecolor="white")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
