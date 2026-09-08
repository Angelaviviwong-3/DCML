#!/usr/bin/env python3
"""RQ1: summarize cross-path orthogonalization and joint conditioning.

The detailed correlation matrices remain available as auxiliary diagnostics.
The main-text figure reports the maximum absolute MAI--MNC correlation for the
composite-level specification and joint condition numbers for both cue
specifications.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import seaborn as sns


PHASE_ORDER = [
    "Set B before projection",
    "Set B after projection",
    "Set C frozen projection",
]

PHASE_LABELS = [
    "Set B\nbefore projection",
    "Set B\nafter projection",
    "Set C\nfrozen projection",
]

DATASET_ORDER = ["Suning", "Amazon Appliances", "Amazon Beauty"]

DATASET_SLUGS = {
    "Suning": "suning",
    "Amazon Appliances": "amazon_appliances",
    "Amazon Beauty": "amazon_beauty",
}

SPECIFICATIONS = {
    "macro_cue": "Composite-level",
    "fine_grained_cue": "Component-level",
}

SPECIFICATION_COLORS = {
    "Composite-level": "#7A5195",
    "Component-level": "#B45F4A",
}

COLORS = {
    "Suning": "#1976A3",
    "Amazon Appliances": "#D07A32",
    "Amazon Beauty": "#5B7F3A",
}

MARKERS = {
    "Suning": "o",
    "Amazon Appliances": "s",
    "Amazon Beauty": "D",
}

LABEL_OFFSETS = {
    "Suning": {0: (6, -14), 2: (6, 8)},
    "Amazon Appliances": {0: (6, 8), 2: (6, -14)},
    "Amazon Beauty": {0: (6, 8), 2: (6, 5)},
}


def load_diagnostics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "Dataset",
        "Diagnostic_Sample",
        "Max_Absolute_Cross_Path_Correlation",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{path}: missing columns {missing}")
    frame = frame.loc[
        frame["Dataset"].isin(DATASET_ORDER)
        & frame["Diagnostic_Sample"].isin(PHASE_ORDER)
    ].copy()
    frame["Max_Absolute_Cross_Path_Correlation"] = pd.to_numeric(
        frame["Max_Absolute_Cross_Path_Correlation"], errors="raise"
    )
    counts = frame.groupby("Dataset")["Diagnostic_Sample"].nunique()
    incomplete = [dataset for dataset in DATASET_ORDER if counts.get(dataset, 0) != 3]
    if incomplete:
        raise ValueError(f"Incomplete RQ1 diagnostics for: {', '.join(incomplete)}")
    return frame


def find_joint_file(root: Path, dataset: str) -> Path:
    slug = DATASET_SLUGS[dataset]
    filename = f"{slug}_DML_joint_residual_identification_20260722.csv"
    candidates = [_dcml_paths.workspace_path(root, filename), _dcml_paths.workspace_path(root, slug, "audit", filename)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {filename}; checked {candidates}")


def load_joint_diagnostics(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset in DATASET_ORDER:
        frame = pd.read_csv(find_joint_file(root, dataset))
        frame["Dataset_Label"] = dataset
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[
        combined["Specification"].isin(SPECIFICATIONS)
        & combined["Projection_Status"].isin(
            ["before_projection", "frozen_projection"]
        )
    ].copy()
    combined["Specification_Label"] = combined["Specification"].map(SPECIFICATIONS)
    combined["Correlation_Condition_Number"] = pd.to_numeric(
        combined["Correlation_Condition_Number"], errors="raise"
    )
    expected = len(DATASET_ORDER) * len(SPECIFICATIONS) * 2
    if len(combined) != expected:
        raise ValueError(f"Expected {expected} joint diagnostic rows, found {len(combined)}")
    return combined


def plot_cross_path(ax: plt.Axes, frame: pd.DataFrame) -> None:
    x = list(range(len(PHASE_ORDER)))
    for dataset in DATASET_ORDER:
        subset = (
            frame.loc[frame["Dataset"].eq(dataset)]
            .set_index("Diagnostic_Sample")
            .loc[PHASE_ORDER]
        )
        values = subset["Max_Absolute_Cross_Path_Correlation"].to_numpy(dtype=float)
        ax.plot(
            x,
            values,
            color=COLORS[dataset],
            marker=MARKERS[dataset],
            markersize=6,
            linewidth=1.8,
            label=dataset,
            zorder=3,
        )
        for phase_index in (0, 2):
            ax.annotate(
                f"{values[phase_index]:.3f}",
                (x[phase_index], values[phase_index]),
                xytext=LABEL_OFFSETS[dataset][phase_index],
                textcoords="offset points",
                color=COLORS[dataset],
                fontsize=7.5,
            )

    ax.text(
        1,
        0.018,
        r"Set B maxima $<10^{-4}$",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#333333",
    )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, PHASE_LABELS)
    ax.set_ylabel(r"Maximum absolute MAI--MNC correlation, $|r|$")
    ax.set_ylim(-0.015, 0.90)
    ax.set_xlim(-0.12, 2.18)
    ax.legend(frameon=False, loc="upper right", fontsize=7.8)
    ax.grid(axis="x", visible=False)
    ax.set_title("Composite-level cross-path association", fontsize=10)
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)
    sns.despine(ax=ax)


def plot_joint_conditioning(ax: plt.Axes, frame: pd.DataFrame) -> None:
    x_positions = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    offsets = {"Composite-level": -0.12, "Component-level": 0.12}
    for dataset in DATASET_ORDER:
        for specification in SPECIFICATIONS.values():
            subset = frame.loc[
                frame["Dataset_Label"].eq(dataset)
                & frame["Specification_Label"].eq(specification)
            ].set_index("Projection_Status")
            before = float(subset.loc["before_projection", "Correlation_Condition_Number"])
            frozen = float(subset.loc["frozen_projection", "Correlation_Condition_Number"])
            x = x_positions[dataset] + offsets[specification]
            color = SPECIFICATION_COLORS[specification]
            ax.annotate(
                "",
                xy=(x, frozen),
                xytext=(x, before),
                arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.7},
            )
            ax.scatter(
                [x],
                [before],
                marker="o",
                s=46,
                facecolors="white",
                edgecolors=color,
                linewidths=1.5,
                zorder=3,
            )
            ax.scatter(
                [x],
                [frozen],
                marker="s",
                s=42,
                facecolors=color,
                edgecolors=color,
                linewidths=1.2,
                zorder=3,
            )

    ax.set_yscale("log")
    ax.set_ylim(1, 100)
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(range(len(DATASET_ORDER)), ["Suning", "Appliances", "Beauty"])
    ax.set_ylabel("Residual correlation condition number")
    ax.set_title("Set C joint conditioning", fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.text(-0.12, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)
    legend = [
        Line2D([0], [0], color=SPECIFICATION_COLORS["Composite-level"], lw=2, label="Composite-level"),
        Line2D([0], [0], color=SPECIFICATION_COLORS["Component-level"], lw=2, label="Component-level"),
        Line2D([0], [0], marker="o", color="#444444", markerfacecolor="white", lw=0, label="Before projection"),
        Line2D([0], [0], marker="s", color="#444444", markerfacecolor="#444444", lw=0, label="Frozen projection"),
    ]
    ax.legend(
        handles=legend,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=7.8,
    )
    sns.despine(ax=ax)


def plot(frame: pd.DataFrame, joint: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    plot_cross_path(axes[0], frame)
    plot_joint_conditioning(axes[1], joint)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Figure_6_rq1_cross_path_orthogonalization_20260722"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            str(_dcml_paths.project_path('Unified_Visualization/output/RQ1_gs_correlation_diagnostics_20260722.csv'))
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            str(_dcml_paths.project_path('Unified_Visualization/output'))
        ),
    )
    parser.add_argument(
        "--joint-root",
        type=Path,
        default=Path(str(_dcml_paths.project_path('processed_data'))),
        help="Processed-data root or flat directory containing the three joint diagnostic CSV files.",
    )
    args = parser.parse_args()
    plot(
        load_diagnostics(args.input_csv),
        load_joint_diagnostics(args.joint_root),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
