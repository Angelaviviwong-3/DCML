#!/usr/bin/env python3
"""RQ2 Figure 8: strict-reversal magnitudes under both cue specifications."""

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
DEFAULT_ATE = (
    _dcml_paths.workspace_path(PROJECT_ROOT, "20260718_results/processed_data/suning/Final_Causal_Output/"
    "Final_ATE_Results_20260717.csv")
)
DEFAULT_PROFILE = (
    _dcml_paths.workspace_path(PROJECT_ROOT, "20260720_results/processed_data/suning/Final_Causal_Output/"
    "suning_IQR_Scaled_Effect_Profiles_20260720.csv")
)
DEFAULT_OUTPUT_DIR = _dcml_paths.project_path("Unified_Visualization/output")
PDF_NAME = "Figure_8_rq2_dual_specification_iqr_effects_20260723.pdf"
PNG_NAME = "Figure_8_rq2_dual_specification_iqr_effects_20260723.png"

PRIMARY_ROLE = "primary_history_eligible_aggregate_6"
STAGES = ["click", "cart", "purchase"]
STAGE_LABELS = {"click": "Click", "cart": "Cart", "purchase": "Purchase"}
CUES = ["T_con_soc", "T_int_sem"]
CUE_LABELS = {
    "T_con_soc": "Social proof",
    "T_int_sem": "Historical category affinity",
}
COLORS = {"T_con_soc": "#7B3F66", "T_int_sem": "#4D7D43"}


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


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_plot_data(ate_path: Path, profile_path: Path) -> pd.DataFrame:
    if not ate_path.is_file():
        raise FileNotFoundError(f"Missing RQ2 ATE result: {ate_path}")
    if not profile_path.is_file():
        raise FileNotFoundError(f"Missing RQ2 IQR profile: {profile_path}")
    ate = pd.read_csv(ate_path, low_memory=False)
    profile = pd.read_csv(profile_path, low_memory=False)
    if "H_level" in ate:
        ate = ate[ate["H_level"].astype(str).eq("All (ATE)")]
    ate = ate[as_bool(ate["Estimable"])]

    primary = profile[
        profile["Specification_Role"].eq(PRIMARY_ROLE)
        & profile["Component"].isin(CUES)
    ].copy()
    iqr_width = (
        primary.groupby("Component")["Analysis_Sample_Residual_IQR"].first().to_dict()
    )
    if set(iqr_width) != set(CUES):
        raise RuntimeError("The primary Suning IQR profile does not contain both strict-reversal cues.")

    rows = ate[
        ate["Resolution"].isin(["Aggregate", "Disaggregate"])
        & ate["Component"].isin(CUES)
        & ate["Stage"].isin(STAGES)
    ].copy()
    rows["IQR_Width"] = rows["Component"].map(iqr_width)
    rows["Effect_pp"] = rows["Coefficient"] * rows["IQR_Width"] * 100
    rows["CI_Lower_pp"] = rows["CI_Lower_95"] * rows["IQR_Width"] * 100
    rows["CI_Upper_pp"] = rows["CI_Upper_95"] * rows["IQR_Width"] * 100
    return rows


def plot_resolution(
    ax: plt.Axes,
    data: pd.DataFrame,
    resolution: str,
    title: str,
) -> None:
    subset = data[data["Resolution"].eq(resolution)].copy()
    for cue in CUES:
        rows = subset[subset["Component"].eq(cue)].copy()
        rows["stage_order"] = rows["Stage"].map(
            {stage: index for index, stage in enumerate(STAGES)}
        )
        rows = rows.sort_values("stage_order")
        ax.errorbar(
            rows["stage_order"],
            rows["Effect_pp"],
            yerr=[
                rows["Effect_pp"] - rows["CI_Lower_pp"],
                rows["CI_Upper_pp"] - rows["Effect_pp"],
            ],
            marker="o",
            markersize=5,
            linewidth=1.5,
            capsize=3,
            color=COLORS[cue],
            label=CUE_LABELS[cue],
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(range(3), [STAGE_LABELS[stage] for stage in STAGES])
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ate", type=Path, default=DEFAULT_ATE)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    ate_path = args.ate.expanduser().resolve()
    profile_path = args.profile.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    data = load_plot_data(ate_path, profile_path)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True, constrained_layout=True)
    plot_resolution(axes[0], data, "Aggregate", "a  Composite-level cue specification")
    plot_resolution(axes[1], data, "Disaggregate", "b  Component-level cue specification")
    axes[0].set_ylabel("Outcome change for an IQR increase (percentage points)")
    axes[0].legend(loc="upper left", fontsize=8)

    pdf_path = output_dir / PDF_NAME
    png_path = output_dir / PNG_NAME
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, facecolor="white")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
