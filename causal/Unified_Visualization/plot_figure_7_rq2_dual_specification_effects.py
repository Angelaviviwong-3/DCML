#!/usr/bin/env python3
"""RQ2 Figure 7: all Suning funnel effects under both cue specifications."""

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
DEFAULT_RESULTS_ROOT = _dcml_paths.workspace_path(PROJECT_ROOT, "20260723_results")
DEFAULT_OUTPUT_DIR = _dcml_paths.project_path("Unified_Visualization/output")
PDF_NAME = "Figure_7_rq2_dual_specification_effects_20260723.pdf"
PNG_NAME = "Figure_7_rq2_dual_specification_effects_20260723.png"

SUNING_FILE = (
    "processed_data/suning/Final_Causal_Output/"
    "suning_RQ2_individual_effects_multiplicity_20260723.csv"
)
STAGES = ["click", "cart", "purchase"]
STAGE_LABELS = {"click": "Click", "cart": "Cart", "purchase": "Purchase"}

COMPOSITE_CUES = [
    "T_con_mkt",
    "T_con_soc",
    "T_con_rat",
    "T_int_fac",
    "T_int_vis",
    "T_int_sem",
]
COMPONENT_CUES = [
    "a_pri",
    "a_gft",
    "a_sub",
    "a_urg",
    "T_con_soc",
    "T_con_rat",
    "a_spec",
    "a_str",
    "T_int_fac",
    "T_int_sem",
]
CUE_LABELS = {
    "T_con_mkt": "Marketing salience",
    "T_con_soc": "Social proof",
    "T_con_rat": "Rating valence",
    "T_int_fac": "Factual density",
    "T_int_vis": "Functional visuals",
    "T_int_sem": "Historical category affinity",
    "a_pri": "Price shock",
    "a_gft": "Gift appeal",
    "a_sub": "Subsidy authenticity",
    "a_urg": "Urgency",
    "a_spec": "Specification overlay",
    "a_str": "Internal structure view",
}
COLORS = {
    "T_con_mkt": "#B55D3A",
    "T_con_soc": "#7B3F66",
    "T_con_rat": "#D19A32",
    "T_int_fac": "#277C78",
    "T_int_vis": "#3F6FA3",
    "T_int_sem": "#4D7D43",
    "a_pri": "#A94F36",
    "a_gft": "#D0733B",
    "a_sub": "#C28B2C",
    "a_urg": "#8F3F5F",
    "a_spec": "#2F6F9F",
    "a_str": "#5B80B2",
}
MARKERS = {
    "T_con_mkt": "o",
    "T_con_soc": "s",
    "T_con_rat": "D",
    "T_int_fac": "^",
    "T_int_vis": "v",
    "T_int_sem": "P",
    "a_pri": "o",
    "a_gft": "s",
    "a_sub": "D",
    "a_urg": "X",
    "a_spec": "^",
    "a_str": "v",
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


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_ate(results_root: Path) -> pd.DataFrame:
    path = results_root / SUNING_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing RQ2 result: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if "H_level" in frame:
        frame = frame[frame["H_level"].astype(str).eq("All (ATE)")]
    return frame


def estimable_rows(frame: pd.DataFrame, resolution: str) -> pd.DataFrame:
    rows = frame[frame["Resolution"].eq(resolution)].copy()
    if "Estimable" in rows:
        rows = rows[as_bool(rows["Estimable"])]
    return rows


def plot_suning(
    ax: plt.Axes,
    frame: pd.DataFrame,
    resolution: str,
    cues: list[str],
    panel_label: str,
) -> tuple[list[object], list[str]]:
    rows = estimable_rows(frame, resolution)
    specification = "Composite-level" if resolution == "Aggregate" else "Component-level"
    handles: list[object] = []
    labels: list[str] = []
    for cue in cues:
        cue_rows = rows[rows["Component"].eq(cue)].copy()
        cue_rows["stage_order"] = cue_rows["Stage"].map(
            {stage: index for index, stage in enumerate(STAGES)}
        )
        cue_rows = cue_rows.sort_values("stage_order")
        if cue_rows.empty:
            continue
        handle = ax.errorbar(
            cue_rows["stage_order"],
            cue_rows["Coefficient"],
            yerr=[
                cue_rows["Coefficient"] - cue_rows["CI_Lower_95"],
                cue_rows["CI_Upper_95"] - cue_rows["Coefficient"],
            ],
            marker=MARKERS.get(cue, "o"),
            markersize=4.5,
            linewidth=1.25,
            capsize=2,
            color=COLORS[cue],
            alpha=0.95,
            markerfacecolor="white",
            markeredgewidth=1.1,
        )
        holm_column = next(
            (
                name
                for name in ("Significant_Holm_FWER_05", "Significant_Holm_05")
                if name in cue_rows
            ),
            None,
        )
        if holm_column:
            significant = cue_rows[as_bool(cue_rows[holm_column])]
            if not significant.empty:
                ax.scatter(
                    significant["stage_order"],
                    significant["Coefficient"],
                    marker=MARKERS.get(cue, "o"),
                    s=29,
                    color=COLORS[cue],
                    edgecolors=COLORS[cue],
                    linewidths=0.8,
                    zorder=4,
                )
        if "Significant_BH_FDR_05" in cue_rows:
            bh_significant = as_bool(cue_rows["Significant_BH_FDR_05"])
            holm_significant = (
                as_bool(cue_rows[holm_column])
                if holm_column
                else pd.Series(False, index=cue_rows.index)
            )
            bh_only = cue_rows[bh_significant & ~holm_significant]
            for row in bh_only.itertuples(index=False):
                ax.annotate(
                    r"$\dagger$",
                    (row.stage_order, row.Coefficient),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#111111",
                    zorder=5,
                )
        handles.append(handle)
        labels.append(CUE_LABELS[cue])
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(range(3), [STAGE_LABELS[stage] for stage in STAGES])
    ax.set_ylabel("Stage-specific ATE estimate")
    ax.set_title(
        f"{panel_label}  {specification}",
        loc="left",
        fontweight="bold",
        pad=7,
    )
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    return handles, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    frame = load_ate(results_root)
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 8.8), sharex=True, sharey=True)
    plt.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.90,
        bottom=0.08,
        hspace=0.48,
    )

    composite_handles, composite_labels = plot_suning(
        axes[0], frame, "Aggregate", COMPOSITE_CUES, "(a)"
    )
    component_handles, component_labels = plot_suning(
        axes[1], frame, "Disaggregate", COMPONENT_CUES, "(b)"
    )
    axes[1].set_xlabel("Observed funnel stage")

    fig.legend(
        composite_handles,
        composite_labels,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.975),
        ncol=6,
        fontsize=7.8,
    )
    fig.legend(
        component_handles,
        component_labels,
        loc="center",
        bbox_to_anchor=(0.535, 0.49),
        ncol=5,
        fontsize=7.4,
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
