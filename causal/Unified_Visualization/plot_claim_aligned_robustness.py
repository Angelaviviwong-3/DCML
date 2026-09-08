#!/usr/bin/env python3
"""Visualize claim-aligned robustness, sensitivity, and scope classifications."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


VERSION = "20260729-2"
ROOT = Path(__file__).resolve().parents[1]
SOURCE = _dcml_paths.workspace_path(ROOT, 'processed_data') / "audit" / f"Claim_Aligned_Robustness_Matrix_{VERSION}.csv"
OUTPUT_DIR = _dcml_paths.workspace_path(ROOT, "Unified_Visualization/output")

DATASETS = ["Suning", "Amazon Appliances", "Amazon Beauty"]
ANALYSES = [
    "Out-of-time nuisance transport",
    "Certified-sample specification sensitivity",
    "Gram-Schmidt projection sensitivity",
    "Joint treatment-row placebo",
    "Benchmarked omitted-confounding sensitivity",
    "Temporal certification and observed-selection sensitivity",
    "Recommendation candidate-set sensitivity",
]
LABELS = {
    "Out-of-time nuisance transport": "Nuisance transport",
    "Certified-sample specification sensitivity": "Sample specification",
    "Gram-Schmidt projection sensitivity": "GS implementation",
    "Joint treatment-row placebo": "Joint placebo",
    "Benchmarked omitted-confounding sensitivity": "Omitted confounding",
    "Temporal certification and observed-selection sensitivity": "Certification selection",
    "Recommendation candidate-set sensitivity": "Candidate set",
}
STATUS_ORDER = [
    "PASS",
    "PASS_WITH_LOCAL_EXCEPTIONS",
    "SENSITIVITY_LIMIT",
    "SCOPE_LIMIT",
    "PREDICTIVE_ONLY",
]
STATUS_LABELS = {
    "PASS": "Pass",
    "PASS_WITH_LOCAL_EXCEPTIONS": "Pass*",
    "SENSITIVITY_LIMIT": "Sensitive",
    "SCOPE_LIMIT": "Scope",
    "PREDICTIVE_ONLY": "Prediction",
}
COLORS = ["#2F7D5B", "#8FBF75", "#D28B37", "#C9A45C", "#4F78A8"]


def draw(frame: pd.DataFrame) -> tuple[Path, Path]:
    lookup = {
        (str(row.Analysis), str(row.Dataset)): str(row.Status)
        for row in frame.itertuples(index=False)
    }
    matrix = np.empty((len(ANALYSES), len(DATASETS)), dtype=int)
    for row_index, analysis in enumerate(ANALYSES):
        for column_index, dataset in enumerate(DATASETS):
            status = lookup.get((analysis, dataset))
            if status not in STATUS_ORDER:
                raise RuntimeError(f"Missing or unknown status for {analysis}/{dataset}: {status}")
            matrix[row_index, column_index] = STATUS_ORDER.index(status)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    ax.imshow(matrix, cmap=ListedColormap(COLORS), vmin=-0.5, vmax=4.5, aspect="auto")
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_yticks(np.arange(len(ANALYSES)))
    ax.set_yticklabels([LABELS[value] for value in ANALYSES])
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ANALYSES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_index in range(len(ANALYSES)):
        for column_index in range(len(DATASETS)):
            status = STATUS_ORDER[matrix[row_index, column_index]]
            ax.text(
                column_index,
                row_index,
                STATUS_LABELS[status],
                ha="center",
                va="center",
                color="white" if status != "PASS_WITH_LOCAL_EXCEPTIONS" else "#17391F",
                fontweight="bold",
                fontsize=8.5,
            )
    ax.set_title("Claim-aligned robustness, sensitivity, and scope audit", loc="left", fontweight="bold")
    legend = [
        Patch(facecolor=color, edgecolor="none", label=STATUS_LABELS[status])
        for status, color in zip(STATUS_ORDER, COLORS, strict=True)
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.88, bottom=0.19)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"Figure_Claim_Aligned_Robustness_{VERSION}.pdf"
    png_path = OUTPUT_DIR / f"Figure_Claim_Aligned_Robustness_{VERSION}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing {SOURCE}. Run build_claim_aligned_robustness.py first."
        )
    for path in draw(pd.read_csv(SOURCE)):
        print(path)


if __name__ == "__main__":
    main()
