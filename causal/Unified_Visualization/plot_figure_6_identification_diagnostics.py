#!/usr/bin/env python3
"""RQ1 Figure 6: temporal certification and out-of-time nuisance diagnostics."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization_common import (
    DATASETS,
    DATASET_LABEL,
    DEFAULT_CAUSAL_ROOT,
    STAGE_LABEL,
    configure_style,
    dataset_files,
    output_path,
    read_csv,
    save_pdf,
)


FILENAME = "Figure_6_identification_diagnostics_20260721.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-root", type=Path, default=DEFAULT_CAUSAL_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.causal_root.expanduser().resolve()
    configure_style()

    audit_rows = []
    transport_rows = []
    for dataset in DATASETS:
        files = dataset_files(root, dataset)
        audit = read_csv(files["audit"]).iloc[0]
        audit_rows.append(
            {
                "dataset": dataset,
                "item_rate": float(audit["set_c_item_certification_rate"]),
                "row_rate": float(audit["set_c_row_certification_rate"]),
            }
        )
        diagnostic = read_csv(files["transport"]).copy()
        diagnostic["dataset"] = dataset
        transport_rows.append(diagnostic)
    audit = pd.DataFrame(audit_rows)
    transport = pd.concat(transport_rows, ignore_index=True)
    transport["series"] = transport.apply(
        lambda row: f"{DATASET_LABEL[row['dataset']]} {STAGE_LABEL[str(row['Stage']).lower()]}", axis=1
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), constrained_layout=True)
    x = np.arange(len(audit))
    width = 0.34
    axes[0, 0].bar(x - width / 2, audit["item_rate"] * 100, width, color="#2B6F6D", label="Item rate")
    axes[0, 0].bar(x + width / 2, audit["row_rate"] * 100, width, color="#C8573C", label="Set C row rate")
    axes[0, 0].set_xticks(x, [DATASET_LABEL[d] for d in audit["dataset"]], rotation=12, ha="right")
    axes[0, 0].set_ylabel("Temporally certified (%)")
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].set_title("a  Temporal content certification", loc="left", fontweight="bold")
    axes[0, 0].legend(ncol=2, loc="upper right")

    diagnostics = [
        ("ROC_AUC_Set_C", "Set C ROC AUC", (0.4, 1.0), None),
        ("Brier_Skill_vs_Train_Prevalence_Null", "Brier skill", (-0.05, 0.55), 0.0),
        ("Calibration_Slope", "Calibration slope", (0.0, 2.8), 1.0),
    ]
    for ax, (column, ylabel, ylim, reference), letter in zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]], diagnostics, ["b", "c", "d"]
    ):
        values = pd.to_numeric(transport[column], errors="coerce")
        bars = ax.bar(np.arange(len(transport)), values, color="#557A95")
        ax.set_xticks(np.arange(len(transport)), transport["series"], rotation=22, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(f"{letter}  Out-of-time nuisance diagnostic", loc="left", fontweight="bold")
        if reference is not None:
            ax.axhline(reference, color="#6A6A6A", linewidth=0.9, linestyle=":")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)

    save_pdf(fig, output_path(root, FILENAME, args.output))


if __name__ == "__main__":
    main()
