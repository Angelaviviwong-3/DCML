#!/usr/bin/env python3
"""Plot focal causal-ML omitted-confounding sensitivity values.

The figure uses equal-strength confounding values rather than clipped
benchmark-adjusted ATEs, so ``cf_d = 1`` boundaries cannot distort the scale.
"""

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


VERSION = "20260729"
CAUSAL_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = _dcml_paths.workspace_path(CAUSAL_ROOT, "Unified_Visualization/output")
SOURCE = OUTPUT_DIR / f"Table_DML_OVB_Focal_Summary_{VERSION}.csv"

DATASET_ORDER = ["suning", "amazon_appliances", "amazon_beauty"]


def draw(frame: pd.DataFrame) -> tuple[Path, Path]:
    required = {
        "Dataset_Key",
        "Dataset_Label",
        "Focal_Label",
        "Point_RV_Percent",
        "Approx_CI_RV_Percent_UserCluster",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Focal summary is missing columns: {missing}")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.4, 4.3),
        sharex=True,
        gridspec_kw={"width_ratios": [1.4, 1.2, 1.0], "wspace": 0.55},
    )

    all_values = pd.concat(
        [frame["Point_RV_Percent"], frame["Approx_CI_RV_Percent_UserCluster"]]
    )
    x_max = max(5.0, float(np.nanmax(all_values)) * 1.12)

    for panel_index, (ax, dataset) in enumerate(zip(axes, DATASET_ORDER, strict=True)):
        subset = frame[frame["Dataset_Key"].eq(dataset)].copy().reset_index(drop=True)
        if subset.empty:
            raise RuntimeError(f"No focal sensitivity rows for {dataset}")
        subset = subset.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(subset), dtype=float)
        ax.hlines(
            y,
            subset["Approx_CI_RV_Percent_UserCluster"],
            subset["Point_RV_Percent"],
            color="#A9A9A9",
            linewidth=1.3,
            zorder=1,
        )
        ax.scatter(
            subset["Point_RV_Percent"],
            y,
            color="#245A84",
            marker="o",
            s=34,
            label="Point estimate",
            zorder=3,
        )
        ax.scatter(
            subset["Approx_CI_RV_Percent_UserCluster"],
            y,
            color="#C75B39",
            marker="s",
            s=29,
            label="Approx. 95% CI boundary",
            zorder=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(subset["Focal_Label"])
        label = str(subset["Dataset_Label"].iloc[0])
        ax.set_title(f"{chr(97 + panel_index)}  {label}", loc="left", fontweight="bold")
        ax.set_xlim(0, x_max)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, zorder=0)
        ax.set_xlabel("Equal-strength confounding value (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.14, right=0.98)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"Figure_DML_OVB_Sensitivity_{VERSION}.pdf"
    png_path = OUTPUT_DIR / f"Figure_DML_OVB_Sensitivity_{VERSION}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing {SOURCE}. Run postprocess_dml_ovb_sensitivity.py first."
        )
    for path in draw(pd.read_csv(SOURCE)):
        print(path)


if __name__ == "__main__":
    main()
