#!/usr/bin/env python3
"""RQ1: train-fitted Gram-Schmidt cross-path correlation diagnostics."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATASETS = {
    "Suning": {
        "directory": "suning",
        "train": "DML_Results/DCML_GS_Training_Residuals_20260717.parquet",
        "test": "DML_Results/DCML_Residuals_20260717.parquet",
        "manifest": "Final_Causal_Output/suning_GS_projection_manifest_20260717.json",
    },
    "Amazon Appliances": {
        "directory": "amazon_appliances",
        "train": "DML_Results/amazon_appliances_DML_GS_Training_Residuals_20260717.parquet",
        "test": "DML_Results/amazon_appliances_DML_Residuals_Final_20260717.parquet",
        "manifest": "Final_Causal_Output/amazon_appliances_GS_projection_manifest_20260717.json",
    },
    "Amazon Beauty": {
        "directory": "amazon_beauty",
        "train": "DML_Results/amazon_beauty_DML_GS_Training_Residuals_20260717.parquet",
        "test": "DML_Results/amazon_beauty_DML_Residuals_Final_20260717.parquet",
        "manifest": "Final_Causal_Output/amazon_beauty_GS_projection_manifest_20260717.json",
    },
}

LABELS = [
    r"$T_{con,mkt}$",
    r"$T_{con,soc}$",
    r"$T_{con,rat}$",
    r"$T_{int,fac}$",
    r"$T_{int,vis}$",
    r"$T_{int,sem}$",
]


def project(frame: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    result = frame.copy()
    for target in manifest["analytical_targets"]:
        params = manifest["projections"][target]["parameters"]
        fitted = np.full(len(frame), float(params.get("const", 0.0)))
        for control in manifest["heuristic_controls"]:
            fitted += frame[control].to_numpy(dtype=float) * float(params.get(control, 0.0))
        result[target] = frame[target].to_numpy(dtype=float) - fitted
    return result


def cross_path_max(correlation: pd.DataFrame) -> float:
    return float(np.nanmax(np.abs(correlation.iloc[:3, 3:].to_numpy(dtype=float))))


def add_cross_path_box(ax: plt.Axes) -> None:
    ax.add_patch(patches.Rectangle((3, 0), 3, 3, fill=False, edgecolor="#B22222", linewidth=1.8))
    ax.add_patch(patches.Rectangle((0, 3), 3, 3, fill=False, edgecolor="#B22222", linewidth=1.8))


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

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({"font.family": "serif"})
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 12.2))
    diagnostics: list[dict] = []
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    for row, (dataset, config) in enumerate(DATASETS.items()):
        base = args.processed_root / config["directory"]
        manifest = json.loads((base / config["manifest"]).read_text())["Aggregate"]
        columns = manifest["heuristic_controls"] + manifest["analytical_targets"]
        train = pd.read_parquet(base / config["train"], columns=columns)
        test = pd.read_parquet(base / config["test"], columns=columns)
        matrices = [
            ("Set B before projection", train.corr()),
            ("Set B after projection", project(train, manifest).corr()),
            ("Set C frozen projection", project(test, manifest).corr()),
        ]
        for col, (role, corr) in enumerate(matrices):
            maximum = cross_path_max(corr)
            diagnostics.append(
                {
                    "Dataset": dataset,
                    "Diagnostic_Sample": role,
                    "Max_Absolute_Cross_Path_Correlation": maximum,
                    "Marketing_Factual_Correlation": float(corr.iloc[0, 3]),
                    "N": len(train) if col < 2 else len(test),
                }
            )
            ax = axes[row, col]
            sns.heatmap(
                corr,
                ax=ax,
                cmap=cmap,
                vmin=-1,
                vmax=1,
                center=0,
                annot=True,
                fmt=".2f",
                cbar=(col == 2),
                square=True,
                linewidths=0.35,
                xticklabels=LABELS,
                yticklabels=LABELS if col == 0 else False,
            )
            add_cross_path_box(ax)
            ax.set_title(f"{dataset}\n{role}\nmax cross-path |r| = {maximum:.3f}", fontsize=10)
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)

    for col, letter in enumerate(("a", "b", "c")):
        axes[0, col].text(-0.15, 1.18, letter, transform=axes[0, col].transAxes, fontweight="bold", fontsize=12)
    fig.tight_layout()
    pdf = args.output_dir / "Figure_7_rq1_gs_orthogonalization_20260722.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(diagnostics).to_csv(
        args.output_dir / "RQ1_gs_correlation_diagnostics_20260722.csv", index=False
    )
    print(pd.DataFrame(diagnostics).to_string(index=False))


if __name__ == "__main__":
    main()
