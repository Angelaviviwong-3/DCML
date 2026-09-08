#!/usr/bin/env python3
"""RQ5 Figure 11: coefficient sensitivity to Gram-Schmidt specification."""

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

from visualization_common import (
    AGGREGATE,
    DATASETS,
    DATASET_LABEL,
    DEFAULT_CAUSAL_ROOT,
    configure_style,
    dataset_files,
    output_path,
    read_csv,
    save_pdf,
)


FILENAME = "Figure_11_gs_sensitivity_20260721.pdf"
VARIANT_LABEL = {
    "no_gs": "No GS",
    "reverse_analytical_to_heuristic": "Reverse order",
}
MARKERS = {"no_gs": "o", "reverse_analytical_to_heuristic": "s"}
COLORS = {"no_gs": "#2B6F6D", "reverse_analytical_to_heuristic": "#C8573C"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-root", type=Path, default=DEFAULT_CAUSAL_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.causal_root.expanduser().resolve()
    configure_style()

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), constrained_layout=True)
    for ax, dataset, letter in zip(axes, DATASETS, ("a", "b", "c")):
        data = read_csv(dataset_files(root, dataset)["gs"])
        data = data[data["Resolution"].eq("Aggregate") & data["Component"].isin(AGGREGATE)].copy()
        if "H_level" in data:
            data = data[data["H_level"].astype(str).eq("All (ATE)")]
        data["Coefficient"] = pd.to_numeric(data["Coefficient"], errors="coerce")
        keys = ["Stage", "Component"]
        standard = data[data["GS_Variant"].eq("standard_heuristic_to_analytical")][keys + ["Coefficient"]]
        standard = standard.rename(columns={"Coefficient": "standard"})
        plotted = []
        for variant, label in VARIANT_LABEL.items():
            alternative = data[data["GS_Variant"].eq(variant)][keys + ["Coefficient"]]
            alternative = alternative.rename(columns={"Coefficient": "alternative"})
            merged = standard.merge(alternative, on=keys, how="inner").dropna()
            if merged.empty:
                continue
            plotted.append(merged[["standard", "alternative"]])
            ax.scatter(
                merged["standard"],
                merged["alternative"],
                s=28,
                alpha=0.80,
                marker=MARKERS[variant],
                color=COLORS[variant],
                label=label,
            )
        if not plotted:
            raise RuntimeError(f"No comparable Aggregate GS coefficients for {dataset}")
        combined = pd.concat(plotted, ignore_index=True)
        lower = float(combined.min().min())
        upper = float(combined.max().max())
        pad = max((upper - lower) * 0.08, 0.002)
        bounds = [lower - pad, upper + pad]
        ax.plot(bounds, bounds, color="#666666", linewidth=0.8, linestyle=":")
        ax.set_xlim(*bounds)
        ax.set_ylim(*bounds)
        ax.set_xlabel("Standard GS coefficient")
        ax.set_ylabel("Sensitivity coefficient")
        ax.set_title(f"{letter}  {DATASET_LABEL[dataset]}", loc="left", fontweight="bold")
        ax.legend(fontsize=7)
    save_pdf(fig, output_path(root, FILENAME, args.output))


if __name__ == "__main__":
    main()
