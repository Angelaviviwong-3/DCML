#!/usr/bin/env python3
"""RQ4 Figure 10: five-run predictive recommendation performance."""

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
    DATASET_LABEL,
    DEFAULT_CAUSAL_ROOT,
    STAGE_LABEL,
    configure_style,
    output_path,
    read_csv,
    save_pdf,
)


FILENAME = "Figure_10_prediction_performance_20260721.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-root", type=Path, default=DEFAULT_CAUSAL_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.causal_root.expanduser().resolve()
    configure_style()

    source = _dcml_paths.workspace_path(root, 'results_of_comparison/Recommendation_Training_Seed_Summary_All_Datasets_20260720.csv')
    data = read_csv(source)
    upper = pd.to_numeric(data["NDCG@10_Mean"], errors="coerce") + pd.to_numeric(
        data["NDCG@10_Sample_SD"], errors="coerce"
    )
    common_upper = max(0.05, float(np.ceil(upper.max() * 20) / 20))
    panels = [
        ("suning", "click"),
        ("suning", "cart"),
        ("suning", "purchase"),
        ("amazon_appliances", "purchase"),
        ("amazon_beauty", "purchase"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    for ax, (dataset, stage), letter in zip(axes.flat, panels, ("a", "b", "c", "d", "e")):
        subset = data[data["Dataset"].eq(dataset) & data["Stage"].str.lower().eq(stage)].copy()
        if subset.empty:
            raise RuntimeError(f"No prediction rows for {dataset}/{stage} in {source}")
        subset["NDCG@10_Mean"] = pd.to_numeric(subset["NDCG@10_Mean"], errors="coerce")
        subset["NDCG@10_Sample_SD"] = pd.to_numeric(subset["NDCG@10_Sample_SD"], errors="coerce")
        subset = subset.dropna(subset=["NDCG@10_Mean", "NDCG@10_Sample_SD"]).sort_values(
            "NDCG@10_Mean", ascending=True
        )
        colors = ["#C8573C" if model == "DCML (Ours)" else "#8A98A5" for model in subset["Model"]]
        ax.barh(
            subset["Model"],
            subset["NDCG@10_Mean"],
            xerr=subset["NDCG@10_Sample_SD"],
            color=colors,
            capsize=2,
        )
        ax.set_xlabel(r"NDCG@10, mean $\pm$ sample SD")
        ax.set_title(f"{letter}  {DATASET_LABEL[dataset]} {STAGE_LABEL[stage]}", loc="left", fontweight="bold")
        ax.set_xlim(0, common_upper)
    axes.flat[-1].axis("off")
    fig.suptitle("Five independent training runs", fontsize=11)
    save_pdf(fig, output_path(root, FILENAME, args.output))


if __name__ == "__main__":
    main()
