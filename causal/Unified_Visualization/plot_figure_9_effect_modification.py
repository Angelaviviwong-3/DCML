#!/usr/bin/env python3
"""RQ3 Figure 9: pooled treatment-by-history interaction contrasts for H3."""

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
    AGGREGATE,
    COMPONENT_LABEL,
    DATASETS,
    DATASET_LABEL,
    DEFAULT_CAUSAL_ROOT,
    STAGE_ORDER,
    as_bool,
    configure_style,
    dataset_files,
    output_path,
    read_csv,
    save_pdf,
)


FILENAME = "Figure_9_effect_modification_20260721.pdf"


def ordered_columns(frame: pd.DataFrame) -> list[str]:
    contrast_order = {"H1-H0": 0, "H2-H0": 1, "H3-H0": 2}
    pairs = frame[["Stage", "Contrast"]].drop_duplicates()
    pairs["stage_order"] = pairs["Stage"].str.lower().map({stage: i for i, stage in enumerate(STAGE_ORDER)})
    pairs["contrast_order"] = pairs["Contrast"].map(contrast_order).fillna(99)
    pairs = pairs.sort_values(["stage_order", "contrast_order", "Contrast"])
    return [f"{stage.title()}\n{contrast}" for stage, contrast in pairs[["Stage", "Contrast"]].itertuples(index=False)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-root", type=Path, default=DEFAULT_CAUSAL_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.causal_root.expanduser().resolve()
    configure_style()

    frames = []
    for dataset in DATASETS:
        data = read_csv(dataset_files(root, dataset)["h3"])
        data = data[data["Resolution"].eq("Aggregate") & data["Component"].isin(AGGREGATE)].copy()
        data = data[data["Stage"].notna() & data["Contrast"].notna()]
        if "Contrast_Valid" in data:
            data = data[as_bool(data["Contrast_Valid"])]
        data["Difference"] = pd.to_numeric(data["Difference"], errors="coerce")
        data["column"] = data["Stage"].str.title() + "\n" + data["Contrast"]
        data["dataset"] = dataset
        frames.append(data)

    all_data = pd.concat(frames, ignore_index=True)
    finite = all_data["Difference"].dropna().abs()
    if finite.empty:
        raise RuntimeError("No valid Aggregate H3 contrasts were found.")
    limit = max(0.01, float(finite.quantile(0.98)))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E6E6E6")

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 8.2), constrained_layout=True)
    image = None
    for ax, dataset, letter in zip(axes, DATASETS, ("a", "b", "c")):
        data = all_data[all_data["dataset"].eq(dataset)].copy()
        columns = ordered_columns(data)
        matrix = data.pivot_table(index="Component", columns="column", values="Difference", aggfunc="first")
        matrix = matrix.reindex(index=AGGREGATE, columns=columns)
        image = ax.imshow(
            np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        ax.set_yticks(np.arange(len(AGGREGATE)), [COMPONENT_LABEL[component] for component in AGGREGATE])
        ax.set_xticks(np.arange(len(columns)), columns, fontsize=7)
        ax.set_title(f"{letter}  {DATASET_LABEL[dataset]}", loc="left", fontweight="bold")
        for row_index, component in enumerate(AGGREGATE):
            for column_index, column in enumerate(columns):
                subset = data[data["Component"].eq(component) & data["column"].eq(column)]
                if not subset.empty and as_bool(subset["Significant_Holm_05"]).iloc[0]:
                    ax.text(column_index, row_index, "*", ha="center", va="center", fontweight="bold")
                elif np.isnan(matrix.loc[component, column]):
                    ax.text(column_index, row_index, "N/A", ha="center", va="center", color="#666666", fontsize=7)

    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.80, pad=0.02)
    colorbar.set_label("Treatment slope difference")
    fig.suptitle("Pooled treatment by behavioral-history interaction contrasts\n* Holm-adjusted p < 0.05", fontsize=11)
    save_pdf(fig, output_path(root, FILENAME, args.output))


if __name__ == "__main__":
    main()
