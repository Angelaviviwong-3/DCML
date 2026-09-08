#!/usr/bin/env python3
"""RQ2 Figure 7: Suning stage effects and Amazon Purchase-only effects."""

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
    DATASET_LABEL,
    DEFAULT_CAUSAL_ROOT,
    STAGE_LABEL,
    STAGE_ORDER,
    configure_style,
    dataset_files,
    output_path,
    primary_aggregate,
    read_csv,
    save_pdf,
)


FILENAME = "Figure_7_stage_specific_effects_20260721.pdf"
COLORS = {
    "T_con_mkt": "#B85C38",
    "T_con_soc": "#7A3E65",
    "T_con_rat": "#D49A3A",
    "T_int_fac": "#2C7A7B",
    "T_int_vis": "#3D6D9C",
    "T_int_sem": "#4F7B45",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-root", type=Path, default=DEFAULT_CAUSAL_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.causal_root.expanduser().resolve()
    configure_style()

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), constrained_layout=True)
    suning = primary_aggregate(read_csv(dataset_files(root, "suning")["ate"]))
    for component in AGGREGATE:
        rows = suning[suning["Component"].eq(component)].copy()
        rows["order"] = rows["Stage"].map({stage: index for index, stage in enumerate(STAGE_ORDER)})
        rows = rows.sort_values("order")
        axes[0].errorbar(
            rows["order"],
            rows["Coefficient"],
            yerr=[rows["Coefficient"] - rows["CI_Lower_95"], rows["CI_Upper_95"] - rows["Coefficient"]],
            marker="o",
            linewidth=1.4,
            capsize=2,
            color=COLORS[component],
            label=COMPONENT_LABEL[component],
        )
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_xticks(range(3), [STAGE_LABEL[stage] for stage in STAGE_ORDER])
    axes[0].set_ylabel("Residualized outcome coefficient")
    axes[0].set_title("a  Suning observed funnel", loc="left", fontweight="bold")
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")

    for ax, dataset, letter in zip(axes[1:], ("amazon_appliances", "amazon_beauty"), ("b", "c")):
        data = primary_aggregate(read_csv(dataset_files(root, dataset)["ate"]))
        data = data[data["Stage"].str.lower().eq("purchase")].set_index("Component").reindex(AGGREGATE)
        estimable = data["Coefficient"].notna()
        y = np.arange(len(AGGREGATE))
        ax.errorbar(
            data.loc[estimable, "Coefficient"],
            y[estimable.to_numpy()],
            xerr=[
                data.loc[estimable, "Coefficient"] - data.loc[estimable, "CI_Lower_95"],
                data.loc[estimable, "CI_Upper_95"] - data.loc[estimable, "Coefficient"],
            ],
            fmt="o",
            capsize=2,
            color="#2B6F6D",
        )
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_yticks(y, [COMPONENT_LABEL[component] for component in AGGREGATE])
        ax.invert_yaxis()
        ax.set_xlabel("Residualized outcome coefficient")
        ax.set_title(f"{letter}  {DATASET_LABEL[dataset]} Purchase", loc="left", fontweight="bold")
        for component, row in data.iterrows():
            if pd.isna(row.get("Coefficient")):
                ax.text(
                    0.04,
                    AGGREGATE.index(component),
                    "N/A",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    ha="left",
                    color="#777777",
                )

    save_pdf(fig, output_path(root, FILENAME, args.output))


if __name__ == "__main__":
    main()
