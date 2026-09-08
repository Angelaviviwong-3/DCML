#!/usr/bin/env python3
"""RQ2 Figure 8: Suning outcome changes for an interquartile-range increase."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from visualization_common import (
    COMPONENT_LABEL,
    DEFAULT_CAUSAL_ROOT,
    STAGE_LABEL,
    STAGE_ORDER,
    configure_style,
    output_path,
    read_csv,
    save_pdf,
)


FILENAME = "Figure_8_suning_iqr_scaled_effects_20260721.pdf"
PRIMARY_ROLE = "primary_history_eligible_aggregate_6"
COMPONENTS = ["T_con_soc", "T_int_sem"]
COLORS = {"T_con_soc": "#7A3E65", "T_int_sem": "#4F7B45"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-root", type=Path, default=DEFAULT_CAUSAL_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.causal_root.expanduser().resolve()
    configure_style()

    source = (
        _dcml_paths.workspace_path(root, 'processed_data/suning/Final_Causal_Output/suning_IQR_Scaled_Effect_Profiles_20260720.csv')
    )
    data = read_csv(source)
    subset = data[data["Specification_Role"].eq(PRIMARY_ROLE)].copy()
    if subset.empty:
        raise RuntimeError(f"No IQR-effect rows for {PRIMARY_ROLE} in {source}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for component in COMPONENTS:
        rows = subset[subset["Component"].eq(component)].copy()
        rows["order"] = rows["Stage"].map({stage: i for i, stage in enumerate(STAGE_ORDER)})
        rows = rows.sort_values("order")
        if rows.empty:
            raise RuntimeError(f"No {component} rows for {PRIMARY_ROLE} in {source}")
        ax.errorbar(
            rows["order"],
            rows["Outcome_Change_Percentage_Points"],
            yerr=[
                rows["Outcome_Change_Percentage_Points"] - rows["Outcome_Change_CI_Lower_95_pp"],
                rows["Outcome_Change_CI_Upper_95_pp"] - rows["Outcome_Change_Percentage_Points"],
            ],
            marker="o",
            linewidth=1.7,
            capsize=3,
            color=COLORS[component],
            label=COMPONENT_LABEL[component],
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(range(3), [STAGE_LABEL[stage] for stage in STAGE_ORDER])
    ax.set_ylabel("Outcome change for an IQR increase (percentage points)")
    ax.set_title("Suning strict-reversal effects: composite-level specification")
    ax.legend(fontsize=8)
    save_pdf(fig, output_path(root, FILENAME, args.output))


if __name__ == "__main__":
    main()
