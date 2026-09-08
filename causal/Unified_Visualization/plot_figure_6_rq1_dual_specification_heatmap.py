#!/usr/bin/env python3
"""RQ1: supported cross-path residual correlations for both cue specifications.

The figure displays only the MNC-by-MAI correlation blocks that the frozen
Gram--Schmidt mapping targets. It compares unprojected and projected Set C
residuals for the composite-level and component-level specifications.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


VERSION = "20260722"

DATASETS = {
    "Suning": {
        "slug": "suning",
        "residual": "suning/DML_Results/DCML_Residuals_20260717.parquet",
        "manifest": "suning/Final_Causal_Output/suning_GS_projection_manifest_20260717.json",
    },
    "Amazon Appliances": {
        "slug": "amazon_appliances",
        "residual": (
            "amazon_appliances/DML_Results/"
            "amazon_appliances_DML_Residuals_Final_20260717.parquet"
        ),
        "manifest": (
            "amazon_appliances/Final_Causal_Output/"
            "amazon_appliances_GS_projection_manifest_20260717.json"
        ),
    },
    "Amazon Beauty": {
        "slug": "amazon_beauty",
        "residual": (
            "amazon_beauty/DML_Results/"
            "amazon_beauty_DML_Residuals_Final_20260717.parquet"
        ),
        "manifest": (
            "amazon_beauty/Final_Causal_Output/"
            "amazon_beauty_GS_projection_manifest_20260717.json"
        ),
    },
}

SPECIFICATIONS = {
    "Aggregate": "Composite-level",
    "Disaggregate": "Component-level",
}

LABELS = {
    "res_T_con_mkt_calib": "Marketing",
    "res_T_con_soc": "Social proof",
    "res_T_con_rat": "Rating",
    "res_T_int_fac_calib": "Factual text",
    "res_T_int_vis_calib": "Functional visual",
    "res_T_int_sem": "Category affinity",
    "res_atomic_a_pri_calib": "Price",
    "res_atomic_a_gft_calib": "Gift",
    "res_atomic_a_sub_calib": "Subsidy",
    "res_atomic_a_urg_calib": "Urgency",
    "res_atomic_a_spec_calib": "Specification",
    "res_atomic_a_str_calib": "Structure",
}


def find_support_file(root: Path, slug: str) -> Path:
    filename = f"{slug}_DML_empirical_positivity_diagnostics_{VERSION}.csv"
    candidates = [_dcml_paths.workspace_path(root, filename), _dcml_paths.workspace_path(root, slug, "audit", filename)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {filename}; checked {candidates}")


def supported_treatments(root: Path, slug: str) -> set[str]:
    frame = pd.read_csv(find_support_file(root, slug))
    passed = frame.loc[
        frame["Empirical_Positivity_Diagnostic"].eq("PASS"), "Treatment"
    ]
    return set(passed.astype(str))


def without_residual_prefix(column: str) -> str:
    return column[4:] if column.startswith("res_") else column


def apply_projection(
    frame: pd.DataFrame,
    manifest: dict,
    controls: list[str],
    targets: list[str],
) -> pd.DataFrame:
    projected = frame.copy()
    for target in targets:
        parameters = manifest["projections"][target]["parameters"]
        fitted = np.full(len(frame), float(parameters.get("const", 0.0)))
        for control in controls:
            fitted += (
                frame[control].to_numpy(dtype=float)
                * float(parameters.get(control, 0.0))
            )
        projected[target] = frame[target].to_numpy(dtype=float) - fitted
    return projected


def cross_path_block(
    frame: pd.DataFrame,
    controls: list[str],
    targets: list[str],
) -> pd.DataFrame:
    correlation = frame[controls + targets].corr()
    block = correlation.loc[controls, targets].copy()
    block.index = [LABELS[column] for column in controls]
    block.columns = [LABELS[column] for column in targets]
    return block


def load_blocks(
    processed_root: Path,
    support_root: Path,
) -> tuple[dict[tuple[str, str, str], pd.DataFrame], pd.DataFrame]:
    blocks: dict[tuple[str, str, str], pd.DataFrame] = {}
    rows: list[dict] = []
    for dataset, config in DATASETS.items():
        supported = supported_treatments(support_root, config["slug"])
        manifest = json.loads((processed_root / config["manifest"]).read_text())
        for manifest_key, specification in SPECIFICATIONS.items():
            section = manifest[manifest_key]
            controls = [
                column
                for column in section["heuristic_controls"]
                if without_residual_prefix(column) in supported
            ]
            targets = [
                column
                for column in section["analytical_targets"]
                if without_residual_prefix(column) in supported
            ]
            columns = controls + targets
            residuals = pd.read_parquet(
                processed_root / config["residual"], columns=columns
            )
            projected = apply_projection(residuals, section, controls, targets)
            for status, values in (
                ("Before projection", residuals),
                ("Frozen projection", projected),
            ):
                block = cross_path_block(values, controls, targets)
                blocks[(dataset, specification, status)] = block
                rows.append(
                    {
                        "Dataset": dataset,
                        "Specification": specification,
                        "Projection_Status": status,
                        "MNC_Cues": len(controls),
                        "MAI_Cues": len(targets),
                        "Max_Absolute_Cross_Path_Correlation": float(
                            np.nanmax(np.abs(block.to_numpy(dtype=float)))
                        ),
                        "Excluded_Treatments": "|".join(
                            sorted(
                                column
                                for column in (
                                    section["heuristic_controls"]
                                    + section["analytical_targets"]
                                )
                                if without_residual_prefix(column) not in supported
                            )
                        ),
                    }
                )
    return blocks, pd.DataFrame(rows)


def draw_heatmap(
    ax: plt.Axes,
    block: pd.DataFrame,
    show_y: bool,
    show_y_label: bool,
    colorbar_ax: plt.Axes | None,
) -> None:
    sns.heatmap(
        block,
        ax=ax,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 10.5},
        linewidths=0.4,
        linecolor="white",
        cbar=colorbar_ax is not None,
        cbar_ax=colorbar_ax,
        xticklabels=True,
        yticklabels=True if show_y else False,
    )
    ax.tick_params(axis="x", labelrotation=35, labelsize=10.5, pad=2)
    ax.tick_params(axis="y", labelrotation=0, labelsize=10.5, pad=3)
    ax.set_xlabel("MAI cues", fontsize=11.5, labelpad=4)
    ax.set_ylabel("MNC cues" if show_y_label else "", fontsize=11.5, labelpad=8)


def plot(
    blocks: dict[tuple[str, str, str], pd.DataFrame],
    diagnostics: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({"font.family": "serif"})
    figure = plt.figure(figsize=(16.2, 11.8))
    grid = figure.add_gridspec(
        3,
        6,
        width_ratios=[1.0, 1.0, 0.22, 1.28, 1.28, 0.05],
        wspace=0.28,
        hspace=0.72,
    )
    heatmap_columns = [0, 1, 3, 4]
    axes = [
        [figure.add_subplot(grid[row, col]) for col in heatmap_columns]
        for row in range(3)
    ]
    colorbar_ax = figure.add_subplot(grid[:, 5])

    column_keys = [
        ("Composite-level", "Before projection"),
        ("Composite-level", "Frozen projection"),
        ("Component-level", "Before projection"),
        ("Component-level", "Frozen projection"),
    ]
    column_titles = [
        "Set C before projection",
        "Frozen Set B projection",
        "Set C before projection",
        "Frozen Set B projection",
    ]

    for row, dataset in enumerate(DATASETS):
        for col, ((specification, status), title) in enumerate(
            zip(column_keys, column_titles)
        ):
            block = blocks[(dataset, specification, status)]
            ax = axes[row][col]
            draw_heatmap(
                ax,
                block,
                show_y=col in (0, 2),
                show_y_label=col == 0,
                colorbar_ax=colorbar_ax if row == 0 and col == 3 else None,
            )
            maximum = float(np.nanmax(np.abs(block.to_numpy(dtype=float))))
            ax.set_title(
                (title + "\n" if row == 0 else "") + f"max $|r|$ = {maximum:.3f}",
                fontsize=12.2,
                pad=7,
            )
        axes[row][0].text(
            -0.66,
            0.5,
            dataset,
            transform=axes[row][0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    figure.text(
        0.29,
        0.985,
        "(a) Composite-level specification",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.72,
        0.985,
        "(b) Component-level specification",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    colorbar_ax.set_ylabel("Pearson correlation, $r$", fontsize=11.5, labelpad=8)
    colorbar_ax.tick_params(labelsize=10.5)
    figure.subplots_adjust(left=0.115, right=0.94, top=0.91, bottom=0.08)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Figure_6_rq1_dual_specification_heatmap_20260722"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(figure)
    diagnostics.to_csv(
        output_dir / "RQ1_dual_specification_cross_path_diagnostics_20260722.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(str(_dcml_paths.project_path('processed_data'))),
    )
    parser.add_argument(
        "--support-root",
        type=Path,
        default=Path(str(_dcml_paths.project_path('processed_data'))),
        help="Processed-data root or flat directory containing positivity summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            str(_dcml_paths.project_path('Unified_Visualization/output'))
        ),
    )
    args = parser.parse_args()
    blocks, diagnostics = load_blocks(args.processed_root, args.support_root)
    plot(blocks, diagnostics, args.output_dir)


if __name__ == "__main__":
    main()
