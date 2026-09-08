#!/usr/bin/env python3
"""RQ3 Figure 9: CATE differences across behavioral-history levels."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CAUSAL_ROOT = (
    _dcml_paths.workspace_path(PROJECT_ROOT, "20260719_results/xzhe162/wh_workspace/casual/causal")
)
SERVER_CAUSAL_ROOT = Path(str(_dcml_paths.project_path('')))
DEFAULT_OUTPUT_DIR = _dcml_paths.project_path("Unified_Visualization/output")
PDF_NAME = "Figure_9_cate_heterogeneity_20260723.pdf"
PNG_NAME = "Figure_9_cate_heterogeneity_20260723.png"

DATASETS = ("suning", "amazon_appliances", "amazon_beauty")
DATASET_LABEL = {
    "suning": "Suning",
    "amazon_appliances": "Amazon Appliances",
    "amazon_beauty": "Amazon Beauty",
}
RESOLUTIONS = ("Aggregate", "Disaggregate")
RESOLUTION_LABEL = {
    "Aggregate": "Composite-level cue specification",
    "Disaggregate": "Component-level cue specification",
}
STAGE_ORDER = ("click", "cart", "purchase")
STAGE_LABEL = {"click": "Click", "cart": "Cart", "purchase": "Purchase"}
COMPONENT_ORDER = (
    "T_con_mkt",
    "a_pri",
    "a_gft",
    "a_sub",
    "a_urg",
    "T_con_soc",
    "T_con_rat",
    "T_int_fac",
    "T_int_vis",
    "a_spec",
    "a_str",
    "T_int_sem",
)
COMPONENT_LABEL = {
    "T_con_mkt": "Marketing salience",
    "a_pri": "Price shock",
    "a_gft": "Gift appeal",
    "a_sub": "Subsidy authenticity",
    "a_urg": "Urgency",
    "T_con_soc": "Social proof",
    "T_con_rat": "Rating valence",
    "T_int_fac": "Factual density",
    "T_int_vis": "Functional visuals",
    "a_spec": "Specification overlay",
    "a_str": "Internal structure view",
    "T_int_sem": "Historical category affinity",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def default_causal_root() -> Path:
    return LOCAL_CAUSAL_ROOT if LOCAL_CAUSAL_ROOT.is_dir() else SERVER_CAUSAL_ROOT


def result_paths(causal_root: Path, dataset: str) -> tuple[Path, Path]:
    final = _dcml_paths.workspace_path(causal_root, 'processed_data') / dataset / "Final_Causal_Output"
    return (
        final / f"{dataset}_H3_omnibus_tests_20260719.csv",
        final / f"{dataset}_H3_interaction_contrasts_20260719.csv",
    )


def parse_levels(value: object) -> list[int]:
    levels = []
    for token in str(value).split("|"):
        token = token.strip()
        if token.startswith("H") and token[1:].isdigit():
            levels.append(int(token[1:]))
    return sorted(set(levels))


def load_selected_contrasts(causal_root: Path) -> pd.DataFrame:
    selected = []
    for dataset in DATASETS:
        omnibus_path, contrast_path = result_paths(causal_root, dataset)
        if not omnibus_path.is_file() or not contrast_path.is_file():
            raise FileNotFoundError(
                f"Missing RQ3 result for {dataset}: {omnibus_path} or {contrast_path}"
            )
        omnibus = pd.read_csv(omnibus_path, low_memory=False)
        contrasts = pd.read_csv(contrast_path, low_memory=False)
        omnibus = omnibus[as_bool(omnibus["Significant_Holm_05"])].copy()
        if "Omnibus_Valid" in omnibus:
            omnibus = omnibus[as_bool(omnibus["Omnibus_Valid"])]
        if "Contrast_Valid" in contrasts:
            contrasts = contrasts[as_bool(contrasts["Contrast_Valid"])]

        for row in omnibus.itertuples(index=False):
            levels = parse_levels(row.Estimable_H_Levels)
            if len(levels) < 2:
                continue
            reference = int(row.Reference_H_Level)
            highest = max(levels)
            contrast_name = f"H{highest}-H{reference}"
            match = contrasts[
                contrasts["Resolution"].eq(row.Resolution)
                & contrasts["Stage"].eq(row.Stage)
                & contrasts["Component"].eq(row.Component)
                & contrasts["Contrast"].eq(contrast_name)
            ]
            if match.empty:
                raise RuntimeError(
                    f"Missing prespecified contrast {dataset} {row.Resolution} "
                    f"{row.Stage} {row.Component} {contrast_name}"
                )
            value = match.iloc[0]
            selected.append(
                {
                    "Dataset": dataset,
                    "Resolution": row.Resolution,
                    "Stage": row.Stage,
                    "Component": row.Component,
                    "Contrast": contrast_name,
                    "Difference": float(value["Difference"]),
                    "Significant_Holm_05": bool(
                        as_bool(pd.Series([value["Significant_Holm_05"]])).iloc[0]
                    ),
                }
            )
    return pd.DataFrame(selected)


def ordered(values: set[str], reference: tuple[str, ...]) -> list[str]:
    return [value for value in reference if value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-root", type=Path, default=default_causal_root())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    causal_root = args.causal_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    data = load_selected_contrasts(causal_root)
    limit = max(0.01, float(data["Difference"].abs().max()))

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(10.8, 9.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.25, 0.85, 0.85]},
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#ECECEC")
    image = None

    for row_index, dataset in enumerate(DATASETS):
        for column_index, resolution in enumerate(RESOLUTIONS):
            ax = axes[row_index, column_index]
            subset = data[
                data["Dataset"].eq(dataset) & data["Resolution"].eq(resolution)
            ].copy()
            components = ordered(set(subset["Component"]), COMPONENT_ORDER)
            stages = ordered(set(subset["Stage"]), STAGE_ORDER)
            matrix = subset.pivot_table(
                index="Component",
                columns="Stage",
                values="Difference",
                aggfunc="first",
            ).reindex(index=components, columns=stages)
            image = ax.imshow(
                np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
                cmap=cmap,
                vmin=-limit,
                vmax=limit,
                aspect="auto",
            )
            ax.set_yticks(
                np.arange(len(components)),
                [COMPONENT_LABEL.get(component, component) for component in components],
            )
            ax.set_xticks(
                np.arange(len(stages)),
                [STAGE_LABEL[stage] for stage in stages],
            )
            if row_index == 0:
                panel_letter = "a" if column_index == 0 else "b"
                title = (
                    f"{panel_letter}  {RESOLUTION_LABEL[resolution]}\n"
                    f"{DATASET_LABEL[dataset]}"
                )
            else:
                title = DATASET_LABEL[dataset]
            ax.set_title(title, loc="left", fontweight="bold")
            for y_index, component in enumerate(components):
                for x_index, stage in enumerate(stages):
                    cell = subset[
                        subset["Component"].eq(component)
                        & subset["Stage"].eq(stage)
                    ]
                    if cell.empty:
                        continue
                    value = float(cell.iloc[0]["Difference"])
                    contrast = str(cell.iloc[0]["Contrast"])
                    star = "*" if bool(cell.iloc[0]["Significant_Holm_05"]) else ""
                    text_color = "white" if abs(value) > limit * 0.55 else "#222222"
                    ax.text(
                        x_index,
                        y_index,
                        f"{value:.3f}{star}\n{contrast}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=text_color,
                    )

    if image is None:
        raise RuntimeError("No valid RQ3 CATE contrasts were available.")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("CATE difference: highest minus reference history level")
    fig.suptitle(
        "Behavioral-history heterogeneity in stage-specific CATEs",
        fontsize=10.5,
    )

    pdf_path = output_dir / PDF_NAME
    png_path = output_dir / PNG_NAME
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, facecolor="white")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
