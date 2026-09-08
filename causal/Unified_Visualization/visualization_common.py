#!/usr/bin/env python3
"""Shared constants and readers for the independent 20260721 result figures."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


VERSION = "20260721"
DEFAULT_CAUSAL_ROOT = Path(str(_dcml_paths.project_path('')))
DATASETS = ("suning", "amazon_appliances", "amazon_beauty")
DATASET_LABEL = {
    "suning": "Suning",
    "amazon_appliances": "Amazon Appliances",
    "amazon_beauty": "Amazon Beauty",
}
COMPONENT_LABEL = {
    "T_con_mkt": "Marketing salience",
    "T_con_soc": "Social proof",
    "T_con_rat": "Rating valence",
    "T_int_fac": "Factual density",
    "T_int_vis": "Functional visuals",
    "T_int_sem": "Historical category affinity",
}
AGGREGATE = [
    "T_con_mkt",
    "T_con_soc",
    "T_con_rat",
    "T_int_fac",
    "T_int_vis",
    "T_int_sem",
]
STAGE_ORDER = ["click", "cart", "purchase"]
STAGE_LABEL = {"click": "Click", "cart": "Cart", "purchase": "Purchase"}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen result is missing: {path}")
    return pd.read_csv(path, low_memory=False)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def dataset_files(causal_root: Path, dataset: str) -> dict[str, Path]:
    processed = _dcml_paths.workspace_path(causal_root, 'processed_data')
    final = processed / dataset / "Final_Causal_Output"
    dml = processed / dataset / "DML_Results"
    if dataset == "suning":
        ate = final / "Final_ATE_Results_20260717.csv"
        gs = final / "suning_GS_order_sensitivity_20260717.csv"
    else:
        ate = final / f"{dataset}_Final_ATE_20260717.csv"
        gs = final / f"{dataset}_GS_order_sensitivity_20260717.csv"
    return {
        "audit": processed / dataset / "audit/mcre_temporal_snapshot_summary_20260717.csv",
        "ate": ate,
        "h3": final / f"{dataset}_H3_interaction_contrasts_20260719.csv",
        "gs": gs,
        "transport": dml / f"{dataset}_nuisance_temporal_transport_diagnostics_20260719.csv",
    }


def output_path(causal_root: Path, filename: str, requested: Path | None) -> Path:
    target = requested.expanduser().resolve() if requested else _dcml_paths.workspace_path(causal_root, 'Unified_Visualization/output') / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_pdf(fig: plt.Figure, target: Path) -> None:
    fig.savefig(target, facecolor="white")
    plt.close(fig)
    print(f"Wrote {target}")


def primary_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[frame["Resolution"].eq("Aggregate")].copy()
    if "H_level" in output:
        output = output[output["H_level"].astype(str).eq("All (ATE)")]
    if "Estimable" in output:
        output = output[as_bool(output["Estimable"])]
    return output[output["Component"].isin(AGGREGATE)]
