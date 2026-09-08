#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustness value table for Amazon purchase-only 20260717 ATEs."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths


import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


CAUSAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_utils import ensure_dir, find_causal_root, setup_logging


ROOT = find_causal_root(__file__)
LOG_DIR = ensure_dir(_dcml_paths.workspace_path(ROOT, 'log'))
CLAIM_BOUNDARY = (
    "Amazon purchase-stage sampled-choice external validation among temporally certified items; not population "
    "checkout conversion, full-funnel reversal, or policy/OPE evidence."
)


def calculate_rv(t_stat: float, dof: float) -> float:
    if not np.isfinite(t_stat) or not np.isfinite(dof) or dof <= 0:
        return np.nan
    f2 = (float(t_stat) ** 2) / float(dof)
    return 0.5 * (np.sqrt(f2**2 + 4 * f2) - f2)


def run() -> None:
    dataset = "amazon_beauty"
    logger = setup_logging(LOG_DIR, f"robustness_value_{dataset}_purchase_20260717")
    start = time.time()
    base_dir = _dcml_paths.workspace_path(ROOT, 'processed_data') / dataset
    save_dir = ensure_dir(base_dir / "Final_Causal_Output")
    ate_path = save_dir / f"{dataset}_Final_ATE_20260717.csv"

    logger.info("=" * 80)
    logger.info("Amazon purchase-only robustness value, dataset=%s", dataset)
    if not ate_path.exists():
        raise FileNotFoundError(f"ATE file not found: {ate_path}")
    df = pd.read_csv(ate_path)
    df = df[(df["H_level"] == "All (ATE)") & (df["Stage"] == "purchase")].copy()
    df["DOF_Approx"] = (pd.to_numeric(df["N"], errors="coerce") - 1).clip(lower=1)
    df["RV"] = [calculate_rv(t, d) for t, d in zip(df["T_Stat"], df["DOF_Approx"])]
    df["RV_Percent"] = df["RV"] * 100
    df["Significant_Raw_05"] = df["P_Value"] < 0.05
    if "P_Value_Holm" in df.columns:
        df["Significant_Holm_05"] = df["P_Value_Holm"] < 0.05
    df["Claim_Boundary"] = CLAIM_BOUNDARY

    keep = [
        "Resolution",
        "Stage",
        "Component",
        "Treatment_Column",
        "Coefficient",
        "Std_Error",
        "T_Stat",
        "P_Value",
        "P_Value_Holm",
        "Significant_Holm_05",
        "Significant_Raw_05",
        "Estimable",
        "Support_Flag",
        "DOF_Approx",
        "RV",
        "RV_Percent",
        "N",
        "Claim_Boundary",
        "Temporal_Analysis_Population",
    ]
    out_path = save_dir / f"{dataset}_Purchase_Robustness_Value_Table_20260717.csv"
    df[[c for c in keep if c in df.columns]].to_csv(out_path, index=False)
    logger.info("Saved RV table: %s", out_path)
    logger.info("Completed in %.2f seconds", time.time() - start)
    logger.info("=" * 80)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
