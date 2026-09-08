#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustness value table for 20260717 Suning ATEs."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths


import sys
from pathlib import Path

import numpy as np
import pandas as pd


CAUSAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_utils import ensure_dir, find_causal_root, setup_logging


ROOT = find_causal_root(__file__)
BASE_DIR = _dcml_paths.workspace_path(ROOT, 'processed_data') / "suning"
ATE_PATH = BASE_DIR / "Final_Causal_Output" / "Final_ATE_Results_20260717.csv"
SAVE_DIR = ensure_dir(BASE_DIR / "Final_Causal_Output")
LOG_DIR = ensure_dir(_dcml_paths.workspace_path(ROOT, 'log'))
logger = setup_logging(LOG_DIR, "robustness_value_suning_20260717")


def calculate_rv(t_stat: float, dof: float) -> float:
    if not np.isfinite(t_stat) or not np.isfinite(dof) or dof <= 0:
        return np.nan
    f2 = (t_stat**2) / dof
    return 0.5 * (np.sqrt(f2**2 + 4 * f2) - f2)


def main() -> None:
    logger.info("=" * 80)
    logger.info("Robustness value for Suning ATEs, 20260717")
    df = pd.read_csv(ATE_PATH)
    out = df.copy()
    out["DOF_Approx"] = (pd.to_numeric(out["N"], errors="coerce") - 1).clip(lower=1)
    out["RV"] = [calculate_rv(t, d) for t, d in zip(out["T_Stat"], out["DOF_Approx"])]
    out["RV_Percent"] = out["RV"] * 100
    out["Significant_Raw_05"] = out["P_Value"] < 0.05
    if "P_Value_Holm" in out.columns:
        out["Significant_Holm_05"] = out["P_Value_Holm"] < 0.05
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
        "Estimable",
        "Support_Flag",
        "N",
        "DOF_Approx",
        "RV",
        "RV_Percent",
        "Temporal_Analysis_Population",
    ]
    keep = [c for c in keep if c in out.columns]
    out[keep].to_csv(SAVE_DIR / "RQ5_Robustness_Value_Table_20260717.csv", index=False)
    logger.info("Saved RV table: %s", SAVE_DIR / "RQ5_Robustness_Value_Table_20260717.csv")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
