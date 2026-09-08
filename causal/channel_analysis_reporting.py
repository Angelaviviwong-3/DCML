#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Correct finite-bootstrap reporting and claim labels for channel analyses."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[0]))
import project_paths as _dcml_paths


import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dcml_utils import add_holm_by_family, ensure_dir, find_causal_root


OUTPUT_VERSION = "20260719"
SOURCE_VERSION = "20260717"
DATASETS = ("suning", "amazon_appliances", "amazon_beauty")


def _source_path(root: Path, dataset: str) -> Path:
    output = _dcml_paths.workspace_path(root, 'processed_data') / dataset / "Final_Causal_Output"
    if dataset == "suning":
        return output / f"Final_Mediation_Results_{SOURCE_VERSION}.csv"
    return output / f"{dataset}_Final_Mediation_Results_{SOURCE_VERSION}.csv"


def _output_path(root: Path, dataset: str) -> Path:
    output = ensure_dir(_dcml_paths.workspace_path(root, 'processed_data') / dataset / "Final_Causal_Output")
    if dataset == "suning":
        return output / f"Final_Channel_Analysis_Reporting_{OUTPUT_VERSION}.csv"
    return output / f"{dataset}_Final_Channel_Analysis_Reporting_{OUTPUT_VERSION}.csv"


def run_dataset(root: Path, dataset: str, bootstrap_reps: int) -> None:
    source = _source_path(root, dataset)
    if not source.is_file():
        raise FileNotFoundError(f"Missing frozen channel-analysis result: {source}")
    frame = pd.read_csv(source)
    original = pd.to_numeric(frame["P_Value"], errors="coerce")
    inferred_tail_count = np.rint(original * bootstrap_reps / 2.0)
    corrected = np.minimum(1.0, 2.0 * (inferred_tail_count + 1.0) / (bootstrap_reps + 1.0))
    corrected[original.isna()] = np.nan
    frame = frame.rename(
        columns={
            "P_Value": "P_Value_Original_Zero_Allowed",
            "P_Value_Holm": "P_Value_Holm_Original_Zero_Allowed",
        }
    )
    frame["P_Value"] = corrected
    frame = add_holm_by_family(frame, ["Funnel_Stage"], p_col="P_Value", out_col="P_Value_Holm")
    frame["Bootstrap_Reps"] = int(bootstrap_reps)
    frame["P_Value_Method"] = "two-sided bootstrap sign test with plus-one finite-simulation correction"
    frame["P_Value_Minimum_Attainable"] = min(1.0, 2.0 / (bootstrap_reps + 1.0))
    frame["Analysis_Label"] = "exploratory_pre_exposure_channel_analysis"
    frame["Formal_Causal_Mediation_Claim"] = False
    frame["Mediator_Temporal_Position"] = "pre-exposure"
    frame["Same_Source_Construct_Risk"] = True
    frame["Source_Result_Version"] = SOURCE_VERSION
    frame["Reporting_Version"] = OUTPUT_VERSION
    frame["Reporting_Note"] = (
        "Coefficients and bootstrap draws are inherited from the frozen 20260717 analysis; only finite-bootstrap "
        "p-value reporting and claim labels are corrected here."
    )
    frame.to_csv(_output_path(root, dataset), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=500,
        help="Must equal the repetitions used to create the frozen 20260717 channel result.",
    )
    args = parser.parse_args()
    if args.bootstrap_reps < 20:
        parser.error("--bootstrap-reps must be at least 20")
    root = find_causal_root(__file__)
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        run_dataset(root, dataset, args.bootstrap_reps)


if __name__ == "__main__":
    main()
