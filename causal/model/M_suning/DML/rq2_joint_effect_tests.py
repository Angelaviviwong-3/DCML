#!/usr/bin/env python3

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths

import argparse
import sys
from pathlib import Path

CAUSAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CAUSAL_ROOT))

from rq2_joint_effect_tests import run_rq2_joint_tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orthog-mode",
        choices=["linear", "linear_quadratic"],
        default="linear",
    )
    args = parser.parse_args()
    run_rq2_joint_tests(CAUSAL_ROOT, "suning", args.orthog_mode)
