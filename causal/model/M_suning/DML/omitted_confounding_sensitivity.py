#!/usr/bin/env python3
"""Suning wrapper for the 20260728 DML omitted-confounding analysis."""

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths


from pathlib import Path
import sys

CAUSAL_ROOT = Path(__file__).resolve().parents[3]
if str(CAUSAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CAUSAL_ROOT))

from dml_ovb_sensitivity import main


if __name__ == "__main__":
    main("suning")
