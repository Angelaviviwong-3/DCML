#!/usr/bin/env python3

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths

import sys
from pathlib import Path

CAUSAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CAUSAL_ROOT))

from dcml_causal_baselines import run_dataset

run_dataset(CAUSAL_ROOT, "amazon_beauty", 0.995)
