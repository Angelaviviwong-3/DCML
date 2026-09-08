#!/usr/bin/env python3

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths

import runpy
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "prediction_multiseed_supplement.py"
sys.argv = [str(SCRIPT), "--dataset", "suning", *sys.argv[1:]]
runpy.run_path(str(SCRIPT), run_name="__main__")
