#!/usr/bin/env python3

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths

import os
import runpy
from pathlib import Path

os.environ["AMAZON_BASELINE_DATASET"] = "amazon_beauty"
runpy.run_path(str(Path(__file__).with_name("step2_train_dcml_recommender_amazon.py")), run_name="__main__")
