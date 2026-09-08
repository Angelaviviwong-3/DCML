#!/usr/bin/env python3

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths

from amazon_dataset_builder_common import build_amazon_dataset


if __name__ == "__main__":
    build_amazon_dataset("amazon_appliances")
