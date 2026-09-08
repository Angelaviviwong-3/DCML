#!/usr/bin/env python3
"""Run the complete 20260717 Suning stage-specific recommendation benchmark."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


import argparse
import os
import subprocess
import sys
from pathlib import Path


MODELS = {
    "MBGCN": (
        "MBGCN/mbgcn_1_build_graph_data.py",
        "MBGCN/mbgcn_3_train_and_export.py",
    ),
    "KMCLR": (
        "KMCLR/KMCLR/kmclr_1_build_data.py",
        "KMCLR/KMCLR/kmclr_3_train_and_export.py",
    ),
    "MGAT": ("MGAT/mgat_1_prep_data.py", "MGAT/mgat_3_train_export.py"),
    "MICRO": ("MICRO/micro_1_prep_data.py", "MICRO/micro_3_train_export.py"),
    "SLMRec": ("SLMRec/slmrec_1_prep_data.py", "SLMRec/slmrec_3_train_export.py"),
    "DICE": ("DICE/dice_1_prep_data.py", "DICE/dice_3_train_export.py"),
    "CIRS": ("CIRS/cirs_1_prep_data.py", "CIRS/cirs_3_train_export.py"),
    "MGCE": ("MGCE/mgce_1_prep_data.py", "MGCE/mgce_3_train_export.py"),
}

CORE_FILES = {
    "MBGCN": "MBGCN/mbgcn_2_core_engine.py",
    "KMCLR": "KMCLR/KMCLR/kmclr_2_core_engine.py",
    "MGAT": "MGAT/mgat_2_core_engine.py",
    "MICRO": "MICRO/micro_2_core_engine.py",
    "SLMRec": "SLMRec/slmrec_2_model_wrapper.py",
    "DICE": "DICE/dice_2_model_wrapper.py",
    "CIRS": "CIRS/cirs_2_model_api.py",
    "MGCE": "MGCE/mgce_2_model_api.py",
}


def validate_script_identity(label: str, path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if _dcml_paths.source_version(path) != "20260717":
        raise RuntimeError(f"{label} script is not the 20260717 version: {path}")
    if label != "CANDIDATES" and label not in source:
        raise RuntimeError(
            f"Script identity mismatch for {label}: {path}. "
            f"The file does not contain its expected model marker {label!r}."
        )
    if "from causal." in source or "import causal." in source:
        raise RuntimeError(f"Stale package-style import detected in {path}; replace it with the current 20260717 file.")
    for stale_version in ("20260711", "20260712", "20260713"):
        if f"import {stale_version}" in source or f"from {stale_version}" in source:
            raise RuntimeError(f"Historical script dependency {stale_version} detected in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--only-prep", action="store_true")
    mode.add_argument("--only-train", action="store_true")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--skip-candidates", action="store_true", help="Do not rebuild the common 20260717 candidate bundle.")
    parser.add_argument("--check-only", action="store_true", help="Check that every required script exists without running it.")
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=None,
        help=(
            "Add this offset to every model's documented stage seed. When omitted, preserve the inherited "
            "SUNING_TRAINING_SEED_OFFSET; otherwise default to 0."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    causal = next(parent for parent in base.parents if parent.name == "causal")
    commands: list[tuple[str, Path]] = []
    if not args.only_train and not args.skip_candidates:
        commands.append(("CANDIDATES", causal / "baseline" / "suning" / "step1_prepare_baseline_data.py"))
    if not args.only_train:
        commands.extend((model, base / MODELS[model][0]) for model in args.models)
    if not args.only_prep:
        commands.extend((model, base / MODELS[model][1]) for model in args.models)

    support_files = []
    if not args.only_prep:
        support_files.extend((model, base / CORE_FILES[model]) for model in args.models)
    support_files.append(("UTILITIES", base / "baseline_utils.py"))
    checked_files = [*commands, *support_files]
    missing = [str(path) for _, path in checked_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing 20260717 scripts:\n" + "\n".join(missing))
    for label, path in checked_files:
        validate_script_identity(label if label != "UTILITIES" else "CANDIDATES", path)
    environment = os.environ.copy()
    inherited_offset = int(environment.get("SUNING_TRAINING_SEED_OFFSET", "0"))
    effective_offset = inherited_offset if args.seed_offset is None else int(args.seed_offset)
    environment["SUNING_TRAINING_SEED_OFFSET"] = str(effective_offset)
    if args.check_only:
        print(f"OK SUNING_TRAINING_SEED_OFFSET={effective_offset}")
        for label, path in commands:
            print(f"OK {label}: {path}")
        for label, path in support_files:
            print(f"OK {label} SUPPORT: {path}")
        return
    print(f"Using SUNING_TRAINING_SEED_OFFSET={effective_offset}", flush=True)
    for label, path in commands:
        print(f"\n===== Running {label}: {path} =====", flush=True)
        subprocess.run([sys.executable, str(path)], check=True, cwd=causal.parent, env=environment)


if __name__ == "__main__":
    main()
