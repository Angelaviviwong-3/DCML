#!/usr/bin/env python3
"""Prepare/train all purchase-only Amazon prediction baselines for one dataset."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_VERSION = "20260718"
VALID_DATASETS = ("amazon_appliances", "amazon_beauty")
MODELS = {
    "MBGCN": (
        "MBGCN/mbgcn_1_build_graph_data.py",
        "MBGCN/mbgcn_3_train_and_export.py",
    ),
    "KMCLR": (
        "KMCLR/kmclr_1_build_data.py",
        "KMCLR/kmclr_3_train_and_export.py",
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
    "KMCLR": "KMCLR/kmclr_2_core_engine.py",
    "MGAT": "MGAT/mgat_2_core_engine.py",
    "MICRO": "MICRO/micro_2_core_engine.py",
    "SLMRec": "SLMRec/slmrec_2_model_wrapper.py",
    "DICE": "DICE/dice_2_model_wrapper.py",
    "CIRS": "CIRS/cirs_2_model_api.py",
    "MGCE": "MGCE/mgce_2_model_api.py",
}
TORCH_MODELS = {"MBGCN", "KMCLR", "MGAT", "MICRO", "SLMRec", "DICE"}


def required_modules(models: list[str], *, prepare: bool, train: bool) -> list[str]:
    modules = {"numpy", "pandas"}
    selected = set(models)
    if prepare and "KMCLR" in selected:
        modules.add("scipy")
    if train and selected & TORCH_MODELS:
        modules.add("torch")
    if train and "MGAT" in selected:
        modules.add("torch_geometric")
    if train and selected & {"CIRS", "MGCE"}:
        modules.add("tensorflow")
    return sorted(modules)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--only-prep", action="store_true")
    mode.add_argument("--only-train", action="store_true")
    parser.add_argument("--dataset", choices=VALID_DATASETS, default=os.environ.get("AMAZON_BASELINE_DATASET"))
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=None)
    args = parser.parse_args()
    if args.dataset not in VALID_DATASETS:
        parser.error("--dataset is required")
    return args


def validate_file(path: Path, marker: str | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    source = path.read_text(encoding="utf-8")
    if _dcml_paths.source_version(path) != SCRIPT_VERSION:
        raise RuntimeError(f"Not a {SCRIPT_VERSION} script: {path}")
    if marker and marker not in source:
        raise RuntimeError(f"Script identity mismatch; expected {marker!r}: {path}")
    if "baseline_utils_20260717" in source or "20260717.txt" in source or "20260717.npy" in source:
        raise RuntimeError(f"Stale 20260717 dependency in {path}")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    base = script_dir / "A_base"
    causal_root = script_dir.parents[1]
    commands: list[tuple[str, Path]] = []
    if not args.only_train:
        commands.extend((model, base / MODELS[model][0]) for model in args.models)
    if not args.only_prep:
        commands.extend((model, base / MODELS[model][1]) for model in args.models)
    support = [("UTILITIES", base / "amazon_baseline_utils.py")]
    if not args.only_prep:
        support.extend((model, base / CORE_FILES[model]) for model in args.models)
    modules = required_modules(
        args.models,
        prepare=not args.only_train,
        train=not args.only_prep,
    )
    missing_modules = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing_modules:
        raise RuntimeError(
            "Missing Python dependencies for the selected Amazon prediction baselines: "
            + ", ".join(missing_modules)
        )
    for label, path in [*commands, *support]:
        validate_file(path, None if label == "UTILITIES" else label)

    environment = os.environ.copy()
    environment["AMAZON_BASELINE_DATASET"] = args.dataset
    inherited = int(environment.get("AMAZON_TRAINING_SEED_OFFSET", "0"))
    effective = inherited if args.seed_offset is None else int(args.seed_offset)
    environment["AMAZON_TRAINING_SEED_OFFSET"] = str(effective)
    if args.check_only:
        print(f"OK dataset={args.dataset} AMAZON_TRAINING_SEED_OFFSET={effective}")
        print(f"OK dependencies: {', '.join(modules)}")
        for label, path in commands:
            print(f"OK {label}: {path}")
        for label, path in support:
            print(f"OK {label} SUPPORT: {path}")
        return

    print(f"Dataset={args.dataset} AMAZON_TRAINING_SEED_OFFSET={effective}", flush=True)
    for label, path in commands:
        print(f"\n===== Running {label}: {path} =====", flush=True)
        subprocess.run([sys.executable, str(path)], check=True, cwd=causal_root.parent, env=environment)


if __name__ == "__main__":
    main()
