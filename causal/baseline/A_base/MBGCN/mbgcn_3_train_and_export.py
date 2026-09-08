#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one independently supervised MBGCN model per funnel stage."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths


import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

for parent in Path(__file__).resolve().parents:
    if (parent / "baseline_utils.py").exists():
        sys.path.insert(0, str(parent))
        break

from baseline_utils import (  # noqa: E402
    SCRIPT_VERSION,
    STAGES,
    baseline_data_dir,
    candidate_items,
    export_prediction_bundle,
    find_causal_root,
    load_candidates,
    set_reproducible_seed,
    setup_logger,
    stage_cache_dir,
    stage_seed,
    write_json,
)
from mbgcn_2_core_engine import MBGCN, TrainDataset, bprloss  # noqa: E402


CONFIG = {"lr": 1e-3, "L2_norm": 1e-3, "batch_size": 1024, "embedding_size": 64, "epoch": 30}


def train_stage(root: Path, stage: str, logger, device: torch.device) -> dict:
    seed = stage_seed(stage, 11)
    set_reproducible_seed(seed)
    cache = stage_cache_dir(root, "mbgcn_cache", stage)
    config = {**CONFIG, "relations": [stage]}
    trainset = TrainDataset(str(cache), [stage])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0, generator=generator)
    model = MBGCN(config, trainset, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    losses = []
    for epoch in range(CONFIG["epoch"]):
        model.train()
        total = 0.0
        for users, items in tqdm(loader, desc=f"MBGCN {stage} epoch {epoch + 1}", leave=False):
            optimizer.zero_grad()
            scores, regularization = model(users.to(device), items.to(device))
            loss = bprloss(scores, len(users)) + regularization
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        losses.append(total / max(len(loader), 1))
        trainset.newit()
        logger.info("MBGCN %s epoch %02d | loss=%.6f", stage, epoch + 1, losses[-1])

    model.eval()
    candidates = load_candidates(root, stage)
    predictions = {}
    with torch.no_grad():
        for user, data in tqdm(candidates.items(), desc=f"MBGCN export {stage}", leave=False):
            items = candidate_items(data)
            scores = model.evaluate(torch.tensor([int(user)], device=device)).squeeze(0)
            predictions[user] = {str(item): float(scores[item].detach().cpu().item()) for item in items}
    metadata = export_prediction_bundle(
        root,
        "MBGCN",
        stage,
        predictions,
        training_edges=int(len(trainset.checkins)),
        training_protocol="independent stage-specific BPR model on Set B",
        extra_metadata={"seed": seed, "first_epoch_loss": losses[0], "last_epoch_loss": losses[-1]},
    )
    return metadata


def main() -> None:
    root = find_causal_root(__file__)
    logger = setup_logger(root, "mbgcn_stage_specific_train_export")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diagnostics = {"script_version": SCRIPT_VERSION, "stage_specific_training": True, "stages": {}}
    for stage in STAGES:
        diagnostics["stages"][stage] = train_stage(root, stage, logger, device)
    write_json(diagnostics, baseline_data_dir(root) / f"MBGCN_training_diagnostics_{SCRIPT_VERSION}.json", indent=2)


if __name__ == "__main__":
    main()
