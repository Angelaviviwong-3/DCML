#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one MGAT model for each Suning funnel outcome."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths


import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
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
    cache_dir,
    candidate_items,
    export_prediction_bundle,
    find_causal_root,
    load_candidates,
    set_reproducible_seed,
    setup_logger,
    stage_seed,
    write_json,
)
from mgat_2_core_engine import MGATModel, bpr_loss  # noqa: E402


def sample_negative_offsets(edges: np.ndarray, adjacency: dict, user_num: int, item_num: int, rng) -> np.ndarray:
    negatives = rng.integers(user_num, user_num + item_num, size=len(edges))
    invalid = np.asarray([int(item) in adjacency.get(int(user), []) for (user, _), item in zip(edges, negatives)])
    while invalid.any():
        negatives[invalid] = rng.integers(user_num, user_num + item_num, size=int(invalid.sum()))
        invalid = np.asarray([int(item) in adjacency.get(int(user), []) for (user, _), item in zip(edges, negatives)])
    return negatives


def train_stage(root: Path, stage: str, logger, device: torch.device, features, user_num: int, item_num: int) -> dict:
    seed = stage_seed(stage, 31)
    set_reproducible_seed(seed)
    rng = np.random.default_rng(seed)
    cache = cache_dir(root, "mgat_cache")
    edges = np.load(cache / f"train_{stage}_20260717.npy")
    adjacency = np.load(cache / f"adjacency_{stage}_20260717.npy", allow_pickle=True).item()
    if not len(edges):
        raise ValueError(f"MGAT {stage} has no training interactions")
    model = MGATModel(features, edges, user_num, item_num, 64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    user_tensor = torch.as_tensor(edges[:, 0], dtype=torch.long, device=device)
    positive_tensor = torch.as_tensor(edges[:, 1], dtype=torch.long, device=device)
    losses = []
    for epoch in range(15):
        model.train()
        negative = sample_negative_offsets(edges, adjacency, user_num, item_num, rng)
        optimizer.zero_grad()
        positive_score, negative_score = model(
            user_tensor,
            positive_tensor,
            torch.as_tensor(negative, dtype=torch.long, device=device),
        )
        loss = bpr_loss(positive_score, negative_score)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        logger.info("MGAT %s epoch %02d | loss=%.6f", stage, epoch + 1, losses[-1])

    model.eval()
    predictions = {}
    with torch.no_grad():
        embeddings = model.compute()
        if not torch.isfinite(embeddings).all():
            raise RuntimeError(f"MGAT {stage} produced non-finite embeddings")
        for user, data in tqdm(load_candidates(root, stage).items(), desc=f"MGAT export {stage}", leave=False):
            items = candidate_items(data)
            offsets = torch.as_tensor([item + user_num for item in items], device=device)
            scores = torch.matmul(embeddings[int(user)], embeddings[offsets].t())
            predictions[user] = {str(item): float(score.detach().cpu().item()) for item, score in zip(items, scores)}
    return export_prediction_bundle(
        root,
        "MGAT",
        stage,
        predictions,
        training_edges=int(len(edges)),
        training_protocol="independent stage-specific multimodal graph-attention BPR model on Set B",
        extra_metadata={
            "seed": seed,
            "message_passing_edges": int(model.edge_index.shape[1]),
            "first_epoch_loss": losses[0],
            "last_epoch_loss": losses[-1],
        },
    )


def main() -> None:
    root = find_causal_root(__file__)
    logger = setup_logger(root, "mgat_stage_specific_train_export")
    cache = cache_dir(root, "mgat_cache")
    with open(cache / "data_size_20260717.txt", "r", encoding="utf-8") as handle:
        user_num, item_num = map(int, handle.read().split())
    features = [np.load(cache / f"Feature{name}_normal_20260717.npy") for name in ("Video", "Text", "Audio")]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diagnostics = {"script_version": SCRIPT_VERSION, "stage_specific_training": True, "stages": {}}
    for stage in STAGES:
        diagnostics["stages"][stage] = train_stage(root, stage, logger, device, features, user_num, item_num)
    write_json(diagnostics, baseline_data_dir(root) / f"MGAT_training_diagnostics_{SCRIPT_VERSION}.json", indent=2)


if __name__ == "__main__":
    main()
