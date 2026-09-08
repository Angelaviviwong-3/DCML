#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train independently supervised KMCLR models for Click, Cart, and Purchase."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
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
from kmclr_2_core_engine import KMCLR_Model, contrastive_loss  # noqa: E402


CONFIG = {"embedding_size": 64, "lr": 1e-3, "layers": 2, "epochs": 25, "beta": 0.1, "max_sample": 4096}


def normalized_adjacency(matrix: sp.csr_matrix) -> torch.Tensor:
    adjacency = sp.bmat([[None, matrix], [matrix.T, None]], format="csr")
    degree = np.asarray(adjacency.sum(1)).reshape(-1)
    inv = np.power(degree, -0.5, where=degree > 0)
    inv[degree <= 0] = 0.0
    normalized = sp.diags(inv).dot(adjacency).dot(sp.diags(inv)).tocoo()
    indices = torch.as_tensor(np.vstack((normalized.row, normalized.col)), dtype=torch.long)
    values = torch.as_tensor(normalized.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, torch.Size(normalized.shape)).coalesce()


def sample_negatives(users: np.ndarray, positives: np.ndarray, user_pos: dict[int, set[int]], item_num: int, rng) -> np.ndarray:
    negatives = rng.integers(0, item_num, size=len(users))
    invalid = np.asarray([int(item) in user_pos[int(user)] for user, item in zip(users, negatives)])
    while invalid.any():
        negatives[invalid] = rng.integers(0, item_num, size=int(invalid.sum()))
        invalid = np.asarray([int(item) in user_pos[int(user)] for user, item in zip(users, negatives)])
    return negatives


def train_stage(root: Path, stage: str, logger, device: torch.device) -> dict:
    seed = stage_seed(stage, 21)
    set_reproducible_seed(seed)
    rng = np.random.default_rng(seed)
    cache = cache_dir(root, "kmclr_cache")
    with open(cache / "data_size_20260717.txt", "r", encoding="utf-8") as handle:
        user_num, item_num = map(int, handle.read().split())
    with open(cache / f"trn_{stage}.pkl", "rb") as handle:
        matrix: sp.csr_matrix = pickle.load(handle)
    coo = matrix.tocoo()
    users = coo.row.astype(np.int64)
    positives = coo.col.astype(np.int64)
    if not len(users):
        raise ValueError(f"KMCLR {stage} has no training interactions")
    user_pos: dict[int, set[int]] = {}
    for user, item in zip(users, positives):
        user_pos.setdefault(int(user), set()).add(int(item))
    adjacency = normalized_adjacency(matrix)
    model = KMCLR_Model(user_num, item_num, {stage: adjacency}, CONFIG, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    losses = []
    for epoch in range(CONFIG["epochs"]):
        model.train()
        user_emb, item_emb = model()
        negatives = sample_negatives(users, positives, user_pos, item_num, rng)
        user_tensor = torch.as_tensor(users, device=device)
        pos_tensor = torch.as_tensor(positives, device=device)
        neg_tensor = torch.as_tensor(negatives, device=device)
        pos_score = torch.sum(user_emb[user_tensor] * item_emb[pos_tensor], dim=1)
        neg_score = torch.sum(user_emb[user_tensor] * item_emb[neg_tensor], dim=1)
        ranking_loss = -F.logsigmoid(pos_score - neg_score).mean()
        regularizer = contrastive_loss(
            F.dropout(user_emb, p=0.1, training=True),
            F.dropout(user_emb, p=0.1, training=True),
            max_sample=CONFIG["max_sample"],
        )
        regularizer += contrastive_loss(
            F.dropout(item_emb, p=0.1, training=True),
            F.dropout(item_emb, p=0.1, training=True),
            max_sample=CONFIG["max_sample"],
        )
        loss = ranking_loss + CONFIG["beta"] * regularizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        logger.info("KMCLR %s epoch %02d | loss=%.6f", stage, epoch + 1, losses[-1])

    model.eval()
    predictions = {}
    with torch.no_grad():
        final_user, final_item = model()
        for user, data in tqdm(load_candidates(root, stage).items(), desc=f"KMCLR export {stage}", leave=False):
            items = candidate_items(data)
            item_tensor = torch.as_tensor(items, device=device)
            scores = torch.matmul(final_user[int(user)], final_item[item_tensor].t())
            predictions[user] = {str(item): float(score.detach().cpu().item()) for item, score in zip(items, scores)}
    return export_prediction_bundle(
        root,
        "KMCLR",
        stage,
        predictions,
        training_edges=int(len(users)),
        training_protocol="independent stage-specific graph encoder with BPR supervision on Set B",
        extra_metadata={"seed": seed, "first_epoch_loss": losses[0], "last_epoch_loss": losses[-1]},
    )


def main() -> None:
    root = find_causal_root(__file__)
    logger = setup_logger(root, "kmclr_stage_specific_train_export")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diagnostics = {"script_version": SCRIPT_VERSION, "stage_specific_training": True, "stages": {}}
    for stage in STAGES:
        diagnostics["stages"][stage] = train_stage(root, stage, logger, device)
    write_json(diagnostics, baseline_data_dir(root) / f"KMCLR_training_diagnostics_{SCRIPT_VERSION}.json", indent=2)


if __name__ == "__main__":
    main()
