#!/usr/bin/env python3
"""Train one MICRO model per Amazon Purchase outcome and export audited scores."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

BASELINE_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = BASELINE_ROOT / "amazon_baseline_utils.py"
if not UTILS_PATH.is_file():
    raise FileNotFoundError(f"Missing 20260718 baseline utilities: {UTILS_PATH}")
sys.path.insert(0, str(BASELINE_ROOT))

from amazon_baseline_utils import (  # noqa: E402
    STAGES,
    baseline_data_dir,
    cache_dir,
    candidate_items,
    export_prediction_bundle,
    find_causal_root,
    load_candidates,
    read_json,
    set_reproducible_seed,
    setup_logger,
    stage_seed,
    write_json,
)
from micro_2_core_engine import MICROModel  # noqa: E402


def normalized_adjacency(interactions: dict[str, list[int]], users: int, items: int) -> torch.Tensor:
    edges = [(int(user), int(item) + users) for user, values in interactions.items() for item in values]
    if not edges:
        raise ValueError("MICRO stage training graph is empty")
    left = np.asarray([edge[0] for edge in edges] + [edge[1] for edge in edges], dtype=np.int64)
    right = np.asarray([edge[1] for edge in edges] + [edge[0] for edge in edges], dtype=np.int64)
    matrix = sp.coo_matrix((np.ones(len(left), dtype=np.float32), (left, right)), shape=(users + items, users + items))
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree, dtype=np.float32)
    inverse[degree > 0] = 1.0 / degree[degree > 0]
    matrix = sp.diags(inverse).dot(matrix).tocoo()
    indices = torch.as_tensor(np.vstack((matrix.row, matrix.col)), dtype=torch.long)
    values = torch.as_tensor(matrix.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce()


def sample_epoch(interactions: dict[str, list[int]], item_count: int, rng: np.random.Generator):
    user_positive = {int(user): set(map(int, items)) for user, items in interactions.items()}
    users = np.asarray(sorted(user_positive), dtype=np.int64)
    positives = np.asarray([rng.choice(tuple(user_positive[user])) for user in users], dtype=np.int64)
    negatives = rng.integers(0, item_count, size=len(users), dtype=np.int64)
    for idx, user in enumerate(users):
        while int(negatives[idx]) in user_positive[int(user)]:
            negatives[idx] = rng.integers(0, item_count)
    return users, positives, negatives


def main() -> None:
    root = find_causal_root(__file__)
    logger = setup_logger(root, "micro_stage_specific_train_export")
    cache = cache_dir(root, "micro_cache")
    users, items = map(int, (cache / "data_size_20260718.txt").read_text(encoding="utf-8").split())
    image = np.load(cache / "image_feat_20260718.npy")
    text = np.load(cache / "text_feat_20260718.npy")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diagnostics = {"script_version": "20260718", "stage_specific_training": True, "stages": {}}

    for stage in STAGES:
        seed = stage_seed(stage, 31)
        set_reproducible_seed(seed)
        rng = np.random.default_rng(seed)
        interactions = read_json(cache / f"train_data_{stage}_20260718.json")
        edge_count = sum(len(values) for values in interactions.values())
        adjacency = normalized_adjacency(interactions, users, items).to(device)
        model = MICROModel(users, items, 64, image, text, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        losses = []
        for epoch in range(25):
            model.train()
            user, positive, negative = sample_epoch(interactions, items, rng)
            user_t = torch.as_tensor(user, device=device)
            positive_t = torch.as_tensor(positive, device=device)
            negative_t = torch.as_tensor(negative, device=device)
            optimizer.zero_grad()
            user_embedding, item_embedding, image_embedding, text_embedding, fused = model(adjacency)
            positive_score = (user_embedding[user_t] * item_embedding[positive_t]).sum(dim=1)
            negative_score = (user_embedding[user_t] * item_embedding[negative_t]).sum(dim=1)
            bpr = -F.logsigmoid(positive_score - negative_score).mean()
            contrastive = model.contrastive_loss(image_embedding, fused, seed + epoch) + model.contrastive_loss(text_embedding, fused, seed + 100 + epoch)
            loss = bpr + 0.1 * contrastive
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            if epoch % 5 == 0 or epoch == 24:
                logger.info("MICRO/%s epoch=%02d loss=%.6f", stage, epoch, float(loss.item()))

        model.eval()
        with torch.no_grad():
            user_embedding, item_embedding, *_ = model(adjacency)
            predictions: dict[str, dict[str, float]] = {}
            for user, data in load_candidates(root, stage).items():
                ids = candidate_items(data)
                scores = user_embedding[int(user)] @ item_embedding[ids].t()
                predictions[str(user)] = {str(item): float(score) for item, score in zip(ids, scores.cpu().tolist())}
        metadata = export_prediction_bundle(
            root,
            "MICRO",
            stage,
            predictions,
            training_edges=edge_count,
            training_protocol="Independent stage-specific MICRO graph and BPR objective on Set B",
            extra_metadata={"epochs": 25, "seed": seed, "first_epoch_loss": losses[0], "last_epoch_loss": losses[-1]},
        )
        diagnostics["stages"][stage] = metadata
    write_json(diagnostics, baseline_data_dir(root) / "MICRO_training_diagnostics_20260718.json", indent=2)


if __name__ == "__main__":
    main()
