#!/usr/bin/env python3
"""Train one SLMRec model per Suning funnel outcome and export audited scores."""

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
from torch.utils.data import DataLoader, Dataset

for parent in Path(__file__).resolve().parents:
    if (parent / "baseline_utils.py").exists():
        sys.path.insert(0, str(parent))
        break

from baseline_utils import (  # noqa: E402
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
from slmrec_2_model_wrapper import SLMRecEngine, bpr_loss  # noqa: E402


class PairDataset(Dataset):
    def __init__(self, path: Path, item_count: int, seed: int):
        data = np.loadtxt(path, dtype=np.int64)
        self.data = data.reshape(-1, 2)
        self.item_count = item_count
        self.rng = np.random.default_rng(seed)
        self.user_positive: dict[int, set[int]] = {}
        for user, item in self.data:
            self.user_positive.setdefault(int(user), set()).add(int(item))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        user, positive = map(int, self.data[index])
        negative = int(self.rng.integers(self.item_count))
        while negative in self.user_positive[user]:
            negative = int(self.rng.integers(self.item_count))
        return user, positive, negative


def main() -> None:
    root = find_causal_root(__file__)
    logger = setup_logger(root, "slmrec_stage_specific_train_export")
    cache = cache_dir(root, "slmrec_cache")
    users, items = map(int, (cache / "data_size_20260717.txt").read_text(encoding="utf-8").split())
    visual = np.load(cache / "v_feat_20260717.npy")
    text = np.load(cache / "t_feat_20260717.npy")
    audio = np.load(cache / "a_feat_20260717.npy")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diagnostics = {"script_version": "20260717", "stage_specific_training": True, "stages": {}}

    for stage in STAGES:
        seed = stage_seed(stage, 41)
        set_reproducible_seed(seed)
        dataset = PairDataset(cache / f"train_{stage}_20260717.txt", items, seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=2048, shuffle=True, generator=generator, num_workers=0)
        model = SLMRecEngine(users, items, 64, visual, text, audio, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        losses = []
        for epoch in range(20):
            model.train()
            total = 0.0
            for user, positive, negative in loader:
                user, positive, negative = user.to(device), positive.to(device), negative.to(device)
                optimizer.zero_grad()
                loss = bpr_loss(model.score(user, positive), model.score(user, negative))
                loss.backward()
                optimizer.step()
                total += float(loss.item())
            epoch_loss = total / max(len(loader), 1)
            losses.append(epoch_loss)
            logger.info("SLMRec/%s epoch=%02d loss=%.6f", stage, epoch, epoch_loss)

        model.eval()
        with torch.no_grad():
            user_embedding, item_embedding = model.compute()
            predictions: dict[str, dict[str, float]] = {}
            for user, data in load_candidates(root, stage).items():
                ids = candidate_items(data)
                scores = user_embedding[int(user)] @ item_embedding[ids].t()
                predictions[str(user)] = {str(item): float(score) for item, score in zip(ids, scores.cpu().tolist())}
        metadata = export_prediction_bundle(
            root,
            "SLMRec",
            stage,
            predictions,
            training_edges=len(dataset),
            training_protocol="Independent stage-specific SLMRec multimodal BPR objective on Set B",
            extra_metadata={"epochs": 20, "seed": seed, "first_epoch_loss": losses[0], "last_epoch_loss": losses[-1]},
        )
        diagnostics["stages"][stage] = metadata
    write_json(diagnostics, baseline_data_dir(root) / "SLMRec_training_diagnostics_20260717.json", indent=2)


if __name__ == "__main__":
    main()
