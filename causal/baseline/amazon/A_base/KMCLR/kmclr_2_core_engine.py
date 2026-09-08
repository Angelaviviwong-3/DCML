#!/usr/bin/env python3
"""Independent graph contrastive encoder used by the 20260718 KMCLR rerun."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


import torch
import torch.nn as nn
import torch.nn.functional as F


class KMCLR_Model(nn.Module):
    def __init__(self, user_num: int, item_num: int, behavior_mats: dict, config: dict, device):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.layers = int(config["layers"])
        dim = int(config["embedding_size"])
        self.user_embedding = nn.Parameter(torch.empty(user_num, dim))
        self.item_embedding = nn.Parameter(torch.empty(item_num, dim))
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)
        self.behavior_mats = {name: matrix.to(device) for name, matrix in behavior_mats.items()}

    def forward(self):
        initial = torch.cat((self.user_embedding, self.item_embedding), dim=0)
        outputs = [initial]
        for adjacency in self.behavior_mats.values():
            current = initial
            layers = [initial]
            for _ in range(self.layers):
                current = torch.sparse.mm(adjacency, current)
                layers.append(current)
            outputs.append(torch.stack(layers, dim=1).mean(dim=1))
        final = torch.stack(outputs, dim=1).mean(dim=1)
        return torch.split(final, (self.user_num, self.item_num))


def contrastive_loss(left: torch.Tensor, right: torch.Tensor, temperature: float = 0.2, max_sample: int = 4096):
    if left.shape[0] > max_sample:
        indices = torch.randperm(left.shape[0], device=left.device)[:max_sample]
        left, right = left[indices], right[indices]
    logits = F.normalize(left, dim=1) @ F.normalize(right, dim=1).t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)
