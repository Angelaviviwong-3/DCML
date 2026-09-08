#!/usr/bin/env python3
"""Compact MICRO-style multimodal graph encoder used by the 20260717 rerun."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MICROModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, image: np.ndarray, text: np.ndarray, device):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.register_buffer("image_features", torch.as_tensor(image, dtype=torch.float32, device=device))
        self.register_buffer("text_features", torch.as_tensor(text, dtype=torch.float32, device=device))
        self.image_projection = nn.Linear(image.shape[1], dim)
        self.text_projection = nn.Linear(text.shape[1], dim)

    def forward(self, normalized_adjacency: torch.Tensor):
        image_embedding = self.image_projection(self.image_features)
        text_embedding = self.text_projection(self.text_features)
        fused_item = (self.item_embedding.weight + image_embedding + text_embedding) / 3.0
        current = torch.cat((self.user_embedding.weight, fused_item), dim=0)
        layers = [current]
        for _ in range(2):
            current = torch.sparse.mm(normalized_adjacency, current)
            layers.append(current)
        output = torch.stack(layers, dim=1).mean(dim=1)
        users, items = torch.split(output, (self.n_users, self.n_items))
        return users, items, image_embedding, text_embedding, fused_item

    @staticmethod
    def contrastive_loss(left: torch.Tensor, right: torch.Tensor, seed: int, maximum: int = 2048) -> torch.Tensor:
        if left.shape[0] > maximum:
            generator = torch.Generator(device=left.device)
            generator.manual_seed(seed)
            indices = torch.randperm(left.shape[0], generator=generator, device=left.device)[:maximum]
            left, right = left[indices], right[indices]
        logits = F.normalize(left, dim=1) @ F.normalize(right, dim=1).t() / 0.2
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)
