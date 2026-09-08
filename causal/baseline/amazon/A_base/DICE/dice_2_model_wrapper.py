#!/usr/bin/env python3
"""DICE interest/popularity disentanglement module for the 20260718 rerun."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


import torch
import torch.nn as nn
import torch.nn.functional as F


class DICEEngine(nn.Module):
    def __init__(self, users: int, items: int, dim: int):
        super().__init__()
        self.user_interest = nn.Embedding(users, dim)
        self.user_popularity = nn.Embedding(users, dim)
        self.item_interest = nn.Embedding(items, dim)
        self.item_popularity = nn.Embedding(items, dim)
        for embedding in (self.user_interest, self.user_popularity, self.item_interest, self.item_popularity):
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, users: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor):
        interest = (self.user_interest(users), self.item_interest(positives), self.item_interest(negatives))
        popularity = (self.user_popularity(users), self.item_popularity(positives), self.item_popularity(negatives))
        return interest, popularity

    def final_embeddings(self):
        return self.user_interest.weight + self.user_popularity.weight, self.item_interest.weight + self.item_popularity.weight


def dice_loss(interest, popularity, positive_more_popular: torch.Tensor, penalty: float = 0.01) -> torch.Tensor:
    user_i, positive_i, negative_i = interest
    user_p, positive_p, negative_p = popularity
    interest_gap = (user_i * (positive_i - negative_i)).sum(dim=1)
    popularity_gap = (user_p * (positive_p - negative_p)).sum(dim=1)
    interest_loss = -F.logsigmoid(interest_gap).mean()
    signed_popularity_gap = torch.where(positive_more_popular, popularity_gap, -popularity_gap)
    popularity_loss = -F.logsigmoid(signed_popularity_gap).mean()
    independence = (user_i * user_p).sum(dim=1).square().mean() + (positive_i * positive_p).sum(dim=1).square().mean()
    return interest_loss + popularity_loss + penalty * independence
