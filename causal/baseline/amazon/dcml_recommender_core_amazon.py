#!/usr/bin/env python3
"""Core Purchase-only model for the independent Amazon DCML recommender."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


import math

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for the DCML recommender") from exc


SCRIPT_VERSION = "20260718"
STAGES = ("purchase",)


class DCMLRecommender(nn.Module):
    """Purchase-specific collaborative-content model with an internal DCML scoring head."""

    def __init__(self, users: int, items: int, content_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.users = int(users)
        self.items = int(items)
        self.embedding_dim = int(embedding_dim)
        self.user_embedding = nn.Embedding(users, embedding_dim)
        self.item_embedding = nn.Embedding(items, embedding_dim)
        self.stage_item_embedding = nn.ModuleList([nn.Embedding(items, embedding_dim) for _ in STAGES])
        self.stage_item_bias = nn.ModuleList([nn.Embedding(items, 1) for _ in STAGES])
        self.stage_user_transform = nn.Parameter(torch.eye(embedding_dim).repeat(len(STAGES), 1, 1))
        self.stage_embedding = nn.Embedding(len(STAGES), embedding_dim)
        self.involvement_embedding = nn.Embedding(4, embedding_dim)
        self.content_encoder = nn.Sequential(
            nn.Linear(content_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Tanh(),
        )
        self.content_scale_raw = nn.Parameter(torch.zeros(len(STAGES)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.normal_(self.stage_embedding.weight, std=0.02)
        nn.init.normal_(self.involvement_embedding.weight, std=0.02)
        for embedding in self.stage_item_embedding:
            nn.init.normal_(embedding.weight, std=0.02)
        for bias in self.stage_item_bias:
            nn.init.zeros_(bias.weight)
        linear = self.content_encoder[0]
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)

    def relevance_score(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        stages: torch.Tensor,
        h_levels: torch.Tensor,
        item_content: torch.Tensor,
    ) -> torch.Tensor:
        user_base = self.user_embedding(users)
        transforms = self.stage_user_transform[stages]
        user_stage = torch.bmm(user_base.unsqueeze(1), transforms).squeeze(1)
        user_stage = user_stage + self.stage_embedding(stages) + self.involvement_embedding(h_levels)

        item_stage = self.item_embedding(items)
        stage_specific = torch.stack(
            [self.stage_item_embedding[index](items) for index in range(len(STAGES))],
            dim=1,
        )
        gather_index = stages.view(-1, 1, 1).expand(-1, 1, self.embedding_dim)
        item_stage = item_stage + stage_specific.gather(1, gather_index).squeeze(1)
        collaborative = (user_stage * item_stage).sum(dim=1) / math.sqrt(self.embedding_dim)

        content_vector = self.content_encoder(item_content)
        content_match = (F.normalize(user_stage, dim=1) * F.normalize(content_vector, dim=1)).sum(dim=1)
        content_scale = F.softplus(self.content_scale_raw[stages])
        biases = torch.stack(
            [self.stage_item_bias[index](items).squeeze(1) for index in range(len(STAGES))],
            dim=1,
        ).gather(1, stages.view(-1, 1)).squeeze(1)
        return collaborative + content_scale * content_match + biases

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        stages: torch.Tensor,
        h_levels: torch.Tensor,
        item_content: torch.Tensor,
    ) -> torch.Tensor:
        return self.relevance_score(users, items, stages, h_levels, item_content)


def bpr_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(positive_scores - negative_scores).mean()


def funnel_order_loss(stage_scores: list[torch.Tensor]) -> torch.Tensor:
    """Return zero because Amazon has no observed Click/Cart funnel outcomes."""
    if not stage_scores:
        raise ValueError("stage_scores cannot be empty")
    return stage_scores[0].new_zeros(())
