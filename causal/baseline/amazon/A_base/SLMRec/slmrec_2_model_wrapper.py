#!/usr/bin/env python3
"""Self-supervised multimodal recommendation encoder for the 20260718 rerun."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SLMRecEngine(nn.Module):
    def __init__(self, users: int, items: int, dim: int, visual: np.ndarray, text: np.ndarray, audio: np.ndarray, device):
        super().__init__()
        self.user_embedding = nn.Embedding(users, dim)
        self.item_embedding = nn.Embedding(items, dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.register_buffer("visual", torch.as_tensor(visual, dtype=torch.float32, device=device))
        self.register_buffer("text", torch.as_tensor(text, dtype=torch.float32, device=device))
        self.register_buffer("audio", torch.as_tensor(audio, dtype=torch.float32, device=device))
        self.visual_projection = nn.Linear(visual.shape[1], dim)
        self.text_projection = nn.Linear(text.shape[1], dim)
        self.audio_projection = nn.Linear(audio.shape[1], dim)

    def compute(self):
        visual = self.visual_projection(self.visual)
        text = self.text_projection(self.text)
        audio = self.audio_projection(self.audio)
        items = (self.item_embedding.weight + visual + text + audio) / 4.0
        return self.user_embedding.weight, items

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        item_embedding = (
            self.item_embedding(items)
            + self.visual_projection(self.visual[items])
            + self.text_projection(self.text[items])
            + self.audio_projection(self.audio[items])
        ) / 4.0
        return (self.user_embedding(users) * item_embedding).sum(dim=1)


def bpr_loss(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(positive - negative).mean()
