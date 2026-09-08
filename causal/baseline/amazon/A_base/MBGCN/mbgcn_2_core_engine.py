#!/usr/bin/env python3
"""Independent stage-graph encoder used by the 20260718 MBGCN rerun."""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def bprloss(scores: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
    positive = scores[:, :1]
    negative = scores[:, 1:]
    return -F.logsigmoid(positive - negative).mean()


class TrainDataset(Dataset):
    def __init__(self, cache_path: str, relations: list[str]):
        self.path = Path(cache_path)
        self.relations = relations
        self.user_num, self.item_num = map(int, (self.path / "data_size_20260718.txt").read_text(encoding="utf-8").split())
        self.checkins = np.loadtxt(self.path / f"{relations[0]}_20260718.txt", dtype=np.int64).reshape(-1, 2)
        self.relation_dict = {relation: self._adjacency(relation) for relation in relations}
        self.sample_index = 0
        self._load_samples()

    def _adjacency(self, relation: str) -> torch.Tensor:
        pairs = np.loadtxt(self.path / f"{relation}_20260718.txt", dtype=np.int64).reshape(-1, 2)
        users = pairs[:, 0]
        items = pairs[:, 1] + self.user_num
        source = np.concatenate((users, items))
        target = np.concatenate((items, users))
        indices = torch.as_tensor(np.vstack((source, target)), dtype=torch.long)
        values = torch.ones(len(source), dtype=torch.float32)
        size = self.user_num + self.item_num
        adjacency = torch.sparse_coo_tensor(indices, values, (size, size)).coalesce()
        degree = torch.sparse.sum(adjacency, dim=1).to_dense()
        inverse = torch.zeros_like(degree)
        inverse[degree > 0] = degree[degree > 0].pow(-0.5)
        normalized_values = inverse[adjacency.indices()[0]] * adjacency.values() * inverse[adjacency.indices()[1]]
        return torch.sparse_coo_tensor(adjacency.indices(), normalized_values, adjacency.shape).coalesce()

    def _load_samples(self) -> None:
        path = self.path / "sample_file" / f"sample_{self.sample_index % 5}_20260718.txt"
        self.samples = torch.as_tensor(np.loadtxt(path, dtype=np.int64).reshape(-1, 3), dtype=torch.long)

    def newit(self) -> None:
        self.sample_index += 1
        self._load_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples[index]
        return row[0], row[1:]


class MBGCN(nn.Module):
    def __init__(self, config: dict, trainset: TrainDataset, device):
        super().__init__()
        self.user_num = trainset.user_num
        self.item_num = trainset.item_num
        dim = int(config["embedding_size"])
        self.l2 = float(config["L2_norm"])
        self.user_embedding = nn.Parameter(torch.empty(self.user_num, dim))
        self.item_embedding = nn.Parameter(torch.empty(self.item_num, dim))
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)
        self.relation_dict = {name: adjacency.to(device) for name, adjacency in trainset.relation_dict.items()}

    def embeddings(self):
        initial = torch.cat((self.user_embedding, self.item_embedding), dim=0)
        relation_outputs = []
        for adjacency in self.relation_dict.values():
            first = torch.sparse.mm(adjacency, initial)
            second = torch.sparse.mm(adjacency, first)
            relation_outputs.append(torch.stack((initial, first, second), dim=1).mean(dim=1))
        final = torch.stack(relation_outputs, dim=1).mean(dim=1) if relation_outputs else initial
        return torch.split(final, (self.user_num, self.item_num))

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        user_embedding, item_embedding = self.embeddings()
        selected_users = user_embedding[users].unsqueeze(1)
        selected_items = item_embedding[items]
        scores = (selected_users * selected_items).sum(dim=2)
        regularization = self.l2 * (selected_users.square().mean() + selected_items.square().mean())
        return scores, regularization

    def evaluate(self, users: torch.Tensor):
        user_embedding, item_embedding = self.embeddings()
        return user_embedding[users] @ item_embedding.t()
