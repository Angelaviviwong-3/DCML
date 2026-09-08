#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent bidirectional MGAT engine for the 20260718 rerun."""

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[4]))
import project_paths as _dcml_paths


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, softmax


class GraphGAT(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, features, edge_index, size):
        edge_index, _ = remove_self_loops(edge_index)
        projected = torch.matmul(features, self.weight)
        return self.propagate(edge_index, size=size, x=projected)

    def message(self, edge_index_i, x_i, x_j, size_i, size):
        inner = torch.mul(x_i, F.leaky_relu(x_j)).sum(dim=-1)
        attention = softmax(inner * torch.sigmoid(inner), index=edge_index_i, num_nodes=size_i)
        return x_j * attention.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out + self.bias


class MGATModel(nn.Module):
    def __init__(self, features, edges, user_num: int, item_num: int, latent_dim: int):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.user_embeddings = nn.Embedding(user_num, latent_dim)
        self.item_embeddings = nn.Embedding(item_num, latent_dim)
        nn.init.xavier_normal_(self.user_embeddings.weight)
        nn.init.xavier_normal_(self.item_embeddings.weight)
        self.register_buffer("visual", torch.as_tensor(features[0], dtype=torch.float32))
        self.register_buffer("text", torch.as_tensor(features[1], dtype=torch.float32))
        self.register_buffer("audio", torch.as_tensor(features[2], dtype=torch.float32))
        self.visual_projection = nn.Linear(self.visual.shape[1], latent_dim)
        self.text_projection = nn.Linear(self.text.shape[1], latent_dim)
        self.audio_projection = nn.Linear(self.audio.shape[1], latent_dim)
        self.visual_gat = GraphGAT(latent_dim, latent_dim)
        self.text_gat = GraphGAT(latent_dim, latent_dim)
        self.audio_gat = GraphGAT(latent_dim, latent_dim)
        directed = torch.as_tensor(edges, dtype=torch.long).t().contiguous()
        self.register_buffer("edge_index", torch.cat([directed, directed.flip(0)], dim=1))
        self.result_embeddings = None

    def compute(self):
        user = self.user_embeddings.weight
        item = self.item_embeddings.weight
        visual = torch.cat([user, item + self.visual_projection(self.visual)], dim=0)
        text = torch.cat([user, item + self.text_projection(self.text)], dim=0)
        audio = torch.cat([user, item + self.audio_projection(self.audio)], dim=0)
        size = (self.user_num + self.item_num, self.user_num + self.item_num)
        visual = F.normalize(visual + self.visual_gat(visual, self.edge_index, size), p=2, dim=1)
        text = F.normalize(text + self.text_gat(text, self.edge_index, size), p=2, dim=1)
        audio = F.normalize(audio + self.audio_gat(audio, self.edge_index, size), p=2, dim=1)
        self.result_embeddings = F.normalize((visual + text + audio) / 3.0, p=2, dim=1)
        return self.result_embeddings

    def forward(self, users, positives, negatives):
        embeddings = self.compute()
        user = embeddings[users]
        positive = embeddings[positives]
        negative = embeddings[negatives]
        return torch.sum(user * positive, dim=1), torch.sum(user * negative, dim=1)


def bpr_loss(positive, negative):
    return -F.logsigmoid(positive - negative).mean()
