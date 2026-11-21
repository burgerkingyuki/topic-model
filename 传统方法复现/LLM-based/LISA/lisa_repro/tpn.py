from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .progress import progress


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_topics: int) -> None:
        super().__init__()
        dims = [input_dim] + hidden_dims + [num_topics]
        layers = []
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(0.1)])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TPNResult:
    doc_probs: np.ndarray
    topic_probs: np.ndarray


def _build_neighbor_indices(embeddings: np.ndarray, k: int) -> np.ndarray:
    if len(embeddings) < 2:
        return np.zeros((len(embeddings), 1), dtype=int)
    k = max(1, min(k, len(embeddings) - 1))
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)
    return np.argsort(-sims, axis=1)[:, :k]


def train_tpn(
    doc_embeddings: np.ndarray,
    topic_embeddings: np.ndarray,
    num_topics: int,
    hidden_dims: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    tau: float,
    alpha: float,
    mu_neighbors: int,
    seed: int = 42,
) -> TPNResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    doc_model = MLP(doc_embeddings.shape[1], hidden_dims, num_topics).to(device)
    topic_model = MLP(topic_embeddings.shape[1], hidden_dims, num_topics).to(device)
    optimizer = torch.optim.Adagrad(list(doc_model.parameters()) + list(topic_model.parameters()), lr=lr)
    doc_neighbors = _build_neighbor_indices(doc_embeddings, mu_neighbors)
    topic_neighbors = _build_neighbor_indices(topic_embeddings, mu_neighbors)

    doc_x = torch.tensor(doc_embeddings, dtype=torch.float32)
    topic_x = torch.tensor(topic_embeddings, dtype=torch.float32)
    dataset = TensorDataset(doc_x, topic_x)
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=True)

    def _mask_diag(mat: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(mat.size(0), device=mat.device, dtype=torch.bool)
        return mat.masked_fill(eye, float("-inf"))

    eps = 1e-9

    for _ in progress(range(epochs), total=epochs, desc="LiSA TPN training"):
        doc_model.train()
        topic_model.train()
        optimizer.zero_grad()

        doc_logits = doc_model(doc_x.to(device))
        topic_logits = topic_model(topic_x.to(device))
        doc_prob = torch.softmax(doc_logits, dim=-1)
        topic_prob = torch.softmax(topic_logits, dim=-1)

        pair_sim = torch.sum(doc_prob * topic_prob, dim=-1)
        loss_l1 = -torch.log(pair_sim.clamp_min(eps)).mean()

        doc_neighbor_choice = np.array([np.random.choice(doc_neighbors[i]) for i in range(len(doc_neighbors))], dtype=int)
        topic_neighbor_choice = np.array([np.random.choice(topic_neighbors[i]) for i in range(len(topic_neighbors))], dtype=int)
        doc_hat = doc_prob[torch.tensor(doc_neighbor_choice, device=device)]
        topic_hat = topic_prob[torch.tensor(topic_neighbor_choice, device=device)]

        sim_dt = torch.matmul(doc_prob, topic_hat.T) / tau
        sim_td = torch.matmul(topic_prob, doc_hat.T) / tau
        sim_tt = torch.matmul(topic_hat, topic_hat.T) / tau
        sim_dd = torch.matmul(doc_hat, doc_hat.T) / tau

        pos_dt = torch.diag(sim_dt)
        pos_td = torch.diag(sim_td)
        neg_dt = torch.logsumexp(_mask_diag(sim_dt), dim=1)
        neg_td = torch.logsumexp(_mask_diag(sim_td), dim=1)
        neg_tt = torch.logsumexp(_mask_diag(sim_tt), dim=1)
        neg_dd = torch.logsumexp(_mask_diag(sim_dd), dim=1)
        loss_l2_td = -(pos_dt - torch.logsumexp(torch.stack([neg_dt, neg_tt], dim=0), dim=0)).mean()
        loss_l2_dt = -(pos_td - torch.logsumexp(torch.stack([neg_td, neg_dd], dim=0), dim=0)).mean()
        loss_l2 = loss_l2_td + loss_l2_dt

        avg_doc = doc_prob.mean(dim=0)
        avg_topic = topic_prob.mean(dim=0)
        loss_l3 = -torch.sum(avg_doc * torch.log(avg_doc + eps)) - torch.sum(avg_topic * torch.log(avg_topic + eps))

        loss = loss_l1 + loss_l2 - alpha * loss_l3
        loss.backward()
        optimizer.step()

    doc_model.eval()
    topic_model.eval()
    with torch.no_grad():
        doc_probs = torch.softmax(doc_model(doc_x.to(device)), dim=-1).cpu().numpy()
        topic_probs = torch.softmax(topic_model(topic_x.to(device)), dim=-1).cpu().numpy()
    return TPNResult(doc_probs=doc_probs, topic_probs=topic_probs)
