from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .cost import critical_fractile, newsvendor_cost_torch


class LinearQuantilePolicy(nn.Module):
    """Predict a contextual newsvendor decision with linear quantile regression."""

    def __init__(self, n_features: int):
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be positive")
        self.linear = nn.Linear(n_features, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(features).squeeze(-1))


class DirectDecisionPolicy(nn.Module):
    """Learn a contextual order quantity directly from realized decision cost."""

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        if n_features < 1 or hidden < 1:
            raise ValueError("n_features and hidden must be positive")
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(features).squeeze(-1))


def train_quantile(
    model: LinearQuantilePolicy,
    features: torch.Tensor,
    demand: torch.Tensor,
    epochs: int = 300,
    learning_rate: float = 1e-2,
    underage: float = 5.0,
    overage: float = 1.0,
) -> LinearQuantilePolicy:
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    tau = critical_fractile(underage, overage)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad()
        quantity = model(features)
        residual = demand - quantity
        loss = torch.maximum(tau * residual, (tau - 1.0) * residual).mean()
        loss.backward()
        optimizer.step()
    return model


def train_direct(
    model: DirectDecisionPolicy,
    features: torch.Tensor,
    demand: torch.Tensor,
    epochs: int = 300,
    learning_rate: float = 1e-2,
    underage: float = 5.0,
    overage: float = 1.0,
) -> DirectDecisionPolicy:
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad()
        quantity = model(features)
        loss = newsvendor_cost_torch(quantity, demand, underage, overage).mean()
        loss.backward()
        optimizer.step()
    return model


def knn_saa(
    train_features: np.ndarray,
    train_demand: np.ndarray,
    test_features: np.ndarray,
    k: int = 30,
    underage: float = 5.0,
    overage: float = 1.0,
) -> np.ndarray:
    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("feature arrays must be two-dimensional")
    if train_features.shape[0] != train_demand.shape[0]:
        raise ValueError("training features and demand must have matching rows")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("train and test feature dimensions must match")
    if not 1 <= k <= len(train_demand):
        raise ValueError("k must be between 1 and the training sample size")

    tau = critical_fractile(underage, overage)
    decisions = []
    for context in test_features:
        squared_distance = ((train_features - context) ** 2).sum(axis=1)
        indices = np.argpartition(squared_distance, k - 1)[:k]
        decisions.append(np.quantile(train_demand[indices], tau))
    return np.asarray(decisions)
