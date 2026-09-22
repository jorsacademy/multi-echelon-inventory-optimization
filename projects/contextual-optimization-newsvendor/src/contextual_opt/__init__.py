"""Contextual optimization research sandbox."""

from .cost import critical_fractile, newsvendor_cost_numpy, newsvendor_cost_torch
from .data import generate_contextual_demand
from .policies import (
    DirectDecisionPolicy,
    LinearQuantilePolicy,
    knn_saa,
    train_direct,
    train_quantile,
)

__all__ = [
    "DirectDecisionPolicy",
    "LinearQuantilePolicy",
    "critical_fractile",
    "generate_contextual_demand",
    "knn_saa",
    "newsvendor_cost_numpy",
    "newsvendor_cost_torch",
    "train_direct",
    "train_quantile",
]
