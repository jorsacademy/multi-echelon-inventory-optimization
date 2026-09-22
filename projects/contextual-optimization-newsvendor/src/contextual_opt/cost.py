import numpy as np
import torch


def critical_fractile(underage: float = 5.0, overage: float = 1.0) -> float:
    if underage <= 0 or overage <= 0:
        raise ValueError("underage and overage costs must be positive")
    return underage / (underage + overage)


def newsvendor_cost_numpy(
    quantity: np.ndarray,
    demand: np.ndarray,
    underage: float = 5.0,
    overage: float = 1.0,
) -> np.ndarray:
    quantity = np.asarray(quantity)
    demand = np.asarray(demand)
    return underage * np.maximum(demand - quantity, 0.0) + overage * np.maximum(
        quantity - demand,
        0.0,
    )


def newsvendor_cost_torch(
    quantity: torch.Tensor,
    demand: torch.Tensor,
    underage: float = 5.0,
    overage: float = 1.0,
) -> torch.Tensor:
    return underage * torch.relu(demand - quantity) + overage * torch.relu(quantity - demand)
