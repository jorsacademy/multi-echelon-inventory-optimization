"""Classical inventory-control policies."""

from __future__ import annotations

import numpy as np

from .environment import InventoryConfig


class SSInventoryPolicy:
    """Continuous-action approximation of an (s, S) replenishment policy."""

    def __init__(self, config: InventoryConfig, reorder_point: float = 55.0, order_up_to: float = 130.0):
        self.config = config
        self.reorder_point = reorder_point
        self.order_up_to = order_up_to

    def act(self, observation: np.ndarray) -> np.ndarray:
        inventory = float(observation[0]) * self.config.max_inventory
        backlog = float(observation[1]) * self.config.max_inventory
        pipeline = float(observation[2]) * self.config.max_order * max(self.config.lead_time, 1)
        inventory_position = inventory + pipeline - backlog
        qty = 0.0 if inventory_position > self.reorder_point else max(self.order_up_to - inventory_position, 0.0)
        return np.array([min(qty / self.config.max_order, 1.0)], dtype=np.float32)


class ForecastOrderUpToPolicy:
    """Order-up-to heuristic using forecast mean and uncertainty."""

    def __init__(self, config: InventoryConfig, safety_factor: float = 1.5, review_buffer: int = 1):
        self.config = config
        self.safety_factor = safety_factor
        self.review_buffer = review_buffer

    def act(self, observation: np.ndarray) -> np.ndarray:
        c = self.config
        inventory = float(observation[0]) * c.max_inventory
        backlog = float(observation[1]) * c.max_inventory
        pipeline = float(observation[2]) * c.max_order * max(c.lead_time, 1)
        forecast_mean = float(observation[3]) * 2.0 * c.base_demand
        forecast_std = float(observation[4]) * c.base_demand
        protection = c.lead_time + self.review_buffer
        target = protection * forecast_mean + self.safety_factor * np.sqrt(max(protection, 1)) * forecast_std
        inventory_position = inventory + pipeline - backlog
        qty = max(target - inventory_position, 0.0)
        return np.array([min(qty / c.max_order, 1.0)], dtype=np.float32)


class ConstantOrderPolicy:
    def __init__(self, config: InventoryConfig, quantity: float = 20.0):
        self.config = config
        self.quantity = quantity

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.array([min(self.quantity / self.config.max_order, 1.0)], dtype=np.float32)
