from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InventoryConfig:
    horizon: int = 60
    demand_rate: float = 4.0
    initial_inventory: int = 8
    holding_cost: float = 0.4
    order_cost: float = 1.0
    setup_cost: float = 8.0
    lost_sales_cost: float = 6.0

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.demand_rate <= 0:
            raise ValueError("demand_rate must be positive")
        if self.initial_inventory < 0:
            raise ValueError("initial_inventory must be nonnegative")
        for value in (
            self.holding_cost,
            self.order_cost,
            self.setup_cost,
            self.lost_sales_cost,
        ):
            if value < 0:
                raise ValueError("cost parameters must be nonnegative")


def simulate_policy(
    reorder_point: int,
    order_up_to: int,
    *,
    seed: int,
    config: InventoryConfig | None = None,
) -> float:
    """Return average per-period cost for an `(s, S)` lost-sales policy."""
    cfg = config or InventoryConfig()
    if reorder_point < 0:
        raise ValueError("reorder_point must be nonnegative")
    if order_up_to <= reorder_point:
        raise ValueError("order_up_to must exceed reorder_point")

    rng = np.random.default_rng(seed)
    inventory = cfg.initial_inventory
    total_cost = 0.0

    for _ in range(cfg.horizon):
        if inventory <= reorder_point:
            quantity = order_up_to - inventory
            total_cost += cfg.setup_cost + cfg.order_cost * quantity
            inventory = order_up_to

        demand = int(rng.poisson(cfg.demand_rate))
        sales = min(inventory, demand)
        lost_sales = demand - sales
        inventory -= sales
        total_cost += cfg.holding_cost * inventory + cfg.lost_sales_cost * lost_sales

    return total_cost / cfg.horizon


def evaluate_policy(
    reorder_point: int,
    order_up_to: int,
    *,
    seeds: tuple[int, ...] = (101, 202, 303, 404),
    config: InventoryConfig | None = None,
) -> float:
    if not seeds:
        raise ValueError("seeds must not be empty")
    costs = [
        simulate_policy(reorder_point, order_up_to, seed=seed, config=config)
        for seed in seeds
    ]
    return float(np.mean(costs))
