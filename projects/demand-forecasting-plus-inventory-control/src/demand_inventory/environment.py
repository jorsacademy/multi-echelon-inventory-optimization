"""Finite-horizon single-item inventory environment with stochastic demand."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .forecasting import SeasonalForecast


@dataclass(frozen=True)
class InventoryConfig:
    horizon: int = 180
    lead_time: int = 2
    max_inventory: float = 250.0
    max_order: float = 120.0
    initial_inventory: float = 80.0
    initial_backlog: float = 0.0
    base_demand: float = 20.0
    weekly_amplitude: float = 7.0
    trend_per_day: float = 0.025
    demand_noise_std: float = 4.0
    holding_cost: float = 0.35
    backlog_cost: float = 2.5
    order_cost: float = 0.12
    fixed_order_cost: float = 1.5


class InventoryControlEnv(gym.Env):
    """Continuous replenishment environment with forecast features in the state."""

    metadata = {"render_modes": []}

    def __init__(self, config: InventoryConfig | None = None):
        super().__init__()
        self.config = config or InventoryConfig()
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32) * 2.0,
            dtype=np.float32,
        )
        self.forecaster = SeasonalForecast()
        self._rng = np.random.default_rng()
        self._demand = np.empty(0)
        self._history: list[float] = []
        self._pipeline: list[float] = []
        self.inventory = 0.0
        self.backlog = 0.0
        self.day = 0
        self.total_cost = 0.0
        self.total_demand = 0.0
        self.total_fulfilled = 0.0
        self.total_ordered = 0.0
        self.stockout_days = 0

    def _generate_demand(self) -> np.ndarray:
        c = self.config
        t = np.arange(c.horizon, dtype=float)
        weekly = c.weekly_amplitude * np.sin(2.0 * np.pi * t / 7.0)
        trend = c.trend_per_day * t
        noise = self._rng.normal(0.0, c.demand_noise_std, size=c.horizon)
        return np.maximum(c.base_demand + weekly + trend + noise, 0.0)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._demand = self._generate_demand()
        self._history = []
        self._pipeline = [0.0 for _ in range(self.config.lead_time)]
        self.inventory = self.config.initial_inventory
        self.backlog = self.config.initial_backlog
        self.day = 0
        self.total_cost = 0.0
        self.total_demand = 0.0
        self.total_fulfilled = 0.0
        self.total_ordered = 0.0
        self.stockout_days = 0
        return self._observation(), {}

    def _forecast(self):
        history = np.asarray(self._history, dtype=float)
        if history.size == 0:
            return self.config.base_demand, self.config.demand_noise_std
        result = self.forecaster.predict(history)
        return result.mean, result.std

    def _observation(self) -> np.ndarray:
        c = self.config
        forecast_mean, forecast_std = self._forecast()
        pipeline_total = float(sum(self._pipeline))
        time_fraction = self.day / max(c.horizon - 1, 1)
        demand_scale = max(2.0 * c.base_demand, 1.0)
        return np.array(
            [
                self.inventory / c.max_inventory,
                min(self.backlog / c.max_inventory, 2.0),
                min(pipeline_total / (c.max_order * max(c.lead_time, 1)), 2.0),
                min(forecast_mean / demand_scale, 2.0),
                min(forecast_std / max(c.base_demand, 1.0), 2.0),
                min((self._history[-1] if self._history else c.base_demand) / demand_scale, 2.0),
                min(time_fraction, 1.0),
            ],
            dtype=np.float32,
        )

    def step(self, action):
        c = self.config
        order_qty = float(np.clip(np.asarray(action, dtype=float)[0], 0.0, 1.0) * c.max_order)

        arrival = self._pipeline.pop(0) if self._pipeline else 0.0
        self._pipeline.append(order_qty)
        self.inventory = min(self.inventory + arrival, c.max_inventory)

        demand = float(self._demand[self.day])
        self._history.append(demand)
        self.total_demand += demand

        required = demand + self.backlog
        fulfilled = min(self.inventory, required)
        self.inventory -= fulfilled
        self.backlog = required - fulfilled
        self.total_fulfilled += fulfilled
        self.total_ordered += order_qty

        if self.backlog > 1e-9:
            self.stockout_days += 1

        holding = c.holding_cost * self.inventory
        backlog_cost = c.backlog_cost * self.backlog
        ordering = c.order_cost * order_qty + (c.fixed_order_cost if order_qty > 1e-9 else 0.0)
        step_cost = holding + backlog_cost + ordering
        self.total_cost += step_cost

        reward = -step_cost
        self.day += 1
        truncated = self.day >= c.horizon
        terminated = False

        service_level = 1.0 if self.total_demand <= 1e-9 else self.total_fulfilled / self.total_demand
        fill_rate = service_level
        info = {
            "demand": demand,
            "fulfilled": fulfilled,
            "inventory": self.inventory,
            "backlog": self.backlog,
            "order_qty": order_qty,
            "holding_cost": holding,
            "backlog_cost": backlog_cost,
            "ordering_cost": ordering,
            "total_cost": self.total_cost,
            "service_level": service_level,
            "fill_rate": fill_rate,
            "stockout_days": self.stockout_days,
            "total_ordered": self.total_ordered,
        }
        return self._observation(), reward, terminated, truncated, info
