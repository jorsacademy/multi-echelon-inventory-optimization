from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DemandDataset:
    features: np.ndarray
    demand: np.ndarray

    def __post_init__(self):
        x = np.asarray(self.features, dtype=float)
        y = np.asarray(self.demand, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("features and demand must be 2D with equal sample count")
        if np.any(y < 0):
            raise ValueError("demand must be nonnegative")

    @property
    def horizon(self) -> int:
        return int(self.demand.shape[1])


def _conditional_mean(features: np.ndarray, horizon: int) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    base = 48.0 + 18.0*x[:, 0] + 7.0*x[:, 1]
    promo = 1.0 + 0.28*np.maximum(x[:, 2], 0.0)
    trend = 6.0*np.tanh(1.3*x[:, 3])
    phase = 2*np.pi*x[:, 4]
    macro = 5.0*x[:, 5]*x[:, 0]
    channel = 4.0*np.sin(1.7*x[:, 6] + 0.5*x[:, 1])

    out = []
    for t in range(horizon):
        season = 8.0*np.sin(phase + 2*np.pi*t/horizon)
        local_trend = trend*(t/(max(horizon-1, 1)) - 0.35)
        nonlinear_promo = 4.0*np.maximum(x[:, 2]-0.25, 0.0) * np.cos(0.7*t)
        mean = (base + season + local_trend + macro + channel + nonlinear_promo) * promo
        out.append(mean)
    return np.maximum(np.stack(out, axis=1), 2.0)


def sample_demand_given_features(features: np.ndarray, *, seed: int, horizon: int = 6) -> np.ndarray:
    """Sample correlated, heteroskedastic demand trajectories conditional on context."""
    rng = np.random.default_rng(seed)
    x = np.asarray(features, dtype=float)
    mean = _conditional_mean(x, horizon)

    n = len(x)
    common = rng.normal(0.0, 1.0, size=(n, 1))
    idio = rng.normal(0.0, 1.0, size=(n, horizon))
    corr = np.zeros_like(idio)
    corr[:, 0] = idio[:, 0]
    for t in range(1, horizon):
        corr[:, t] = 0.62*corr[:, t-1] + np.sqrt(1.0-0.62**2)*idio[:, t]

    scale = (
        3.8
        + 2.2*np.abs(x[:, 1:2])
        + 3.0*np.maximum(x[:, 2:3], 0.0)
        + 1.5*np.abs(x[:, 5:6])
    )
    period_scale = np.linspace(0.90, 1.18, horizon)[None, :]
    noise = scale * period_scale * (0.55*common + 0.85*corr)

    surge_flag = rng.random((n, 1)) < (0.04 + 0.05*np.maximum(x[:, 2:3], 0.0))
    surge_period = rng.integers(0, horizon, size=n)
    surge = np.zeros((n, horizon))
    for i in range(n):
        if surge_flag[i, 0]:
            t = surge_period[i]
            surge[i, t] = rng.uniform(12.0, 28.0)
            if t + 1 < horizon:
                surge[i, t+1] = 0.45*surge[i, t]

    demand = mean + noise + surge
    return np.maximum(demand, 0.0)


def generate_dataset(n_samples: int, *, seed: int, horizon: int = 6) -> DemandDataset:
    rng = np.random.default_rng(seed)
    x = np.column_stack([
        rng.uniform(0.0, 1.0, n_samples),
        rng.normal(0.0, 1.0, n_samples),
        rng.uniform(0.0, 1.0, n_samples),
        rng.normal(0.0, 1.0, n_samples),
        rng.uniform(0.0, 1.0, n_samples),
        rng.normal(0.0, 1.0, n_samples),
        rng.normal(0.0, 1.0, n_samples),
    ]).astype(np.float64)
    y = sample_demand_given_features(x, seed=seed + 9_999_991, horizon=horizon)
    return DemandDataset(x, y)
