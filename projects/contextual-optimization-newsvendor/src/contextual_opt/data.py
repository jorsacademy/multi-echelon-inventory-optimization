from __future__ import annotations

import numpy as np


def generate_contextual_demand(
    n_samples: int = 500,
    n_features: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if not 1 <= n_features <= 3:
        raise ValueError("n_features must be between 1 and 3")

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, n_features))
    beta = np.array([5.0, -3.0, 2.0])[:n_features]
    conditional_mean = 20.0 + features @ beta
    conditional_scale = 2.0 + 1.5 / (1.0 + np.exp(-features[:, 0]))
    demand = np.maximum(
        0.0,
        conditional_mean + rng.normal(scale=conditional_scale),
    )
    return features.astype("float32"), demand.astype("float32")
