from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


@dataclass
class DemandPredictor:
    model: ExtraTreesRegressor

    @classmethod
    def fit(cls, features: np.ndarray, demand: np.ndarray, *, seed: int = 42, n_estimators: int = 240):
        model = ExtraTreesRegressor(
            n_estimators=int(n_estimators),
            random_state=int(seed),
            min_samples_leaf=3,
            max_features=0.9,
            n_jobs=1,
        )
        model.fit(np.asarray(features, dtype=float), np.asarray(demand, dtype=float))
        return cls(model)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(self.model.predict(features), dtype=float), 0.0)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
