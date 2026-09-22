"""Small, dependency-light forecasting models used by the benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastResult:
    mean: float
    std: float


class NaiveForecast:
    """Forecast next demand from the recent moving average."""

    def __init__(self, window: int = 7):
        self.window = window

    def predict(self, history: np.ndarray) -> ForecastResult:
        values = np.asarray(history, dtype=float)
        if values.size == 0:
            return ForecastResult(20.0, 5.0)
        recent = values[-self.window :]
        return ForecastResult(float(recent.mean()), float(max(recent.std(ddof=0), 1.0)))


class SeasonalForecast:
    """Weekly seasonal forecast with fallback to a moving average."""

    def __init__(self, season_length: int = 7, fallback_window: int = 7):
        self.season_length = season_length
        self.fallback = NaiveForecast(fallback_window)

    def predict(self, history: np.ndarray) -> ForecastResult:
        values = np.asarray(history, dtype=float)
        if values.size < self.season_length:
            return self.fallback.predict(values)
        seasonal_points = values[-self.season_length :: -self.season_length]
        if seasonal_points.size == 0:
            return self.fallback.predict(values)
        mean = float(seasonal_points.mean())
        std = float(max(seasonal_points.std(ddof=0), 1.0))
        return ForecastResult(mean, std)


def rolling_mae(series: np.ndarray, model, warmup: int = 14) -> float:
    values = np.asarray(series, dtype=float)
    if values.size <= warmup:
        raise ValueError("series must contain more observations than warmup")
    errors = []
    for t in range(warmup, values.size):
        pred = model.predict(values[:t]).mean
        errors.append(abs(pred - values[t]))
    return float(np.mean(errors))
