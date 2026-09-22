from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def finite_sample_quantile(values: np.ndarray, alpha: float) -> float:
    """Split-conformal 'higher' quantile using ceil((n+1)(1-alpha))."""
    v = np.sort(np.asarray(values, dtype=float).reshape(-1))
    n = len(v)
    if n < 1:
        raise ValueError("at least one calibration score is required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    index = min(max(rank - 1, 0), n - 1)
    return float(v[index])


def residual_scale_from_training(y_train: np.ndarray, pred_train: np.ndarray) -> np.ndarray:
    """Period scale estimated only from the proper-training split."""
    y = np.asarray(y_train, dtype=float)
    p = np.asarray(pred_train, dtype=float)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("training arrays must have identical [N,H] shape")
    return np.maximum(np.sqrt(np.mean((y-p)**2, axis=0)), 1.0)


@dataclass(frozen=True)
class SimultaneousConformalBox:
    scale: np.ndarray
    radius: float
    alpha: float

    @classmethod
    def calibrate(cls, y_cal: np.ndarray, pred_cal: np.ndarray, *, alpha: float = 0.10, scale: np.ndarray):
        y = np.asarray(y_cal, dtype=float)
        p = np.asarray(pred_cal, dtype=float)
        if y.shape != p.shape or y.ndim != 2:
            raise ValueError("calibration arrays must have identical [N,H] shape")
        scale = np.asarray(scale, dtype=float)
        if scale.shape != (y.shape[1],) or np.any(scale <= 0):
            raise ValueError("invalid conformal scale")
        scores = np.max(np.abs(y-p) / scale[None, :], axis=1)
        radius = finite_sample_quantile(scores, alpha)
        return cls(scale=scale, radius=radius, alpha=float(alpha))

    def bounds(self, prediction: np.ndarray):
        p = np.asarray(prediction, dtype=float)
        half = self.radius * self.scale
        lower = np.maximum(p - half, 0.0)
        upper = p + half
        return lower, upper


@dataclass(frozen=True)
class MarginalResidualBox:
    half_width: np.ndarray
    alpha: float

    @classmethod
    def calibrate(cls, y_cal, pred_cal, *, alpha=0.10):
        y = np.asarray(y_cal, dtype=float)
        p = np.asarray(pred_cal, dtype=float)
        if y.shape != p.shape or y.ndim != 2:
            raise ValueError("calibration arrays must match")
        widths = np.asarray([
            finite_sample_quantile(np.abs(y[:, t]-p[:, t]), alpha)
            for t in range(y.shape[1])
        ])
        return cls(widths, float(alpha))

    def bounds(self, prediction):
        p = np.asarray(prediction, dtype=float)
        return np.maximum(p-self.half_width, 0.0), p+self.half_width


@dataclass(frozen=True)
class ClassicalSigmaBox:
    half_width: np.ndarray
    sigma_multiplier: float

    @classmethod
    def fit(cls, y_train, pred_train, *, sigma_multiplier=2.0):
        resid = np.asarray(y_train, dtype=float)-np.asarray(pred_train, dtype=float)
        std = np.maximum(np.std(resid, axis=0, ddof=1), 1.0)
        return cls(std*float(sigma_multiplier), float(sigma_multiplier))

    def bounds(self, prediction):
        p = np.asarray(prediction, dtype=float)
        return np.maximum(p-self.half_width, 0.0), p+self.half_width


def simultaneous_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y = np.asarray(y)
    lo = np.asarray(lower)
    hi = np.asarray(upper)
    covered = np.all((y >= lo-1e-12) & (y <= hi+1e-12), axis=1)
    return float(np.mean(covered))


def marginal_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lower-1e-12) & (y <= upper+1e-12)))
