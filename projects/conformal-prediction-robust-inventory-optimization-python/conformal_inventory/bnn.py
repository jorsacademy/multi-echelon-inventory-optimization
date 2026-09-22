from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BayesianLinear(nn.Module):
    """Mean-field Gaussian linear layer with an analytic Gaussian KL term."""

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("layer dimensions must be positive")
        if not math.isfinite(prior_sigma) or prior_sigma <= 0:
            raise ValueError("prior_sigma must be finite and positive")

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0.0, 0.08))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))
        self.register_buffer("prior_sigma", torch.tensor(float(prior_sigma)))

    @staticmethod
    def scale(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + torch.finfo(rho.dtype).eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_mu + self.scale(self.weight_rho) * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + self.scale(self.bias_rho) * torch.randn_like(self.bias_mu)
        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        kl = self.weight_mu.new_zeros(())
        prior_var = self.prior_sigma.square()
        for mu, rho in (
            (self.weight_mu, self.weight_rho),
            (self.bias_mu, self.bias_rho),
        ):
            sigma = self.scale(rho)
            kl = kl + (
                torch.log(self.prior_sigma / sigma)
                + (sigma.square() + mu.square()) / (2.0 * prior_var)
                - 0.5
            ).sum()
        return kl


class BayesianNeuralNetwork(nn.Module):
    """Small multi-output BNN for contextual demand regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (32, 32),
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        hidden_dims = tuple(int(h) for h in hidden_dims)
        if any(h < 1 for h in hidden_dims):
            raise ValueError("hidden dimensions must be positive")

        sizes = (int(input_dim),) + hidden_dims + (int(output_dim),)
        self.layers = nn.ModuleList(
            BayesianLinear(sizes[i], sizes[i + 1], prior_sigma)
            for i in range(len(sizes) - 1)
        )

    def forward(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")

        draws = []
        for _ in range(num_samples):
            h = x
            for i, layer in enumerate(self.layers):
                h = layer(h)
                if i < len(self.layers) - 1:
                    h = torch.tanh(h)
            draws.append(h)
        return torch.stack(draws, dim=0)

    def kl_divergence(self) -> torch.Tensor:
        zero = self.layers[0].weight_mu.new_zeros(())
        return sum((layer.kl_divergence() for layer in self.layers), zero)


def negative_elbo(
    draws: torch.Tensor,
    targets: torch.Tensor,
    kl: torch.Tensor,
    *,
    dataset_size: int,
    noise_std: float,
) -> torch.Tensor:
    """Per-row MC negative ELBO with Gaussian observation likelihood.

    The likelihood is evaluated separately for each posterior weight draw and only
    then averaged. This avoids the common error of evaluating the likelihood at
    the mean prediction. KL is normalized by the full training-set size.
    """

    if dataset_size < 1 or dataset_size < targets.shape[0]:
        raise ValueError("dataset_size must cover the current batch")
    if not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be finite and positive")
    if draws.ndim != 3 or targets.ndim != 2 or draws.shape[1:] != targets.shape:
        raise ValueError("draws and targets have incompatible shapes")

    log_prob = torch.distributions.Normal(draws, noise_std).log_prob(targets.unsqueeze(0))
    nll = -log_prob.sum(dim=-1).mean()
    return nll + kl / dataset_size


@dataclass
class BayesianDemandPredictor:
    """Scikit-learn-like BNN wrapper used by the inventory pipeline.

    Inputs and outputs are standardized using training data. The public predict
    method returns posterior-mean demand on the original scale, matching the
    DemandPredictor interface used by the conformal and optimization layers.
    """

    model: BayesianNeuralNetwork
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    noise_std: float
    prediction_samples: int = 300
    prediction_seed: int = 1_000_003

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        demand: np.ndarray,
        *,
        seed: int = 42,
        hidden_dims: Sequence[int] = (32, 32),
        prior_sigma: float = 1.0,
        noise_std: float = 0.35,
        learning_rate: float = 0.01,
        steps: int = 600,
        train_samples: int = 3,
        prediction_samples: int = 300,
    ) -> "BayesianDemandPredictor":
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(demand, dtype=np.float32)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("features and demand must be 2D with equal sample count")
        if steps < 1 or train_samples < 1 or prediction_samples < 2:
            raise ValueError("steps/train_samples must be positive and prediction_samples >= 2")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if np.any(y < 0):
            raise ValueError("demand must be nonnegative")

        np.random.seed(seed)
        torch.manual_seed(seed)

        x_mean = x.mean(axis=0, keepdims=True)
        x_scale = x.std(axis=0, keepdims=True)
        x_scale = np.where(x_scale < 1e-6, 1.0, x_scale).astype(np.float32)

        y_mean = y.mean(axis=0, keepdims=True)
        y_scale = y.std(axis=0, keepdims=True)
        y_scale = np.where(y_scale < 1e-6, 1.0, y_scale).astype(np.float32)

        x_t = torch.as_tensor((x - x_mean) / x_scale, dtype=torch.float32)
        y_t = torch.as_tensor((y - y_mean) / y_scale, dtype=torch.float32)

        model = BayesianNeuralNetwork(
            input_dim=x.shape[1],
            output_dim=y.shape[1],
            hidden_dims=hidden_dims,
            prior_sigma=prior_sigma,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for _ in range(int(steps)):
            optimizer.zero_grad()
            draws = model(x_t, num_samples=int(train_samples))
            loss = negative_elbo(
                draws,
                y_t,
                model.kl_divergence(),
                dataset_size=len(x),
                noise_std=noise_std,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        return cls(
            model=model,
            x_mean=x_mean.astype(np.float32),
            x_scale=x_scale.astype(np.float32),
            y_mean=y_mean.astype(np.float32),
            y_scale=y_scale.astype(np.float32),
            noise_std=float(noise_std),
            prediction_samples=int(prediction_samples),
            prediction_seed=int(seed) + 1_000_003,
        )

    def _standardized_features(self, features: np.ndarray) -> torch.Tensor:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.x_mean.shape[1]:
            raise ValueError("features have incompatible shape")
        return torch.as_tensor((x - self.x_mean) / self.x_scale, dtype=torch.float32)

    @torch.no_grad()
    def posterior_samples(
        self,
        features: np.ndarray,
        *,
        num_samples: int | None = None,
    ) -> np.ndarray:
        samples = self.prediction_samples if num_samples is None else int(num_samples)
        if samples < 2:
            raise ValueError("num_samples must be at least two")
        self.model.eval()
        x_t = self._standardized_features(features)
        with torch.random.fork_rng():
            torch.manual_seed(self.prediction_seed)
            draws = self.model(x_t, num_samples=samples).cpu().numpy()
        draws = draws * self.y_scale[None, :, :] + self.y_mean[None, :, :]
        return np.maximum(draws, 0.0)

    def predict(
        self,
        features: np.ndarray,
        *,
        num_samples: int | None = None,
    ) -> np.ndarray:
        return self.posterior_samples(features, num_samples=num_samples).mean(axis=0)

    def epistemic_std(
        self,
        features: np.ndarray,
        *,
        num_samples: int | None = None,
    ) -> np.ndarray:
        return self.posterior_samples(features, num_samples=num_samples).std(axis=0, ddof=0)
