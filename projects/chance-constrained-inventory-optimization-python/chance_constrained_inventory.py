from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass(frozen=True)
class InventoryParameters:
    mu: float = 100.0
    sigma: float = 20.0
    holding_cost: float = 2.0
    shortage_cost: float = 5.0
    alpha: float = 0.95

    def validate(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.holding_cost < 0 or self.shortage_cost < 0:
            raise ValueError("cost parameters must be nonnegative")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1")


def expected_overage(Q: float, params: InventoryParameters) -> float:
    z = (Q - params.mu) / params.sigma
    return params.sigma * norm.pdf(z) + (Q - params.mu) * norm.cdf(z)


def expected_shortage(Q: float, params: InventoryParameters) -> float:
    z = (Q - params.mu) / params.sigma
    return params.sigma * norm.pdf(z) + (params.mu - Q) * (1.0 - norm.cdf(z))


def expected_cost(Q: float, params: InventoryParameters) -> float:
    return (
        params.holding_cost * expected_overage(Q, params)
        + params.shortage_cost * expected_shortage(Q, params)
    )


def chance_constraint_lower_bound(params: InventoryParameters) -> float:
    return params.mu + params.sigma * norm.ppf(params.alpha)


def unconstrained_newsvendor_quantity(params: InventoryParameters) -> float:
    denominator = params.shortage_cost + params.holding_cost
    if denominator == 0:
        return 0.0
    critical_fractile = params.shortage_cost / denominator
    if critical_fractile <= 0:
        return 0.0
    if critical_fractile >= 1:
        return np.inf
    return params.mu + params.sigma * norm.ppf(critical_fractile)


def analytical_solution(params: InventoryParameters) -> float:
    params.validate()
    return max(
        0.0,
        unconstrained_newsvendor_quantity(params),
        chance_constraint_lower_bound(params),
    )


def solve_numerically(params: InventoryParameters):
    params.validate()
    lower_bound = chance_constraint_lower_bound(params)
    initial = max(params.mu, lower_bound)

    def objective(x: np.ndarray) -> float:
        return expected_cost(float(x[0]), params)

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: float(x[0]) - lower_bound,
        }
    ]

    result = minimize(
        objective,
        x0=np.array([initial], dtype=float),
        method="SLSQP",
        bounds=[(0.0, None)],
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return result


def service_probability(Q: float, params: InventoryParameters) -> float:
    return float(norm.cdf((Q - params.mu) / params.sigma))


def main() -> None:
    params = InventoryParameters()

    numerical = solve_numerically(params)
    q_numerical = float(numerical.x[0])
    q_analytical = analytical_solution(params)
    q_unconstrained = unconstrained_newsvendor_quantity(params)
    q_service = chance_constraint_lower_bound(params)

    print(f"Unconstrained newsvendor quantity: {q_unconstrained:.4f}")
    print(f"Chance-constraint lower bound:    {q_service:.4f}")
    print(f"Analytical constrained optimum:   {q_analytical:.4f}")
    print(f"Numerical constrained optimum:    {q_numerical:.4f}")
    print(f"Expected cost at optimum:         {expected_cost(q_numerical, params):.4f}")
    print(f"Service probability:              {service_probability(q_numerical, params):.6f}")
    print(f"Required service probability:     {params.alpha:.6f}")
    print(f"Absolute analytical gap:          {abs(q_numerical - q_analytical):.8f}")


if __name__ == "__main__":
    main()
