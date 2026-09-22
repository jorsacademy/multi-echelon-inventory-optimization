from __future__ import annotations

import numpy as np

from chance_constrained_inventory import (
    InventoryParameters,
    analytical_solution,
    expected_cost,
)


def monte_carlo_validate(
    params: InventoryParameters,
    n_samples: int = 1_000_000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    demand = rng.normal(params.mu, params.sigma, size=n_samples)

    q = analytical_solution(params)
    leftovers = np.maximum(q - demand, 0.0)
    shortages = np.maximum(demand - q, 0.0)

    empirical_service = float(np.mean(demand <= q))
    empirical_cost = float(
        params.holding_cost * leftovers.mean()
        + params.shortage_cost * shortages.mean()
    )

    return {
        "Q": q,
        "empirical_service_probability": empirical_service,
        "required_service_probability": params.alpha,
        "empirical_expected_cost": empirical_cost,
        "analytical_expected_cost": expected_cost(q, params),
    }


def main() -> None:
    params = InventoryParameters()
    results = monte_carlo_validate(params)

    print(f"Order quantity:                  {results['Q']:.4f}")
    print(
        "Empirical service probability: "
        f"{results['empirical_service_probability']:.6f}"
    )
    print(
        "Required service probability:  "
        f"{results['required_service_probability']:.6f}"
    )
    print(
        "Empirical expected cost:        "
        f"{results['empirical_expected_cost']:.4f}"
    )
    print(
        "Analytical expected cost:       "
        f"{results['analytical_expected_cost']:.4f}"
    )


if __name__ == "__main__":
    main()
