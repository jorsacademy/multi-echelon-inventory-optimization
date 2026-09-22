from __future__ import annotations

from dataclasses import replace

from chance_constrained_inventory import (
    InventoryParameters,
    analytical_solution,
    chance_constraint_lower_bound,
    expected_cost,
    service_probability,
)


def main() -> None:
    base = InventoryParameters()
    alphas = [0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99]

    header = (
        f"{'alpha':>8} {'chance_lb':>12} {'optimal_Q':>12} "
        f"{'service':>10} {'exp_cost':>12}"
    )
    print(header)
    print("-" * len(header))

    for alpha in alphas:
        params = replace(base, alpha=alpha)
        q = analytical_solution(params)
        lower_bound = chance_constraint_lower_bound(params)
        service = service_probability(q, params)
        cost = expected_cost(q, params)

        print(
            f"{alpha:8.3f} {lower_bound:12.4f} {q:12.4f} "
            f"{service:10.4f} {cost:12.4f}"
        )


if __name__ == "__main__":
    main()
