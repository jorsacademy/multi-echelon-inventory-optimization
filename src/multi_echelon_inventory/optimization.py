from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import pandas as pd

from .models import InventoryNode
from .monte_carlo import compare_policies
from .policies import PolicyType
from .simulation import MultiEchelonSystem


@dataclass(frozen=True)
class OptimizationResult:
    """Result of a simulation-based service-level parameter search."""

    candidates: pd.DataFrame
    best_policy: PolicyType | None
    best_service_level: float | None
    best_mean_total_cost: float | None
    optimized_nodes: tuple[InventoryNode, ...]

    @property
    def feasible(self) -> bool:
        return self.best_policy is not None

    def build_system(self) -> MultiEchelonSystem:
        if not self.feasible:
            raise ValueError("No feasible candidate is available")
        return MultiEchelonSystem(self.optimized_nodes)


def optimize_service_level(
    system: MultiEchelonSystem,
    service_levels: Sequence[float],
    runs: int = 50,
    periods: int = 365,
    base_seed: int | None = 42,
    policies: Iterable[PolicyType] | None = None,
    min_customer_fill_rate: float = 0.95,
    min_customer_cycle_service_level: float | None = None,
    confidence_level: float = 0.95,
) -> OptimizationResult:
    """Minimize simulated expected cost subject to service constraints.

    A common service-level parameter is applied to every echelon for each grid
    candidate. Each candidate/policy combination is evaluated by Monte Carlo.
    The lowest-mean-cost feasible combination is returned.

    This is simulation optimization by finite grid search, not an exact
    stochastic-programming solver. Increase ``runs`` and refine the candidate grid
    before treating small cost differences as meaningful.
    """
    if not service_levels:
        raise ValueError("service_levels must not be empty")
    levels = [float(level) for level in service_levels]
    if any(not 0 < level < 1 for level in levels):
        raise ValueError("All service levels must be strictly between 0 and 1")
    if len(levels) != len(set(levels)):
        raise ValueError("service_levels must not contain duplicates")
    if not 0 <= min_customer_fill_rate <= 1:
        raise ValueError("min_customer_fill_rate must be between 0 and 1")
    if (
        min_customer_cycle_service_level is not None
        and not 0 <= min_customer_cycle_service_level <= 1
    ):
        raise ValueError(
            "min_customer_cycle_service_level must be between 0 and 1 when set"
        )

    policy_list = list(policies) if policies is not None else list(PolicyType)
    if not policy_list:
        raise ValueError("At least one policy is required")

    rows: list[dict[str, float | str | bool]] = []
    node_sets: dict[tuple[str, float], tuple[InventoryNode, ...]] = {}

    for level in levels:
        candidate_nodes = tuple(
            replace(node, service_level=level) for node in system.nodes
        )
        candidate_system = MultiEchelonSystem(candidate_nodes)

        comparison = compare_policies(
            candidate_system,
            runs=runs,
            periods=periods,
            base_seed=base_seed,
            policies=policy_list,
            confidence_level=confidence_level,
        )

        for _, row in comparison.policy_summary.iterrows():
            policy_value = str(row["Policy"])
            fill_rate = float(row["Mean Customer Fill Rate"])
            cycle_service = float(row["Mean Customer Cycle Service Level"])
            feasible = fill_rate >= min_customer_fill_rate
            if min_customer_cycle_service_level is not None:
                feasible = feasible and (
                    cycle_service >= min_customer_cycle_service_level
                )

            rows.append(
                {
                    "Policy": policy_value,
                    "Service Level Parameter": level,
                    "Mean Total Cost": float(row["Mean Total Cost"]),
                    "Mean Total Cost CI Low": float(
                        row["Mean Total Cost CI Low"]
                    ),
                    "Mean Total Cost CI High": float(
                        row["Mean Total Cost CI High"]
                    ),
                    "Mean Customer Fill Rate": fill_rate,
                    "Mean Customer Cycle Service Level": cycle_service,
                    "Feasible": bool(feasible),
                }
            )
            node_sets[(policy_value, level)] = candidate_nodes

    candidates = pd.DataFrame.from_records(rows).sort_values(
        ["Feasible", "Mean Total Cost"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    feasible_rows = candidates[candidates["Feasible"]]
    if feasible_rows.empty:
        return OptimizationResult(
            candidates=candidates,
            best_policy=None,
            best_service_level=None,
            best_mean_total_cost=None,
            optimized_nodes=tuple(system.nodes),
        )

    best = feasible_rows.iloc[0]
    best_policy = PolicyType(str(best["Policy"]))
    best_level = float(best["Service Level Parameter"])
    return OptimizationResult(
        candidates=candidates,
        best_policy=best_policy,
        best_service_level=best_level,
        best_mean_total_cost=float(best["Mean Total Cost"]),
        optimized_nodes=node_sets[(best_policy.value, best_level)],
    )
