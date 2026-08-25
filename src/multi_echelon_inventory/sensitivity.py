from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import pandas as pd

from .models import InventoryNode
from .monte_carlo import compare_policies
from .policies import PolicyType
from .simulation import MultiEchelonSystem


@dataclass(frozen=True)
class SensitivityResult:
    """Scenario-level Monte Carlo sensitivity-analysis results."""

    scenarios: pd.DataFrame
    policy_results: pd.DataFrame


_SUPPORTED_PARAMETERS = {
    "demand_mean",
    "demand_std",
    "lead_time",
    "lead_time_std",
    "holding_cost",
    "shortage_cost",
    "ordering_cost",
    "shipment_capacity",
    "order_capacity",
    "production_capacity",
}


def _scaled_node(node: InventoryNode, parameter: str, multiplier: float) -> InventoryNode:
    value = getattr(node, parameter)
    if value is None:
        return node

    scaled = float(value) * multiplier
    if parameter == "lead_time":
        scaled = max(0, int(round(scaled)))
    elif parameter in {"demand_mean", "demand_std", "lead_time_std", "ordering_cost"}:
        scaled = max(0.0, scaled)
    else:
        scaled = max(1e-12, scaled)
    return replace(node, **{parameter: scaled})


def sensitivity_analysis(
    system: MultiEchelonSystem,
    parameter: str,
    multipliers: Sequence[float],
    runs: int = 50,
    periods: int = 365,
    base_seed: int | None = 42,
    policies: Iterable[PolicyType] | None = None,
    node_name: str | None = None,
    confidence_level: float = 0.95,
) -> SensitivityResult:
    """Evaluate policy robustness as one model parameter changes.

    ``multipliers`` are applied either to every node or to ``node_name`` only.
    Existing ``None`` capacity values remain unconstrained rather than being
    converted into numeric limits.
    """
    if parameter not in _SUPPORTED_PARAMETERS:
        raise ValueError(
            f"Unsupported parameter {parameter!r}; choose from "
            f"{sorted(_SUPPORTED_PARAMETERS)}"
        )
    if not multipliers:
        raise ValueError("multipliers must not be empty")
    factors = [float(value) for value in multipliers]
    if any(value <= 0 for value in factors):
        raise ValueError("All multipliers must be greater than zero")
    if len(factors) != len(set(factors)):
        raise ValueError("multipliers must not contain duplicates")

    if node_name is not None and node_name not in {node.name for node in system.nodes}:
        raise ValueError(f"Unknown node_name: {node_name}")

    policy_frames: list[pd.DataFrame] = []
    scenario_rows: list[dict[str, float | str]] = []

    for multiplier in factors:
        scenario_nodes = tuple(
            _scaled_node(node, parameter, multiplier)
            if node_name is None or node.name == node_name
            else node
            for node in system.nodes
        )
        scenario_system = MultiEchelonSystem(scenario_nodes)
        comparison = compare_policies(
            scenario_system,
            runs=runs,
            periods=periods,
            base_seed=base_seed,
            policies=policies,
            confidence_level=confidence_level,
        )

        summary = comparison.policy_summary.copy()
        summary.insert(0, "Parameter", parameter)
        summary.insert(1, "Multiplier", multiplier)
        summary.insert(2, "Node", node_name or "ALL")
        policy_frames.append(summary)

        best = summary.iloc[0]
        scenario_rows.append(
            {
                "Parameter": parameter,
                "Multiplier": multiplier,
                "Node": node_name or "ALL",
                "Best Policy": str(best["Policy"]),
                "Best Mean Total Cost": float(best["Mean Total Cost"]),
                "Best Mean Customer Fill Rate": float(
                    best["Mean Customer Fill Rate"]
                ),
                "Best Mean Customer Cycle Service Level": float(
                    best["Mean Customer Cycle Service Level"]
                ),
            }
        )

    return SensitivityResult(
        scenarios=pd.DataFrame.from_records(scenario_rows),
        policy_results=pd.concat(policy_frames, ignore_index=True),
    )
