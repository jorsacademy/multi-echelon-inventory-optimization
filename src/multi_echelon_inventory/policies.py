from __future__ import annotations

from enum import Enum
from math import sqrt
from statistics import NormalDist
from typing import Sequence

from .models import InventoryNode


class PolicyType(str, Enum):
    BASE_STOCK = "base_stock"
    CRITICAL_RATIO = "critical_ratio"
    ECHELON_BASE_STOCK = "echelon_base_stock"


def base_stock_target(node: InventoryNode) -> float:
    """Return a service-level base-stock target for a single node."""
    return node.reorder_point()


def critical_ratio_target(node: InventoryNode) -> float:
    """Return a newsvendor-style lead-time demand quantile target."""
    ratio = node.shortage_cost / (node.shortage_cost + node.holding_cost)
    z = NormalDist().inv_cdf(ratio)
    mean = node.lead_time_demand_mean()
    std = node.lead_time_demand_std()
    return max(0.0, mean + z * std)


def echelon_base_stock_targets(nodes: Sequence[InventoryNode]) -> dict[str, float]:
    """Return echelon-aware targets for a serial network.

    The calculation aggregates downstream demand moments and cumulative lead-time
    exposure. It is intentionally transparent and should be treated as a heuristic,
    not as an exact Clark-Scarf dynamic-programming solution.
    """
    targets: dict[str, float] = {}
    node_list = list(nodes)

    for i, node in enumerate(node_list):
        downstream = node_list[i:]
        cumulative_lead_time = sum(n.lead_time for n in downstream)
        aggregate_mean = sum(n.demand_mean for n in downstream)
        aggregate_variance = sum(n.demand_std ** 2 for n in downstream)

        lead_time_mean = aggregate_mean * cumulative_lead_time
        lead_time_std = sqrt(aggregate_variance * cumulative_lead_time)
        target = lead_time_mean + node.service_z * lead_time_std
        targets[node.name] = max(0.0, target)

    return targets


def policy_targets(
    nodes: Sequence[InventoryNode], policy: PolicyType
) -> dict[str, float]:
    if policy is PolicyType.BASE_STOCK:
        return {node.name: base_stock_target(node) for node in nodes}
    if policy is PolicyType.CRITICAL_RATIO:
        return {node.name: critical_ratio_target(node) for node in nodes}
    if policy is PolicyType.ECHELON_BASE_STOCK:
        return echelon_base_stock_targets(nodes)
    raise ValueError(f"Unsupported policy: {policy}")
