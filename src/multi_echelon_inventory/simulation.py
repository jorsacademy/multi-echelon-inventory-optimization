from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import summarize_history
from .models import InventoryNode, SimulationConfig
from .policies import PolicyType, policy_targets


@dataclass(frozen=True)
class SimulationResult:
    history: pd.DataFrame
    summary: pd.DataFrame


class MultiEchelonSystem:
    """Serial multi-echelon inventory planning and simulation model."""

    def __init__(self, nodes: Sequence[InventoryNode]) -> None:
        self.nodes = list(nodes)
        if not self.nodes:
            raise ValueError("At least one inventory node is required")
        names = [node.name for node in self.nodes]
        if len(names) != len(set(names)):
            raise ValueError("Inventory node names must be unique")

    def recommended_stock_levels(
        self, policy: PolicyType = PolicyType.ECHELON_BASE_STOCK
    ) -> dict[str, float]:
        return policy_targets(self.nodes, policy)

    def simulate(
        self,
        config: SimulationConfig = SimulationConfig(),
        policy: PolicyType = PolicyType.ECHELON_BASE_STOCK,
    ) -> SimulationResult:
        """Simulate each node with stochastic demand and replenishment dynamics.

        Nodes share an echelon-aware planning policy when selected, but physical
        shipments are not coupled between adjacent nodes. Replenishment supply is
        unconstrained, and shortages are backordered.
        """
        targets = self.recommended_stock_levels(policy)
        seed_sequence = np.random.SeedSequence(config.seed)
        child_seeds = seed_sequence.spawn(len(self.nodes))
        records: list[dict[str, float | int | str]] = []

        for node, child_seed in zip(self.nodes, child_seeds):
            rng = np.random.default_rng(child_seed)
            on_hand = float(node.initial_on_hand)
            backorders = 0.0
            pipeline: list[tuple[int, float]] = []
            target = float(targets[node.name])

            for period in range(1, config.periods + 1):
                arrivals = sum(qty for due, qty in pipeline if due == period)
                pipeline = [(due, qty) for due, qty in pipeline if due != period]
                on_hand += arrivals

                if backorders > 0 and on_hand > 0:
                    backlog_fill = min(on_hand, backorders)
                    on_hand -= backlog_fill
                    backorders -= backlog_fill

                demand = float(rng.normal(node.demand_mean, node.demand_std))
                if config.truncate_demand_at_zero:
                    demand = max(0.0, demand)

                filled_demand = min(on_hand, demand)
                on_hand -= filled_demand
                unmet = demand - filled_demand
                backorders += unmet

                on_order_before = sum(qty for _, qty in pipeline)
                inventory_position_before = on_hand + on_order_before - backorders
                order_quantity = max(0.0, target - inventory_position_before)

                if order_quantity > 0:
                    if node.lead_time == 0:
                        on_hand += order_quantity
                    else:
                        pipeline.append((period + node.lead_time, order_quantity))

                on_order_after = sum(qty for _, qty in pipeline)
                inventory_position_after = on_hand + on_order_after - backorders

                holding_cost = on_hand * node.holding_cost
                shortage_cost = backorders * node.shortage_cost
                ordering_cost = node.ordering_cost if order_quantity > 0 else 0.0
                total_cost = holding_cost + shortage_cost + ordering_cost

                records.append(
                    {
                        "Period": period,
                        "Node": node.name,
                        "Demand": demand,
                        "Filled Demand": filled_demand,
                        "Arrivals": arrivals,
                        "Order Quantity": order_quantity,
                        "On Hand": on_hand,
                        "On Order": on_order_after,
                        "Backorders": backorders,
                        "Inventory Position": inventory_position_after,
                        "Target": target,
                        "Holding Cost": holding_cost,
                        "Shortage Cost": shortage_cost,
                        "Ordering Cost": ordering_cost,
                        "Total Cost": total_cost,
                    }
                )

        history = pd.DataFrame.from_records(records)
        summary = summarize_history(history)
        return SimulationResult(history=history, summary=summary)
