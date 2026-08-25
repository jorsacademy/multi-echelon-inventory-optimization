from __future__ import annotations

from dataclasses import dataclass
from math import inf
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
    """Serial multi-echelon inventory planning and simulation model.

    Nodes must be supplied from upstream to downstream. The last node faces
    stochastic external customer demand. Replenishment orders propagate
    upstream, while physical shipments consume upstream on-hand inventory,
    respect optional shipment capacities, and travel through lead-time pipelines.
    """

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
        """Run a coupled serial supply-chain simulation.

        The customer-facing node is the final node in ``self.nodes``. Its
        ``demand_mean`` and ``demand_std`` generate external demand. Every other
        node sees demand only through replenishment orders placed by its
        immediate downstream node.

        The first node replenishes from an unconstrained external source after
        its own lead time. All internal shipments are constrained by upstream
        on-hand stock and, when configured, ``shipment_capacity``.
        """
        n = len(self.nodes)
        targets = self.recommended_stock_levels(policy)
        rng = np.random.default_rng(config.seed)

        on_hand = [float(node.initial_on_hand) for node in self.nodes]
        backorders = [0.0] * n
        inbound_pipeline: list[list[tuple[int, float]]] = [[] for _ in self.nodes]

        records: list[dict[str, float | int | str]] = []

        for period in range(1, config.periods + 1):
            arrivals = [0.0] * n
            shipments = [0.0] * n
            demand = [0.0] * n
            immediate_fill = [0.0] * n
            order_quantity = [0.0] * n
            capacity_remaining = [
                inf if node.shipment_capacity is None else float(node.shipment_capacity)
                for node in self.nodes
            ]

            # Receive all shipments that complete their inbound lead time.
            for i in range(n):
                due_now = sum(
                    qty for due, qty in inbound_pipeline[i] if due == period
                )
                inbound_pipeline[i] = [
                    (due, qty)
                    for due, qty in inbound_pipeline[i]
                    if due != period
                ]
                arrivals[i] = due_now
                on_hand[i] += due_now

            # Clear previously backordered internal demand before new orders.
            for i in range(n - 1):
                quantity = min(on_hand[i], backorders[i], capacity_remaining[i])
                if quantity <= 0:
                    continue
                on_hand[i] -= quantity
                backorders[i] -= quantity
                capacity_remaining[i] -= quantity
                shipments[i] += quantity
                receiver = i + 1
                due = period + self.nodes[receiver].lead_time
                if self.nodes[receiver].lead_time == 0:
                    on_hand[receiver] += quantity
                    arrivals[receiver] += quantity
                else:
                    inbound_pipeline[receiver].append((due, quantity))

            # Clear previously backordered external customer demand.
            customer = n - 1
            old_customer_backlog = backorders[customer]
            backlog_fill = min(on_hand[customer], old_customer_backlog)
            on_hand[customer] -= backlog_fill
            backorders[customer] -= backlog_fill
            shipments[customer] += backlog_fill

            # Generate new external demand only at the customer-facing echelon.
            customer_node = self.nodes[customer]
            customer_demand = float(
                rng.normal(customer_node.demand_mean, customer_node.demand_std)
            )
            if config.truncate_demand_at_zero:
                customer_demand = max(0.0, customer_demand)

            demand[customer] = customer_demand
            backlog_before_new = backorders[customer]
            backorders[customer] += customer_demand
            customer_fill = min(on_hand[customer], backorders[customer])
            on_hand[customer] -= customer_fill
            backorders[customer] -= customer_fill
            shipments[customer] += customer_fill
            immediate_fill[customer] = min(
                customer_demand,
                max(0.0, customer_fill - backlog_before_new),
            )

            # Replenishment decisions propagate from downstream to upstream.
            # A downstream order is demand for its immediate upstream node.
            for i in range(n - 1, 0, -1):
                on_order = sum(qty for _, qty in inbound_pipeline[i])
                inventory_position = on_hand[i] + on_order - backorders[i]
                target = float(targets[self.nodes[i].name])
                qty = max(0.0, target - inventory_position)
                order_quantity[i] = qty

                upstream = i - 1
                old_upstream_backlog = backorders[upstream]
                demand[upstream] += qty
                backorders[upstream] += qty

                shipped_now = min(
                    on_hand[upstream],
                    backorders[upstream],
                    capacity_remaining[upstream],
                )
                if shipped_now > 0:
                    on_hand[upstream] -= shipped_now
                    backorders[upstream] -= shipped_now
                    capacity_remaining[upstream] -= shipped_now
                    shipments[upstream] += shipped_now

                    due = period + self.nodes[i].lead_time
                    if self.nodes[i].lead_time == 0:
                        on_hand[i] += shipped_now
                        arrivals[i] += shipped_now
                    else:
                        inbound_pipeline[i].append((due, shipped_now))

                immediate_fill[upstream] += min(
                    qty,
                    max(0.0, shipped_now - old_upstream_backlog),
                )

            # The most-upstream node orders from an unconstrained external source.
            source_on_order = sum(qty for _, qty in inbound_pipeline[0])
            source_inventory_position = (
                on_hand[0] + source_on_order - backorders[0]
            )
            source_target = float(targets[self.nodes[0].name])
            source_order = max(0.0, source_target - source_inventory_position)
            order_quantity[0] = source_order

            if source_order > 0:
                if self.nodes[0].lead_time == 0:
                    on_hand[0] += source_order
                    arrivals[0] += source_order
                else:
                    inbound_pipeline[0].append(
                        (period + self.nodes[0].lead_time, source_order)
                    )

            # Record state after all period events and ordering decisions.
            for i, node in enumerate(self.nodes):
                on_order = sum(qty for _, qty in inbound_pipeline[i])
                inventory_position = on_hand[i] + on_order - backorders[i]
                holding_cost = on_hand[i] * node.holding_cost
                shortage_cost = backorders[i] * node.shortage_cost
                ordering_cost = (
                    node.ordering_cost if order_quantity[i] > 0 else 0.0
                )
                total_cost = holding_cost + shortage_cost + ordering_cost

                records.append(
                    {
                        "Period": period,
                        "Node": node.name,
                        "Demand": demand[i],
                        "Filled Demand": immediate_fill[i],
                        "Arrivals": arrivals[i],
                        "Shipments": shipments[i],
                        "Order Quantity": order_quantity[i],
                        "On Hand": on_hand[i],
                        "On Order": on_order,
                        "Backorders": backorders[i],
                        "Inventory Position": inventory_position,
                        "Target": float(targets[node.name]),
                        "Holding Cost": holding_cost,
                        "Shortage Cost": shortage_cost,
                        "Ordering Cost": ordering_cost,
                        "Total Cost": total_cost,
                    }
                )

        history = pd.DataFrame.from_records(records)
        summary = summarize_history(history)
        return SimulationResult(history=history, summary=summary)
