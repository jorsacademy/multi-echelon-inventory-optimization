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
    upstream, while physical shipments consume upstream on-hand inventory and
    travel through stochastic lead-time pipelines.
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

    @staticmethod
    def _sample_lead_time(node: InventoryNode, rng: np.random.Generator) -> int:
        """Sample a non-negative integer inbound lead time for ``node``."""
        if node.lead_time_std == 0:
            return node.lead_time
        sampled = rng.normal(node.lead_time, node.lead_time_std)
        return max(0, int(np.rint(sampled)))

    @staticmethod
    def _outbound_capacity(node: InventoryNode) -> float:
        """Return the effective physical outbound capacity for a period."""
        shipment = inf if node.shipment_capacity is None else node.shipment_capacity
        production = (
            inf if node.production_capacity is None else node.production_capacity
        )
        return float(min(shipment, production))

    @staticmethod
    def _order_quantity(node: InventoryNode, target_gap: float) -> float:
        """Apply a node-level replenishment order capacity to a target gap."""
        desired = max(0.0, target_gap)
        if node.order_capacity is None:
            return desired
        return min(desired, float(node.order_capacity))

    def simulate(
        self,
        config: SimulationConfig = SimulationConfig(),
        policy: PolicyType = PolicyType.ECHELON_BASE_STOCK,
    ) -> SimulationResult:
        """Run a coupled serial supply-chain simulation.

        The customer-facing node is the final node in ``self.nodes``. Its demand
        parameters generate external demand. Every other node sees demand only
        through replenishment orders placed by its immediate downstream node.

        Internal shipments are constrained by upstream on-hand inventory,
        optional shipment capacity, and optional production capacity. Orders are
        independently capped by ``order_capacity``. Each inbound movement samples
        a non-negative integer lead time when ``lead_time_std`` is positive.
        """
        n = len(self.nodes)
        targets = self.recommended_stock_levels(policy)

        seed_sequence = np.random.SeedSequence(config.seed)
        demand_seed, lead_time_seed = seed_sequence.spawn(2)
        demand_rng = np.random.default_rng(demand_seed)
        lead_time_rng = np.random.default_rng(lead_time_seed)

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
            scheduled_lead_time_sum = [0.0] * n
            scheduled_lead_time_qty = [0.0] * n
            capacity_remaining = [self._outbound_capacity(node) for node in self.nodes]

            # Receive all shipments that complete their inbound lead time.
            for i in range(n):
                due_now = sum(qty for due, qty in inbound_pipeline[i] if due == period)
                inbound_pipeline[i] = [
                    (due, qty) for due, qty in inbound_pipeline[i] if due != period
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
                lead_time = self._sample_lead_time(self.nodes[receiver], lead_time_rng)
                scheduled_lead_time_sum[receiver] += quantity * lead_time
                scheduled_lead_time_qty[receiver] += quantity
                if lead_time == 0:
                    on_hand[receiver] += quantity
                    arrivals[receiver] += quantity
                else:
                    inbound_pipeline[receiver].append((period + lead_time, quantity))

            # Clear previously backordered external customer demand, subject to
            # the customer-facing node's outbound fulfillment capacity.
            customer = n - 1
            backlog_fill = min(
                on_hand[customer], backorders[customer], capacity_remaining[customer]
            )
            on_hand[customer] -= backlog_fill
            backorders[customer] -= backlog_fill
            capacity_remaining[customer] -= backlog_fill
            shipments[customer] += backlog_fill

            # Generate new external demand only at the customer-facing echelon.
            customer_node = self.nodes[customer]
            customer_demand = float(
                demand_rng.normal(customer_node.demand_mean, customer_node.demand_std)
            )
            if config.truncate_demand_at_zero:
                customer_demand = max(0.0, customer_demand)

            demand[customer] = customer_demand
            backorders[customer] += customer_demand
            customer_fill = min(
                on_hand[customer], backorders[customer], capacity_remaining[customer]
            )
            on_hand[customer] -= customer_fill
            backorders[customer] -= customer_fill
            capacity_remaining[customer] -= customer_fill
            shipments[customer] += customer_fill
            immediate_fill[customer] = min(customer_demand, customer_fill)

            # Replenishment decisions propagate from downstream to upstream.
            # An unshipped upstream backlog is part of the downstream node's open
            # replenishment order and therefore belongs in inventory position.
            for i in range(n - 1, 0, -1):
                on_order = sum(qty for _, qty in inbound_pipeline[i])
                open_upstream_order = backorders[i - 1]
                inventory_position = (
                    on_hand[i] + on_order + open_upstream_order - backorders[i]
                )
                target = float(targets[self.nodes[i].name])
                qty = self._order_quantity(
                    self.nodes[i], target - inventory_position
                )
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

                    lead_time = self._sample_lead_time(self.nodes[i], lead_time_rng)
                    scheduled_lead_time_sum[i] += shipped_now * lead_time
                    scheduled_lead_time_qty[i] += shipped_now
                    if lead_time == 0:
                        on_hand[i] += shipped_now
                        arrivals[i] += shipped_now
                    else:
                        inbound_pipeline[i].append(
                            (period + lead_time, shipped_now)
                        )

                immediate_fill[upstream] += min(
                    qty, max(0.0, shipped_now - old_upstream_backlog)
                )

            # The most-upstream node orders from an unconstrained external source.
            source_on_order = sum(qty for _, qty in inbound_pipeline[0])
            source_inventory_position = on_hand[0] + source_on_order - backorders[0]
            source_target = float(targets[self.nodes[0].name])
            source_order = self._order_quantity(
                self.nodes[0], source_target - source_inventory_position
            )
            order_quantity[0] = source_order

            if source_order > 0:
                lead_time = self._sample_lead_time(self.nodes[0], lead_time_rng)
                scheduled_lead_time_sum[0] += source_order * lead_time
                scheduled_lead_time_qty[0] += source_order
                if lead_time == 0:
                    on_hand[0] += source_order
                    arrivals[0] += source_order
                else:
                    inbound_pipeline[0].append(
                        (period + lead_time, source_order)
                    )

            # Record state after all period events and ordering decisions.
            for i, node in enumerate(self.nodes):
                on_order = sum(qty for _, qty in inbound_pipeline[i])
                open_upstream_order = backorders[i - 1] if i > 0 else 0.0
                inventory_position = (
                    on_hand[i] + on_order + open_upstream_order - backorders[i]
                )
                holding_cost = on_hand[i] * node.holding_cost
                shortage_cost = backorders[i] * node.shortage_cost
                ordering_cost = node.ordering_cost if order_quantity[i] > 0 else 0.0
                total_cost = holding_cost + shortage_cost + ordering_cost
                scheduled_lead_time = (
                    scheduled_lead_time_sum[i] / scheduled_lead_time_qty[i]
                    if scheduled_lead_time_qty[i] > 0
                    else np.nan
                )

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
                        "Open Upstream Order": open_upstream_order,
                        "Backorders": backorders[i],
                        "Inventory Position": inventory_position,
                        "Target": float(targets[node.name]),
                        "Scheduled Lead Time": scheduled_lead_time,
                        "Holding Cost": holding_cost,
                        "Shortage Cost": shortage_cost,
                        "Ordering Cost": ordering_cost,
                        "Total Cost": total_cost,
                    }
                )

        history = pd.DataFrame.from_records(records)
        summary = summarize_history(history)
        return SimulationResult(history=history, summary=summary)
