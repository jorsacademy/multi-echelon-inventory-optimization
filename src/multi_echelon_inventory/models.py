from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True)
class InventoryNode:
    """Inventory parameters for one echelon in a serial supply chain.

    Nodes are ordered from the most upstream echelon to the customer-facing
    echelon. ``lead_time`` is the inbound transportation/replenishment lead time
    for this node. ``shipment_capacity`` limits how many units the node can ship
    to its immediate downstream node per simulation period.
    """

    name: str
    initial_on_hand: float
    demand_mean: float
    demand_std: float
    lead_time: int
    holding_cost: float
    shortage_cost: float
    ordering_cost: float = 0.0
    service_level: float = 0.95
    shipment_capacity: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if self.initial_on_hand < 0:
            raise ValueError("initial_on_hand must be non-negative")
        if self.demand_mean < 0:
            raise ValueError("demand_mean must be non-negative")
        if self.demand_std < 0:
            raise ValueError("demand_std must be non-negative")
        if self.lead_time < 0:
            raise ValueError("lead_time must be non-negative")
        if self.holding_cost <= 0:
            raise ValueError("holding_cost must be greater than zero")
        if self.shortage_cost <= 0:
            raise ValueError("shortage_cost must be greater than zero")
        if self.ordering_cost < 0:
            raise ValueError("ordering_cost must be non-negative")
        if not 0 < self.service_level < 1:
            raise ValueError("service_level must be strictly between 0 and 1")
        if self.shipment_capacity is not None and self.shipment_capacity <= 0:
            raise ValueError("shipment_capacity must be greater than zero when set")

    @property
    def service_z(self) -> float:
        """Standard-normal quantile associated with the target cycle service level."""
        return NormalDist().inv_cdf(self.service_level)

    def lead_time_demand_mean(self) -> float:
        return self.demand_mean * self.lead_time

    def lead_time_demand_std(self) -> float:
        return self.demand_std * (self.lead_time ** 0.5)

    def safety_stock(self) -> float:
        return max(0.0, self.service_z * self.lead_time_demand_std())

    def reorder_point(self) -> float:
        return self.lead_time_demand_mean() + self.safety_stock()


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a discrete-time inventory simulation."""

    periods: int = 365
    seed: int | None = 42
    truncate_demand_at_zero: bool = True

    def __post_init__(self) -> None:
        if self.periods <= 0:
            raise ValueError("periods must be greater than zero")
