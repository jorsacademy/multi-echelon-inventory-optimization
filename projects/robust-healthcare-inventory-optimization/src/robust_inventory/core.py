from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

import numpy as np

Segment = tuple[str, str]
DemandScenario = dict[Segment, list[int]]


@dataclass(frozen=True)
class InventoryParameters:
    unit_cost: float = 45.0
    holding_cost_rate: float = 0.0125
    shortage_cost: float = 200.0
    obsolescence_cost: float = 30.0
    fixed_order_cost: float = 500.0
    emergency_premium: float = 0.40
    lead_time: int = 3
    max_shelf_life: int = 18
    service_level: float = 0.95
    budget: float = 2_500_000.0
    max_capacity: int = 50_000
    periods: int = 12

    def __post_init__(self) -> None:
        if self.unit_cost <= 0:
            raise ValueError("unit_cost must be positive")
        if not 0 < self.service_level < 1:
            raise ValueError("service_level must be in (0, 1)")
        if self.lead_time < 0 or self.max_shelf_life < 1 or self.periods < 1:
            raise ValueError("time parameters must be non-negative/positive")
        if self.budget < 0 or self.max_capacity < 0:
            raise ValueError("budget and capacity must be non-negative")


class DemandScenarioGenerator:
    hospital_types = ("Large_Teaching", "Medium_General", "Small_Community")
    regions = ("Metropolitan", "Suburban", "Rural")

    hospital_distribution: Mapping[Segment, int] = {
        ("Large_Teaching", "Metropolitan"): 3,
        ("Large_Teaching", "Suburban"): 1,
        ("Large_Teaching", "Rural"): 0,
        ("Medium_General", "Metropolitan"): 2,
        ("Medium_General", "Suburban"): 3,
        ("Medium_General", "Rural"): 2,
        ("Small_Community", "Metropolitan"): 0,
        ("Small_Community", "Suburban"): 1,
        ("Small_Community", "Rural"): 3,
    }
    base_demand = {
        "Large_Teaching": 850,
        "Medium_General": 320,
        "Small_Community": 120,
    }
    region_multipliers = {"Metropolitan": 1.2, "Suburban": 1.0, "Rural": 0.8}
    seasonal_pattern = np.array([1.0, 1.1, 1.3, 1.8, 1.9, 1.7, 1.5, 1.0, 0.9, 1.4, 1.6, 1.2])

    def __init__(self, params: InventoryParameters, seed: int = 42) -> None:
        self.params = params
        self.seed = seed

    @property
    def active_segments(self) -> tuple[Segment, ...]:
        return tuple(k for k, count in self.hospital_distribution.items() if count > 0)

    def generate_scenarios(
        self, num_scenarios: int = 500
    ) -> tuple[dict[int, DemandScenario], np.ndarray]:
        if num_scenarios < 1:
            raise ValueError("num_scenarios must be >= 1")
        rng = np.random.default_rng(self.seed)
        scenarios: dict[int, DemandScenario] = {}
        probs = np.full(num_scenarios, 1.0 / num_scenarios)

        for s in range(num_scenarios):
            adoption = rng.lognormal(mean=-0.5 * 0.30**2, sigma=0.30)
            economic = float(np.clip(rng.normal(1.0, 0.1), 0.7, 1.3))
            scenario: DemandScenario = {}
            for hospital_type, region in self.active_segments:
                count = self.hospital_distribution[(hospital_type, region)]
                values: list[int] = []
                for t in range(self.params.periods):
                    seasonal = self.seasonal_pattern[t % len(self.seasonal_pattern)]
                    mean = (
                        self.base_demand[hospital_type]
                        * self.region_multipliers[region]
                        * seasonal
                        * count
                        * adoption
                        * economic
                    )
                    demand = max(0.0, rng.normal(mean, mean * 0.15))
                    if rng.random() < 0.10:
                        demand *= rng.uniform(0.6, 0.9)
                    values.append(int(round(demand)))
                scenario[(hospital_type, region)] = values
            scenarios[s] = scenario
        return scenarios, probs


@dataclass(frozen=True)
class ReorderRule:
    reorder_point: int
    order_quantity: int


@dataclass
class InventoryPolicy:
    initial_inventory: dict[tuple[str, str, int], int]
    reorder_policy: dict[Segment, ReorderRule]

    @property
    def total_initial_units(self) -> int:
        return sum(self.initial_inventory.values())


@dataclass
class SimulationResult:
    expected_total_cost: float
    service_levels: dict[Segment, float]
    expected_cost_breakdown: dict[str, float]
    scenario_costs: list[float]


class RobustInventoryOptimizer:
    """Scenario-based robust inventory planner with a reproducible simulation engine.

    The current implementation is a stochastic/robust heuristic baseline. It does not
    claim to solve the Wasserstein DRO reformulation exactly; that extension is kept
    explicit in the project roadmap instead of being silently approximated.
    """

    def __init__(
        self,
        params: InventoryParameters | None = None,
        *,
        num_scenarios: int = 500,
        seed: int = 42,
    ) -> None:
        self.params = params or InventoryParameters()
        self.generator = DemandScenarioGenerator(self.params, seed=seed)
        self.scenarios, self.scenario_probs = self.generator.generate_scenarios(num_scenarios)

    def expected_monthly_demand(self) -> dict[Segment, float]:
        result: dict[Segment, float] = {}
        for seg in self.generator.active_segments:
            sample = [d for scenario in self.scenarios.values() for d in scenario[seg]]
            result[seg] = float(np.mean(sample))
        return result

    def demand_std(self) -> dict[Segment, float]:
        result: dict[Segment, float] = {}
        for seg in self.generator.active_segments:
            sample = [d for scenario in self.scenarios.values() for d in scenario[seg]]
            result[seg] = float(np.std(sample, ddof=0))
        return result

    def solve_heuristic(self) -> InventoryPolicy:
        means = self.expected_monthly_demand()
        stds = self.demand_std()
        z = NormalDist().inv_cdf(self.params.service_level)
        initial: dict[tuple[str, str, int], int] = {}
        rules: dict[Segment, ReorderRule] = {}

        for seg in self.generator.active_segments:
            mean = means[seg]
            std = stds[seg]
            lead_time_mean = mean * self.params.lead_time
            lead_time_std = std * sqrt(max(1, self.params.lead_time))
            safety_stock = z * lead_time_std
            cycle_stock = mean / 2.0
            target_initial = max(0, int(round(cycle_stock + safety_stock)))

            n_ages = min(5, self.params.max_shelf_life)
            weights = np.exp(-np.arange(n_ages) * 0.45)
            weights /= weights.sum()
            allocated = 0
            for idx, weight in enumerate(weights, start=1):
                qty = int(round(target_initial * float(weight)))
                if idx == n_ages:
                    qty = max(0, target_initial - allocated)
                allocated += qty
                if qty:
                    initial[(seg[0], seg[1], idx)] = qty

            rules[seg] = ReorderRule(
                reorder_point=max(0, int(round(lead_time_mean + safety_stock))),
                order_quantity=max(1, int(round(mean * 2.0))),
            )

        initial = self._enforce_initial_constraints(initial)
        return InventoryPolicy(initial_inventory=initial, reorder_policy=rules)

    def _enforce_initial_constraints(
        self, inventory: dict[tuple[str, str, int], int]
    ) -> dict[tuple[str, str, int], int]:
        max_units_budget = int(self.params.budget // self.params.unit_cost)
        max_units = min(max_units_budget, self.params.max_capacity)
        total = sum(inventory.values())
        if total <= max_units:
            return inventory
        if max_units <= 0:
            return {}
        factor = max_units / total
        scaled = {key: int(qty * factor) for key, qty in inventory.items()}
        remaining = max_units - sum(scaled.values())
        for key in sorted(scaled, key=lambda k: (k[2], k[0], k[1])):
            if remaining <= 0:
                break
            scaled[key] += 1
            remaining -= 1
        return {k: v for k, v in scaled.items() if v > 0}

    def simulate_policy_performance(self, policy: InventoryPolicy) -> SimulationResult:
        total_cost = 0.0
        component_totals = {
            "holding": 0.0,
            "ordering": 0.0,
            "shortage": 0.0,
            "obsolescence": 0.0,
        }
        demand_totals = {seg: 0.0 for seg in self.generator.active_segments}
        fulfilled_totals = {seg: 0.0 for seg in self.generator.active_segments}
        scenario_costs: list[float] = []

        for s, scenario in self.scenarios.items():
            prob = float(self.scenario_probs[s])
            inventory: dict[tuple[str, str, int], int] = dict(policy.initial_inventory)
            pipeline: dict[tuple[Segment, int], int] = {}
            scenario_components = {k: 0.0 for k in component_totals}

            for t in range(1, self.params.periods + 1):
                for seg in self.generator.active_segments:
                    i, j = seg
                    arrival = pipeline.pop((seg, t), 0)
                    if arrival:
                        inventory[(i, j, 1)] = inventory.get((i, j, 1), 0) + arrival

                    available = sum(
                        inventory.get((i, j, a), 0)
                        for a in range(1, self.params.max_shelf_life + 1)
                    )
                    rule = policy.reorder_policy[seg]
                    on_order = sum(
                        qty
                        for (pseg, due), qty in pipeline.items()
                        if pseg == seg and due > t
                    )
                    inventory_position = available + on_order
                    if inventory_position <= rule.reorder_point:
                        due = t + self.params.lead_time
                        if due <= self.params.periods:
                            pipeline[(seg, due)] = (
                                pipeline.get((seg, due), 0) + rule.order_quantity
                            )
                        scenario_components["ordering"] += (
                            self.params.fixed_order_cost
                            + self.params.unit_cost * rule.order_quantity
                        )

                    demand = scenario[seg][t - 1]
                    demand_totals[seg] += prob * demand
                    remaining = demand
                    fulfilled = 0

                    for age in range(self.params.max_shelf_life, 0, -1):
                        key = (i, j, age)
                        stock = inventory.get(key, 0)
                        if stock <= 0 or remaining <= 0:
                            continue
                        used = min(stock, remaining)
                        inventory[key] = stock - used
                        remaining -= used
                        fulfilled += used

                    fulfilled_totals[seg] += prob * fulfilled
                    scenario_components["shortage"] += (
                        remaining * self.params.shortage_cost
                    )

                    aged: dict[tuple[str, str, int], int] = {}
                    for age in range(1, self.params.max_shelf_life + 1):
                        stock = inventory.get((i, j, age), 0)
                        if stock <= 0:
                            continue
                        if age == self.params.max_shelf_life:
                            scenario_components["obsolescence"] += (
                                stock * self.params.obsolescence_cost
                            )
                        else:
                            aged[(i, j, age + 1)] = aged.get((i, j, age + 1), 0) + stock
                    for age in range(1, self.params.max_shelf_life + 1):
                        inventory.pop((i, j, age), None)
                    inventory.update(aged)

                    ending = sum(
                        inventory.get((i, j, a), 0)
                        for a in range(1, self.params.max_shelf_life + 1)
                    )
                    scenario_components["holding"] += (
                        self.params.holding_cost_rate * self.params.unit_cost * ending
                    )

                if sum(inventory.values()) > self.params.max_capacity:
                    raise RuntimeError("policy simulation exceeded inventory capacity")

            scenario_cost = sum(scenario_components.values())
            scenario_costs.append(scenario_cost)
            total_cost += prob * scenario_cost
            for key in component_totals:
                component_totals[key] += prob * scenario_components[key]

        service_levels = {
            seg: (fulfilled_totals[seg] / demand_totals[seg] if demand_totals[seg] else 1.0)
            for seg in self.generator.active_segments
        }
        return SimulationResult(total_cost, service_levels, component_totals, scenario_costs)

    def cvar(self, scenario_costs: Iterable[float], alpha: float = 0.95) -> float:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        values = np.sort(np.asarray(list(scenario_costs), dtype=float))
        if values.size == 0:
            raise ValueError("scenario_costs cannot be empty")
        start = min(values.size - 1, int(np.floor(alpha * values.size)))
        return float(values[start:].mean())

    def analyze(self, policy: InventoryPolicy) -> dict:
        simulation = self.simulate_policy_performance(policy)
        investment = policy.total_initial_units * self.params.unit_cost
        return {
            "initial_units": policy.total_initial_units,
            "initial_investment": investment,
            "budget_utilization": investment / self.params.budget if self.params.budget else 0.0,
            "expected_total_cost": simulation.expected_total_cost,
            "expected_cost_breakdown": simulation.expected_cost_breakdown,
            "service_levels": simulation.service_levels,
            "cvar_95": self.cvar(simulation.scenario_costs, 0.95),
        }
