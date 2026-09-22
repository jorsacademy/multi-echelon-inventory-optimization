from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class InventoryParameters:
    order_cost: np.ndarray
    holding_cost: np.ndarray
    shortage_cost: np.ndarray
    order_capacity: np.ndarray
    initial_inventory: float = 8.0

    @classmethod
    def default(cls, horizon: int):
        return cls(
            order_cost=np.linspace(4.7, 5.5, horizon),
            holding_cost=np.linspace(0.75, 1.05, horizon),
            shortage_cost=np.linspace(9.0, 11.5, horizon),
            order_capacity=np.linspace(92.0, 108.0, horizon),
            initial_inventory=8.0,
        )

    def __post_init__(self):
        arrays = [
            np.asarray(self.order_cost, dtype=float),
            np.asarray(self.holding_cost, dtype=float),
            np.asarray(self.shortage_cost, dtype=float),
            np.asarray(self.order_capacity, dtype=float),
        ]
        H = len(arrays[0])
        if any(a.shape != (H,) for a in arrays):
            raise ValueError("all parameter arrays must have the same horizon")
        if any(np.any(a < 0) for a in arrays[:3]) or np.any(arrays[3] <= 0):
            raise ValueError("invalid inventory costs/capacities")


@dataclass(frozen=True)
class InventoryPlan:
    orders: np.ndarray
    objective: float
    status: str
    scenario_count: int
    worst_case_inventory_cost: float


def box_vertices(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if lo.shape != hi.shape or lo.ndim != 1 or np.any(lo > hi):
        raise ValueError("invalid box")
    H = len(lo)
    return np.asarray([
        [hi[t] if bit[t] else lo[t] for t in range(H)]
        for bit in itertools.product([0, 1], repeat=H)
    ], dtype=float)


def trajectory_cost(
    orders: np.ndarray,
    demand: np.ndarray,
    params: InventoryParameters,
):
    q = np.asarray(orders, dtype=float)
    d = np.asarray(demand, dtype=float)
    inventory = float(params.initial_inventory)
    inv_cost = 0.0
    immediate_filled = 0.0
    stockout_periods = 0

    for t in range(len(q)):
        available_for_new = max(inventory + q[t], 0.0)
        immediate_filled += min(available_for_new, d[t])
        if available_for_new + 1e-12 < d[t]:
            stockout_periods += 1

        inventory = inventory + q[t] - d[t]
        inv_cost += (
            params.holding_cost[t]*max(inventory, 0.0)
            + params.shortage_cost[t]*max(-inventory, 0.0)
        )

    total = float(np.dot(params.order_cost, q) + inv_cost)
    fill_rate = float(immediate_filled / max(float(np.sum(d)), 1e-12))
    stockout_rate = float(stockout_periods / len(q))
    return total, fill_rate, stockout_rate, float(inventory), float(inv_cost)


def _solve_scenario_lp(
    scenarios: np.ndarray,
    params: InventoryParameters,
    *,
    robust_max: bool,
) -> InventoryPlan:
    D = np.asarray(scenarios, dtype=float)
    if D.ndim != 2:
        raise ValueError("scenarios must be [S,H]")
    S, H = D.shape
    if len(params.order_cost) != H:
        raise ValueError("scenario horizon mismatch")

    q0 = 0
    h0 = H
    b0 = h0 + S*H
    z0 = b0 + S*H
    nvar = z0 + (1 if robust_max else 0)

    c = np.zeros(nvar)
    c[q0:q0+H] = params.order_cost
    if robust_max:
        c[z0] = 1.0
    else:
        for s in range(S):
            for t in range(H):
                c[h0+s*H+t] = params.holding_cost[t] / S
                c[b0+s*H+t] = params.shortage_cost[t] / S

    Aeq, beq = [], []
    for s in range(S):
        cumulative_demand = 0.0
        for t in range(H):
            cumulative_demand += D[s, t]
            row = np.zeros(nvar)
            row[q0:q0+t+1] = -1.0
            row[h0+s*H+t] = 1.0
            row[b0+s*H+t] = -1.0
            Aeq.append(row)
            beq.append(params.initial_inventory - cumulative_demand)

    Aub, bub = [], []
    if robust_max:
        for s in range(S):
            row = np.zeros(nvar)
            for t in range(H):
                row[h0+s*H+t] = params.holding_cost[t]
                row[b0+s*H+t] = params.shortage_cost[t]
            row[z0] = -1.0
            Aub.append(row)
            bub.append(0.0)

    bounds = []
    for t in range(H):
        bounds.append((0.0, float(params.order_capacity[t])))
    bounds += [(0.0, None)] * (2*S*H)
    if robust_max:
        bounds.append((0.0, None))

    result = linprog(
        c,
        A_ub=np.asarray(Aub) if Aub else None,
        b_ub=np.asarray(bub) if bub else None,
        A_eq=np.asarray(Aeq),
        b_eq=np.asarray(beq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"inventory LP failed: {result.message}")

    orders = np.asarray(result.x[q0:q0+H], dtype=float)
    if robust_max:
        worst_inventory = float(result.x[z0])
    else:
        costs = [trajectory_cost(orders, d, params)[4] for d in D]
        worst_inventory = float(max(costs))

    return InventoryPlan(
        orders=orders,
        objective=float(result.fun),
        status="OPTIMAL",
        scenario_count=S,
        worst_case_inventory_cost=worst_inventory,
    )


def solve_deterministic_plan(
    predicted_demand: np.ndarray,
    params: InventoryParameters,
) -> InventoryPlan:
    return _solve_scenario_lp(
        np.asarray(predicted_demand, dtype=float)[None, :],
        params,
        robust_max=False,
    )


def solve_box_robust_plan(
    lower: np.ndarray,
    upper: np.ndarray,
    params: InventoryParameters,
) -> InventoryPlan:
    vertices = box_vertices(lower, upper)
    return _solve_scenario_lp(vertices, params, robust_max=True)


def solve_saa_plan(
    demand_scenarios: np.ndarray,
    params: InventoryParameters,
) -> InventoryPlan:
    """Expected-cost SAA reference over supplied scenarios."""
    return _solve_scenario_lp(demand_scenarios, params, robust_max=False)
