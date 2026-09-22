from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .planning import (
    InventoryParameters,
    solve_box_robust_plan,
    solve_deterministic_plan,
    solve_saa_plan,
    trajectory_cost,
)
from .data import sample_demand_given_features


@dataclass(frozen=True)
class MethodMetrics:
    name: str
    mean_cost: float
    median_cost: float
    mean_fill_rate: float
    mean_stockout_rate: float
    mean_clairvoyant_gap_pct: float
    mean_order_quantity: float


def evaluate_methods(
    predictions: np.ndarray,
    realized_demand: np.ndarray,
    params: InventoryParameters,
    interval_builders: dict,
):
    predictions = np.asarray(predictions, dtype=float)
    actual = np.asarray(realized_demand, dtype=float)
    if predictions.shape != actual.shape:
        raise ValueError("prediction/actual shape mismatch")

    records = {"Deterministic": []}
    for name in interval_builders:
        records[name] = []

    for pred, demand in zip(predictions, actual):
        plans = {"Deterministic": solve_deterministic_plan(pred, params)}
        for name, builder in interval_builders.items():
            lo, hi = builder(pred)
            plans[name] = solve_box_robust_plan(lo, hi, params)

        clairvoyant = solve_deterministic_plan(demand, params)
        clairvoyant_cost = trajectory_cost(clairvoyant.orders, demand, params)[0]

        for name, plan in plans.items():
            cost, fill, stockout, _, _ = trajectory_cost(plan.orders, demand, params)
            records[name].append((
                cost,
                fill,
                stockout,
                100.0*(cost-clairvoyant_cost)/max(abs(clairvoyant_cost), 1e-12),
                float(np.sum(plan.orders)),
            ))

    summaries = []
    for name, rows in records.items():
        a = np.asarray(rows, dtype=float)
        summaries.append(MethodMetrics(
            name=name,
            mean_cost=float(a[:, 0].mean()),
            median_cost=float(np.median(a[:, 0])),
            mean_fill_rate=float(a[:, 1].mean()),
            mean_stockout_rate=float(a[:, 2].mean()),
            mean_clairvoyant_gap_pct=float(a[:, 3].mean()),
            mean_order_quantity=float(a[:, 4].mean()),
        ))
    return tuple(summaries)


@dataclass(frozen=True)
class DistributionReferenceMetrics:
    name: str
    mean_cost: float
    mean_fill_rate: float
    mean_stockout_rate: float
    mean_order_quantity: float


def evaluate_true_distribution_reference(
    predictor,
    contexts: np.ndarray,
    params: InventoryParameters,
    interval_builders: dict,
    *,
    seed: int = 1234,
    saa_scenarios: int = 256,
    evaluation_scenarios: int = 512,
):
    """
    Compare practical plans with an information-advantaged SAA reference.

    The SAA reference can sample the true synthetic conditional demand
    generator. Practical methods never receive this generator. The reference
    is therefore not a deployable baseline and is not called an exact lower
    bound.
    """
    x = np.asarray(contexts, dtype=float)
    records = {"Distribution-aware SAA": []}
    for name in ["Deterministic", *interval_builders.keys()]:
        records[name] = []

    for i, context in enumerate(x):
        pred = predictor.predict(context[None, :])[0]
        plans = {"Deterministic": solve_deterministic_plan(pred, params)}
        for name, builder in interval_builders.items():
            lo, hi = builder(pred)
            plans[name] = solve_box_robust_plan(lo, hi, params)

        repeated = np.repeat(context[None, :], saa_scenarios, axis=0)
        train_scenarios = sample_demand_given_features(
            repeated,
            seed=seed + 10_007*i,
            horizon=len(params.order_cost),
        )
        plans["Distribution-aware SAA"] = solve_saa_plan(train_scenarios, params)

        eval_repeated = np.repeat(context[None, :], evaluation_scenarios, axis=0)
        eval_scenarios = sample_demand_given_features(
            eval_repeated,
            seed=seed + 1_000_003 + 10_007*i,
            horizon=len(params.order_cost),
        )

        for name, plan in plans.items():
            rows = [trajectory_cost(plan.orders, d, params) for d in eval_scenarios]
            a = np.asarray([[r[0], r[1], r[2]] for r in rows], dtype=float)
            records[name].append((
                float(a[:,0].mean()),
                float(a[:,1].mean()),
                float(a[:,2].mean()),
                float(np.sum(plan.orders)),
            ))

    out = []
    for name, rows in records.items():
        a = np.asarray(rows, dtype=float)
        out.append(DistributionReferenceMetrics(
            name=name,
            mean_cost=float(a[:,0].mean()),
            mean_fill_rate=float(a[:,1].mean()),
            mean_stockout_rate=float(a[:,2].mean()),
            mean_order_quantity=float(a[:,3].mean()),
        ))
    return tuple(out)
