from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd

from .models import SimulationConfig
from .policies import PolicyType
from .simulation import MultiEchelonSystem


@dataclass(frozen=True)
class MonteCarloResult:
    """Results from repeated policy simulations.

    ``run_results`` contains one system-level row per policy and replication.
    ``node_results`` contains node-level KPIs for every replication.
    ``policy_summary`` aggregates system-level outcomes across replications and
    includes normal-approximation confidence intervals for key mean metrics.
    """

    run_results: pd.DataFrame
    node_results: pd.DataFrame
    policy_summary: pd.DataFrame


def _mean_ci(series: pd.Series, confidence_level: float) -> tuple[float, float, float]:
    mean = float(series.mean())
    if len(series) <= 1:
        return mean, mean, mean
    std = float(series.std(ddof=1))
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    half_width = z * std / np.sqrt(len(series))
    return mean, mean - half_width, mean + half_width


def compare_policies(
    system: MultiEchelonSystem,
    runs: int = 100,
    periods: int = 365,
    base_seed: int | None = 42,
    policies: Iterable[PolicyType] | None = None,
    confidence_level: float = 0.95,
) -> MonteCarloResult:
    """Compare inventory policies with repeated stochastic simulations.

    Each replication uses the same seed across policies. This common-random-number
    design reduces comparison noise. Confidence intervals use a normal
    approximation around replication means and should be interpreted as simulation
    uncertainty estimates rather than guarantees about the real supply chain.
    """
    if runs <= 0:
        raise ValueError("runs must be greater than zero")
    if periods <= 0:
        raise ValueError("periods must be greater than zero")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between 0 and 1")

    policy_list = list(policies) if policies is not None else list(PolicyType)
    if not policy_list:
        raise ValueError("At least one policy is required")
    if len(policy_list) != len(set(policy_list)):
        raise ValueError("policies must not contain duplicates")

    seed_sequence = (
        np.random.SeedSequence()
        if base_seed is None
        else np.random.SeedSequence(base_seed)
    )
    replicate_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(runs)
    ]

    customer_name = system.nodes[-1].name
    run_rows: list[dict[str, float | int | str]] = []
    node_frames: list[pd.DataFrame] = []

    for run_index, seed in enumerate(replicate_seeds, start=1):
        for policy in policy_list:
            result = system.simulate(
                SimulationConfig(periods=periods, seed=seed),
                policy=policy,
            )
            node_summary = result.summary.copy()
            node_summary.insert(0, "Policy", policy.value)
            node_summary.insert(1, "Run", run_index)
            node_summary.insert(2, "Seed", seed)
            node_frames.append(node_summary)

            customer_row = node_summary.loc[
                node_summary["Node"] == customer_name
            ].iloc[0]
            run_rows.append(
                {
                    "Policy": policy.value,
                    "Run": run_index,
                    "Seed": seed,
                    "Total Cost": float(node_summary["Total Cost"].sum()),
                    "Total Holding Cost": float(node_summary["Holding Cost"].sum()),
                    "Total Shortage Cost": float(node_summary["Shortage Cost"].sum()),
                    "Total Ordering Cost": float(node_summary["Ordering Cost"].sum()),
                    "Avg Network On Hand": float(node_summary["Avg On Hand"].sum()),
                    "Avg Network Backorders": float(
                        node_summary["Avg Backorders"].sum()
                    ),
                    "Customer Fill Rate": float(customer_row["Fill Rate"]),
                    "Customer Cycle Service Level": float(
                        customer_row["Cycle Service Level"]
                    ),
                }
            )

    run_results = pd.DataFrame.from_records(run_rows)
    node_results = pd.concat(node_frames, ignore_index=True)

    summary_rows: list[dict[str, float | str]] = []
    for policy, frame in run_results.groupby("Policy", sort=False):
        costs = frame["Total Cost"]
        mean_cost, cost_ci_low, cost_ci_high = _mean_ci(costs, confidence_level)
        mean_fill, fill_ci_low, fill_ci_high = _mean_ci(
            frame["Customer Fill Rate"], confidence_level
        )
        mean_csl, csl_ci_low, csl_ci_high = _mean_ci(
            frame["Customer Cycle Service Level"], confidence_level
        )
        summary_rows.append(
            {
                "Policy": policy,
                "Runs": int(len(frame)),
                "Mean Total Cost": mean_cost,
                "Mean Total Cost CI Low": cost_ci_low,
                "Mean Total Cost CI High": cost_ci_high,
                "Std Total Cost": float(costs.std(ddof=1)) if len(frame) > 1 else 0.0,
                "P05 Total Cost": float(costs.quantile(0.05)),
                "P50 Total Cost": float(costs.quantile(0.50)),
                "P95 Total Cost": float(costs.quantile(0.95)),
                "Mean Customer Fill Rate": mean_fill,
                "Mean Customer Fill Rate CI Low": max(0.0, fill_ci_low),
                "Mean Customer Fill Rate CI High": min(1.0, fill_ci_high),
                "Mean Customer Cycle Service Level": mean_csl,
                "Mean Customer Cycle Service Level CI Low": max(0.0, csl_ci_low),
                "Mean Customer Cycle Service Level CI High": min(1.0, csl_ci_high),
                "Mean Network On Hand": float(frame["Avg Network On Hand"].mean()),
                "Mean Network Backorders": float(
                    frame["Avg Network Backorders"].mean()
                ),
            }
        )

    policy_summary = (
        pd.DataFrame.from_records(summary_rows)
        .sort_values("Mean Total Cost", kind="stable")
        .reset_index(drop=True)
    )
    return MonteCarloResult(
        run_results=run_results,
        node_results=node_results,
        policy_summary=policy_summary,
    )
