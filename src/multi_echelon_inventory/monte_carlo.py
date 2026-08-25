from __future__ import annotations

from dataclasses import dataclass
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
    ``policy_summary`` aggregates system-level outcomes across replications.
    """

    run_results: pd.DataFrame
    node_results: pd.DataFrame
    policy_summary: pd.DataFrame


def compare_policies(
    system: MultiEchelonSystem,
    runs: int = 100,
    periods: int = 365,
    base_seed: int | None = 42,
    policies: Iterable[PolicyType] | None = None,
) -> MonteCarloResult:
    """Compare inventory policies with repeated stochastic simulations.

    Each replication uses the same seed across policies. Because the simulator
    separates demand and lead-time random streams, this provides common random
    numbers for customer demand and partially aligned lead-time draws, reducing
    noise in policy comparisons without forcing identical event paths.
    """
    if runs <= 0:
        raise ValueError("runs must be greater than zero")
    if periods <= 0:
        raise ValueError("periods must be greater than zero")

    policy_list = list(policies) if policies is not None else list(PolicyType)
    if not policy_list:
        raise ValueError("At least one policy is required")
    if len(policy_list) != len(set(policy_list)):
        raise ValueError("policies must not contain duplicates")

    if base_seed is None:
        seed_sequence = np.random.SeedSequence()
    else:
        seed_sequence = np.random.SeedSequence(base_seed)
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
                    "Total Holding Cost": float(
                        node_summary["Holding Cost"].sum()
                    ),
                    "Total Shortage Cost": float(
                        node_summary["Shortage Cost"].sum()
                    ),
                    "Total Ordering Cost": float(
                        node_summary["Ordering Cost"].sum()
                    ),
                    "Avg Network On Hand": float(
                        node_summary["Avg On Hand"].sum()
                    ),
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
        summary_rows.append(
            {
                "Policy": policy,
                "Runs": float(len(frame)),
                "Mean Total Cost": float(costs.mean()),
                "Std Total Cost": float(costs.std(ddof=1)) if len(frame) > 1 else 0.0,
                "P05 Total Cost": float(costs.quantile(0.05)),
                "P50 Total Cost": float(costs.quantile(0.50)),
                "P95 Total Cost": float(costs.quantile(0.95)),
                "Mean Customer Fill Rate": float(
                    frame["Customer Fill Rate"].mean()
                ),
                "Mean Customer Cycle Service Level": float(
                    frame["Customer Cycle Service Level"].mean()
                ),
                "Mean Network On Hand": float(
                    frame["Avg Network On Hand"].mean()
                ),
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
