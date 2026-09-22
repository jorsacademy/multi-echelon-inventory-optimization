from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, rand, tpe

from .simulation import evaluate_policy

SEARCH_SEEDS = (101, 202, 303, 404)
VALIDATION_SEEDS = (1001, 1002, 1003, 1004, 1005, 1006)


@dataclass(frozen=True)
class OptimizationResult:
    algorithm: str
    reorder_point: int
    order_up_to: int
    search_loss: float
    validation_loss: float
    evaluations: int


def search_space() -> dict[str, object]:
    """Integer `(s, gap)` space; `S = s + gap` guarantees `S > s`."""
    return {
        "reorder_point": hp.uniformint("reorder_point", 0, 12),
        "gap": hp.uniformint("gap", 2, 15),
    }


def objective(params: dict[str, float]) -> dict[str, object]:
    reorder_point = int(params["reorder_point"])
    order_up_to = reorder_point + int(params["gap"])
    loss = evaluate_policy(reorder_point, order_up_to, seeds=SEARCH_SEEDS)
    return {
        "loss": loss,
        "status": STATUS_OK,
        "reorder_point": reorder_point,
        "order_up_to": order_up_to,
    }


def run_hyperopt(
    *,
    algorithm: str = "tpe",
    max_evals: int = 30,
    seed: int = 7,
) -> OptimizationResult:
    if max_evals <= 0:
        raise ValueError("max_evals must be positive")

    algorithms: dict[str, Callable[..., object]] = {
        "tpe": tpe.suggest,
        "random": rand.suggest,
    }
    if algorithm not in algorithms:
        raise ValueError(f"unknown algorithm: {algorithm}")

    trials = Trials()
    fmin(
        fn=objective,
        space=search_space(),
        algo=algorithms[algorithm],
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(seed),
        show_progressbar=False,
        verbose=False,
    )

    best_trial = min(trials.results, key=lambda result: float(result["loss"]))
    reorder_point = int(best_trial["reorder_point"])
    order_up_to = int(best_trial["order_up_to"])
    validation_loss = evaluate_policy(
        reorder_point,
        order_up_to,
        seeds=VALIDATION_SEEDS,
    )
    return OptimizationResult(
        algorithm=algorithm,
        reorder_point=reorder_point,
        order_up_to=order_up_to,
        search_loss=float(best_trial["loss"]),
        validation_loss=validation_loss,
        evaluations=len(trials),
    )


def compare_searches(*, max_evals: int = 30, seed: int = 7) -> list[OptimizationResult]:
    """Compare TPE with a budget-matched random-search baseline."""
    results = [
        run_hyperopt(algorithm="tpe", max_evals=max_evals, seed=seed),
        run_hyperopt(algorithm="random", max_evals=max_evals, seed=seed),
    ]
    return sorted(results, key=lambda result: result.validation_loss)
