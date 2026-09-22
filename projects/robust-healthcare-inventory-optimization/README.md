# Robust Healthcare Inventory Optimization

Scenario-based inventory planning for perishable healthcare products under demand uncertainty.

> **License:** source-available for non-commercial use only. Commercial use is prohibited unless separately licensed in writing. See `LICENSE`.

## What this repository implements

The model represents hospital-type/region segments, seasonal and stochastic demand, finite shelf life, shortage and holding costs, fixed regular-order costs, capacity/budget constraints, and delayed replenishment. It includes a reproducible Monte Carlo scenario generator, an empirical safety-stock/reorder heuristic, an age-aware FEFO simulation engine, segment service-level measurement, expected cost accounting, and CVaR reporting.

This repository intentionally distinguishes implemented functionality from research extensions. The current solver is a scenario-based robust heuristic baseline; it does **not** claim that the Wasserstein DRO reformulation, Benders decomposition, or genetic algorithm described in the original mathematical concept is already solved exactly. See `docs/model.md`.

## Key corrections from the prototype

- Monthly demand is estimated directly from monthly samples rather than being divided by the planning horizon a second time.
- Regular orders are stored in an order pipeline and arrive only after the configured lead time.
- Only orders actually triggered by the reorder rule are delivered later.
- Demand is consumed FEFO (oldest inventory first), with explicit age advancement and expiry cost.
- Service levels are computed from simulated fulfilled demand instead of comparing annual demand only with initial stock.
- Cost breakdown is calculated from actual simulation events rather than fixed illustrative percentages.
- Scenario generation uses a local NumPy random generator, avoiding global RNG side effects.

## Install

```bash
python -m pip install -e '.[dev]'
```

## Run the example

```bash
python examples/healthcare_inventory_demo.py
```

## Run quality checks

```bash
ruff check .
pytest --cov=robust_inventory --cov-report=term-missing
```

GitHub Actions runs linting and tests on Python 3.10, 3.11, and 3.12.

## Package layout

```text
src/robust_inventory/     Core model, scenario generator, heuristic, simulation, risk metrics
tests/                    Unit and regression tests
examples/                 Runnable demonstration
docs/model.md              Mathematical scope and implementation status
.github/workflows/         CI configuration
```

## Research roadmap

1. Exact two-stage stochastic MILP with explicit first-stage and recourse decisions.
2. Wasserstein DRO reformulation with documented ambiguity-set assumptions.
3. Chance/service-reliability constraints with out-of-sample validation.
4. Benders decomposition for larger scenario sets.
5. Optional metaheuristic layer for integer-policy search and benchmarking.

## License

Copyright (c) 2026 JORS Academy. The code is provided under the **JORS Academy Non-Commercial Source License 1.0**. Personal, educational, academic, evaluation, and non-commercial research use is permitted subject to the license terms. Commercial use, commercial internal deployment, paid services, SaaS, resale, and commercial-client deliverables are prohibited without a separate written license.
