# Hyperopt Inventory Policy Optimization

Bayesian/TPE optimization of a stochastic `(s, S)` inventory policy with Hyperopt 0.3.0.

## What this repository demonstrates

- A reproducible stochastic lost-sales inventory simulator.
- Integer policy search using `hp.uniformint`.
- A dictionary objective returning `loss` and `STATUS_OK`.
- Explicit `Trials` storage for evaluation history.
- TPE search via `tpe.suggest`.
- Budget-matched random search via `rand.suggest`.
- Common random numbers during search and separate validation seeds.
- Unit/integration tests and GitHub Actions.

## Policy

The policy is defined by a reorder point `s` and an order-up-to level `S`. When on-hand inventory is at or below `s`, the model immediately replenishes to `S`. Demand is Poisson and unmet demand is treated as lost sales.

The search space is parameterized as `(s, gap)` where `S = s + gap`, which guarantees a valid policy by construction.

## Install

```bash
python -m pip install -e '.[dev]'
```

## Run

```bash
hyperopt-inventory-demo
```

or:

```bash
python -m hyperopt_inventory.cli
```

## Test

```bash
pytest
```

The test suite executes real Hyperopt `fmin()` calls for both TPE and random search and enforces at least 90% project coverage.
