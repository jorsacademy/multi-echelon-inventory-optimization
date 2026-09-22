# Demand Forecasting + Inventory Control

Hybrid forecasting and inventory-control benchmark for industrial engineering and operations-research experiments.

The repository separates two questions that are often mixed together:

1. How accurately can future demand be forecast?
2. How should replenishment decisions use that forecast under lead times, holding cost, ordering cost and stockout pressure?

A lower forecast error does not automatically imply a lower inventory cost. This benchmark evaluates both layers independently and then measures end-to-end operational performance.

## Problem formulation

A single-item inventory system is reviewed daily over a finite horizon. Demand is stochastic and contains weekly seasonality, trend and random noise.

The controller observes:

- on-hand inventory,
- backlog,
- outstanding pipeline inventory,
- forecast mean,
- forecast uncertainty,
- most recent demand,
- normalized time.

The action is a continuous replenishment quantity normalized to `[0, 1]`, where `1` maps to the configured maximum order quantity.

Orders arrive after a configurable deterministic lead time.

The period cost is

`holding cost + backlog cost + variable ordering cost + fixed ordering cost`.

The objective is to minimize expected cumulative cost while maintaining a high service level.

## Forecasting layer

Two dependency-light models are included:

### Naive moving-average forecast

Uses the recent demand window as the next-period expectation.

### Weekly seasonal forecast

Uses observations at the same weekly seasonal lag, with the moving-average model as fallback when insufficient history exists.

The included `rolling_mae` utility evaluates one-step-ahead forecast quality without leaking future observations.

These models intentionally provide transparent baselines. Extensions can replace them with XGBoost, LightGBM, Prophet, LSTM, GRU or probabilistic forecasting models while keeping the inventory-control experiment unchanged.

## Inventory-control policies

### Constant order

Orders a fixed quantity each day. This is a deliberately simple reference policy.

### `(s, S)` policy

Computes inventory position as

`on hand + pipeline - backlog`

and replenishes to an order-up-to level when inventory position falls below the reorder point.

### Forecast-informed order-up-to policy

Uses forecast mean and forecast uncertainty over the lead-time protection period:

`target = protection_period * forecast_mean + safety_factor * sqrt(protection_period) * forecast_std`

The order closes the gap between this target and current inventory position.

### PPO and SAC

The same continuous-action environment supports Stable-Baselines3 PPO and SAC. RL dependencies are optional so classical experiments and CI do not need PyTorch.

## KPIs

Policies are evaluated on identical seeded demand scenarios using:

- total inventory cost,
- service level / fill rate,
- stockout days,
- final inventory,
- final backlog,
- total ordered quantity,
- cumulative return.

Forecast models are evaluated independently with forecast error such as MAE.

This separation matters: a forecasting model may improve MAE while still producing inferior replenishment decisions because inventory cost is asymmetric and lead-time dependent.

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── src/
│   └── demand_inventory/
│       ├── __init__.py
│       ├── forecasting.py
│       ├── environment.py
│       ├── baselines.py
│       ├── train_rl.py
│       └── evaluate.py
├── tests/
│   ├── test_forecasting.py
│   ├── test_environment.py
│   └── test_baselines.py
└── .github/workflows/ci.yml
```

## Installation

Classical forecasting/control and tests:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
```

For PPO/SAC:

```bash
pip install -e '.[rl,dev]'
```

## Evaluate classical policies

```bash
python -m demand_inventory.evaluate --episodes 30 --seed 100
```

Episode-level results are written to `results/comparison.csv`.

## Train SAC

```bash
python -m demand_inventory.train_rl --algo sac --timesteps 100000
```

## Train PPO

```bash
python -m demand_inventory.train_rl --algo ppo --timesteps 100000
```

## Compare trained RL models

```bash
python -m demand_inventory.evaluate \
  --episodes 50 \
  --ppo-model artifacts/ppo_inventory_control.zip \
  --sac-model artifacts/sac_inventory_control.zip
```

## Experimental design

For a defensible forecasting + control study:

1. generate or load a historical demand training segment,
2. fit forecasting models without future leakage,
3. compare forecast metrics such as MAE/RMSE and calibration,
4. use each model inside the same inventory policy,
5. evaluate all control policies on identical unseen demand seeds,
6. report both forecasting metrics and operational KPIs,
7. test robustness under demand trend shifts, volatility changes and longer lead times.

## Industrial-engineering relevance

The project connects:

- demand forecasting,
- inventory theory,
- safety-stock design,
- stochastic simulation,
- reinforcement learning,
- service-level optimization,
- cost-to-serve analysis.

The central decision question is not simply “Which model predicts demand best?” but “Which forecasting-and-control architecture gives the best operational outcome under uncertainty?”

## Research extensions

Useful next steps include:

- probabilistic forecasts and quantile-based safety stock,
- XGBoost/LightGBM demand models,
- GRU/LSTM forecasting,
- intermittent-demand methods such as Croston variants,
- multi-echelon inventory networks,
- perishable inventory,
- supplier reliability and stochastic lead times,
- capacity-constrained replenishment,
- MILP/rolling-horizon optimization baseline,
- distributional RL,
- constrained service-level RL,
- offline RL from ERP transaction history,
- forecast-value-of-information analysis,
- statistical significance tests over evaluation seeds.

## CI

GitHub Actions tests Python 3.10, 3.11 and 3.12. CI installs only the lightweight classical dependencies, runs unit tests, and smoke-tests the baseline policy evaluation. Long PPO/SAC training is intentionally excluded.
