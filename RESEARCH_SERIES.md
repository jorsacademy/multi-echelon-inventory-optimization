# Inventory and Replenishment Optimization Research Series

This file maps inventory-related repositories across deterministic planning, stochastic control, robust optimization, simulation, and learning-based decision methods. It is an index only: every listed repository remains an independent project.

## Inventory planning foundations

- `seasonal-inventory-planning-python` — compact seasonal inventory-planning model.
- `multi-echelon-inventory-optimization` — serial multi-echelon replenishment, simulation, policy comparison, service-level optimization, and Monte Carlo analysis.
- `multi-period-warehouse-rental-lp-optimization` — capacity/rental planning over time; related to inventory infrastructure but a distinct LP decision layer.

## Forecast-then-control and contextual inventory

- `demand-forecasting-plus-inventory-control` — explicit forecasting plus downstream inventory-control pipeline.
- `contextual-optimization-newsvendor` — context-conditioned newsvendor decisions.
- `predict-then-optimize-production-planning-spo-plus-pytorch` — not an inventory-only project, but a useful bridge from prediction to downstream production/inventory decisions.

## Stochastic and probabilistic formulations

- `chance-constrained-inventory-optimization-python` — probabilistic service-level constraints.
- `pymdptoolbox-inventory-control` — inventory as a Markov decision process.
- `approximate-dynamic-programming-fleet-inventory` — approximate dynamic programming for sequential inventory/fleet decisions.
- `hyperopt-inventory-policy-optimization` — black-box search over inventory-policy parameters.

## Robust and distributionally robust inventory

- `robust-healthcare-inventory-optimization` — robust inventory decisions in a healthcare context.
- `wasserstein-dro-inventory-optimization-python` — Wasserstein distributionally robust inventory optimization.
- `conformal-prediction-robust-inventory-optimization-python` — predictive uncertainty from conformal methods connected to robust inventory decisions.

## Why these repositories stay separate

Inventory research changes materially depending on:

- single-echelon versus multi-echelon structure;
- static planning versus sequential policy control;
- known distributions versus scenarios versus ambiguity sets;
- service-level constraints versus expected-cost objectives;
- exact optimization versus simulation-based policy search;
- forecast-then-optimize versus decision-focused or contextual learning.

Therefore, inventory is a useful portfolio series, but not a reason to collapse the projects into one repository.
