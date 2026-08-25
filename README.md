# Multi-Echelon Inventory Optimization

A Python reference implementation for serial multi-echelon inventory planning, capacity-constrained material flow, stochastic simulation, simulation optimization, and Monte Carlo analysis.

This repository is intended for educational, research, and non-commercial use. It is not a production ERP, MRP, or commercial supply-chain optimization engine.

## Features

The package includes:

- serial upstream-to-downstream material flow,
- stochastic customer demand,
- stochastic non-negative integer lead times,
- lead-time-demand uncertainty in safety-stock calculations,
- replenishment pipelines and open upstream orders,
- backorders,
- per-period order, production, and shipment capacities,
- holding, shortage, and fixed ordering costs,
- base-stock policy,
- critical-ratio policy,
- echelon base-stock heuristic,
- heuristic `(s, S)` policy,
- heuristic `(R, Q)` reorder-point/fixed-quantity policy,
- Monte Carlo policy comparison using common random numbers,
- confidence intervals for Monte Carlo mean metrics,
- service-level-constrained simulation optimization,
- one-factor-at-a-time Monte Carlo sensitivity analysis,
- reproducible random seeds,
- automated tests on Python 3.10 through 3.13.

## Modeling scope

Nodes are supplied in upstream-to-downstream order:

```text
External Source -> Supplier -> Manufacturer -> Distributor -> Customer
```

Only the final node receives exogenous stochastic customer demand. Replenishment orders propagate upstream and physical material propagates downstream. Internal shipments consume upstream on-hand inventory and are constrained by the tighter of production and shipment capacity when those limits are configured.

The first node replenishes from an unconstrained external source. All other nodes receive material only through their immediate upstream node.

The project deliberately distinguishes transparent reference heuristics from exact stochastic optimization. The echelon policy is not an exact Clark-Scarf dynamic-programming solution, and the service-level optimizer is a finite simulation-based grid search rather than a mathematical-programming solver.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest
```

## Basic model

```python
from multi_echelon_inventory import (
    InventoryNode,
    MultiEchelonSystem,
    PolicyType,
    SimulationConfig,
)

nodes = [
    InventoryNode(
        name="Supplier",
        initial_on_hand=1000,
        demand_mean=20,
        demand_std=6,
        lead_time=10,
        lead_time_std=2.0,
        holding_cost=0.10,
        shortage_cost=100,
        ordering_cost=25,
        service_level=0.95,
        order_capacity=100,
        production_capacity=70,
        shipment_capacity=80,
    ),
    InventoryNode(
        name="Manufacturer",
        initial_on_hand=750,
        demand_mean=20,
        demand_std=6,
        lead_time=5,
        lead_time_std=1.0,
        holding_cost=0.20,
        shortage_cost=200,
        ordering_cost=20,
        service_level=0.95,
        order_capacity=80,
        production_capacity=55,
        shipment_capacity=60,
    ),
    InventoryNode(
        name="Distributor",
        initial_on_hand=500,
        demand_mean=20,
        demand_std=6,
        lead_time=3,
        lead_time_std=0.75,
        holding_cost=0.15,
        shortage_cost=150,
        ordering_cost=15,
        service_level=0.95,
        order_capacity=60,
        shipment_capacity=50,
    ),
]

system = MultiEchelonSystem(nodes)
result = system.simulate(
    SimulationConfig(periods=365, seed=42),
    policy=PolicyType.ECHELON_BASE_STOCK,
)
print(result.summary)
```

## Lead-time uncertainty

For independent demand and lead time, planning calculations use:

```text
Var(D during L) = E[L] * Var(D) + E[D]^2 * Var(L)
```

When `lead_time_std` is positive, each inbound movement samples a normal lead time, rounds it to the nearest integer, and truncates it at zero.

## Capacity constraints

Each node can specify three independent limits:

```text
order_capacity       maximum replenishment order placed per period
production_capacity  maximum outbound fulfillment per period
shipment_capacity    maximum outbound transport/dispatch per period
```

`None` means unconstrained. Physical outbound flow is limited by on-hand inventory and the tighter of production and shipment capacity.

## Policies

### Base stock

The base-stock target uses mean lead-time demand plus service-level safety stock. Orders replenish the inventory position toward the target.

### Critical ratio

The critical ratio is:

```text
shortage_cost / (shortage_cost + holding_cost)
```

It is converted to a standard-normal quantile and applied to lead-time demand.

### Echelon base stock

The customer-facing demand process is propagated upstream while cumulative mean lead time and lead-time variance increase for higher echelons. This is a transparent serial-network heuristic rather than an exact globally optimal solution.

### `(s, S)`

The implementation uses:

```text
s = service-level reorder point
S = s + EOQ-style lot size
```

When inventory position is at or below `s`, the node orders enough to move toward `S`, subject to `order_capacity`.

### `(R, Q)`

Here `R` is the reorder point and `Q` is a fixed EOQ-style replenishment quantity:

```text
R = service-level reorder point
Q = sqrt(2 * ordering_cost * demand_mean / holding_cost)
```

When inventory position is at or below `R`, the node places `Q`, subject to `order_capacity`. This repository uses `R` to mean reorder point, not review interval.

## Monte Carlo policy comparison

```python
from multi_echelon_inventory import compare_policies

comparison = compare_policies(
    system,
    runs=500,
    periods=365,
    base_seed=123,
    policies=[
        PolicyType.BASE_STOCK,
        PolicyType.CRITICAL_RATIO,
        PolicyType.ECHELON_BASE_STOCK,
        PolicyType.S_S,
        PolicyType.R_Q,
    ],
    confidence_level=0.95,
)

print(comparison.policy_summary)
```

Each replication uses the same seed across policies. The summary reports mean cost, cost standard deviation, P05/P50/P95 cost quantiles, customer fill rate, customer cycle service level, network inventory, network backorders, and confidence intervals around key replication means.

The intervals quantify Monte Carlo sampling uncertainty. They do not represent a guarantee that the modeled supply chain matches real operations.

## Service-level-constrained simulation optimization

`optimize_service_level` performs a finite grid search over a common service-level parameter and selected policies. It minimizes simulated mean total cost among candidates satisfying customer-service constraints.

```python
from multi_echelon_inventory import optimize_service_level

optimization = optimize_service_level(
    system,
    service_levels=[0.90, 0.925, 0.95, 0.975, 0.99],
    runs=200,
    periods=365,
    base_seed=123,
    policies=[
        PolicyType.BASE_STOCK,
        PolicyType.ECHELON_BASE_STOCK,
        PolicyType.S_S,
        PolicyType.R_Q,
    ],
    min_customer_fill_rate=0.97,
    min_customer_cycle_service_level=0.90,
)

print(optimization.candidates)
print(optimization.best_policy)
print(optimization.best_service_level)
print(optimization.best_mean_total_cost)
```

If no candidate satisfies the requested constraints, `optimization.feasible` is `False`. A denser grid and more Monte Carlo replications should be used when candidate costs are close.

## Sensitivity analysis

`sensitivity_analysis` performs one-factor-at-a-time scenario analysis. A parameter can be scaled across all nodes or at one named node.

```python
from multi_echelon_inventory import sensitivity_analysis

sensitivity = sensitivity_analysis(
    system,
    parameter="demand_std",
    multipliers=[0.75, 1.0, 1.25, 1.5],
    runs=100,
    periods=365,
    base_seed=123,
)

print(sensitivity.scenarios)
print(sensitivity.policy_results)
```

Supported parameters include demand mean and standard deviation, mean and standard deviation of lead time, holding cost, shortage cost, ordering cost, and configured order/production/shipment capacities.

## Output metrics

Period history includes demand, immediate fill, arrivals, shipments, replenishment orders, on-hand inventory, pipeline inventory, open upstream orders, backorders, inventory position, policy target, scheduled lead time, and costs.

Node summaries include average on-hand inventory, average inventory position, average backorders, demand units, immediate fill units, fill rate, cycle service level, stockout periods, holding cost, shortage cost, ordering cost, and total cost.

## Project structure

```text
multi-echelon-inventory-optimization/
├── .github/workflows/tests.yml
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── examples/
│   └── three_echelon_example.py
├── src/multi_echelon_inventory/
│   ├── __init__.py
│   ├── metrics.py
│   ├── models.py
│   ├── monte_carlo.py
│   ├── optimization.py
│   ├── policies.py
│   ├── sensitivity.py
│   └── simulation.py
└── tests/
    ├── test_advanced_analysis.py
    ├── test_models.py
    ├── test_monte_carlo.py
    ├── test_policies.py
    └── test_simulation.py
```

## Limitations

The current model assumes a serial, single-product network, normally distributed customer demand, normal lead-time perturbations truncated at zero, backordering, one immediate upstream source per node, and no explicit production transformation or bill of materials.

It does not yet implement branching networks, multiple products, correlated demand, empirical distributions, minimum order quantities, setup-time queues, production yields, allocation among competing downstream nodes, parameter estimation, exact Clark-Scarf optimization, or general stochastic mathematical programming.

## Testing

```bash
pytest
```

GitHub Actions runs the test suite on pushes and pull requests to `main` for Python 3.10, 3.11, 3.12, and 3.13.

## License

This repository is source-available for non-commercial use only. See `LICENSE` for the complete terms.
