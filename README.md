# Multi-Echelon Inventory Optimization

A compact Python reference implementation for serial multi-echelon inventory planning and discrete-time inventory simulation.

This repository is intended for educational, research, and non-commercial use. It is not a production ERP, MRP, or supply-chain optimization engine.

## What this project does

The package models a serial supply chain made of inventory nodes such as suppliers, manufacturers, distributors, and retailers. It includes:

- reorder-point and base-stock calculations,
- lead-time demand uncertainty,
- cycle-service-level-based safety stock,
- a newsvendor-style critical-ratio policy,
- an echelon-aware base-stock heuristic for serial networks,
- discrete-time simulation with stochastic demand,
- replenishment pipelines and lead times,
- backorders,
- inventory-position-based ordering,
- holding, shortage, and ordering cost metrics,
- service-level and fill-rate reporting,
- deterministic random seeds for reproducible experiments.

## Important modeling note

The original prototype used a square-root expression based on mean demand, holding cost, and stockout cost and labeled the result as echelon safety stock. That expression is not a standard multi-echelon safety-stock optimization formula. It also generated simulated stock levels directly from a normal distribution instead of simulating inventory flows.

This version replaces those parts with explicit assumptions and operational inventory dynamics. The echelon policy included here is a transparent heuristic based on aggregated downstream demand moments and cumulative replenishment exposure. It should not be interpreted as an exact Clark-Scarf dynamic-programming solution.

## Project structure

```text
multi-echelon-inventory-optimization/
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── examples/
│   └── three_echelon_example.py
├── src/
│   └── multi_echelon_inventory/
│       ├── __init__.py
│       ├── metrics.py
│       ├── models.py
│       ├── policies.py
│       └── simulation.py
└── tests/
    ├── test_models.py
    ├── test_policies.py
    └── test_simulation.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

For development and tests:

```bash
pip install -e .[dev]
pytest
```

## Example

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
        demand_mean=50,
        demand_std=10,
        lead_time=10,
        holding_cost=0.10,
        shortage_cost=100,
        ordering_cost=25,
        service_level=0.95,
    ),
    InventoryNode(
        name="Manufacturer",
        initial_on_hand=750,
        demand_mean=30,
        demand_std=8,
        lead_time=5,
        holding_cost=0.20,
        shortage_cost=200,
        ordering_cost=20,
        service_level=0.95,
    ),
    InventoryNode(
        name="Distributor",
        initial_on_hand=500,
        demand_mean=20,
        demand_std=6,
        lead_time=3,
        holding_cost=0.15,
        shortage_cost=150,
        ordering_cost=15,
        service_level=0.95,
    ),
]

system = MultiEchelonSystem(nodes)
levels = system.recommended_stock_levels(PolicyType.ECHELON_BASE_STOCK)
print(levels)

result = system.simulate(
    SimulationConfig(periods=365, seed=42),
    policy=PolicyType.ECHELON_BASE_STOCK,
)
print(result.summary)
```

A complete runnable example is available in `examples/three_echelon_example.py`.

## Policies

### Base stock

For a node with mean demand `mu`, demand standard deviation `sigma`, lead time `L`, and target cycle service level `alpha`:

```text
lead-time mean demand = mu * L
lead-time demand std  = sigma * sqrt(L)
safety stock          = z(alpha) * sigma * sqrt(L)
base-stock target     = lead-time mean demand + safety stock
```

The simulator replenishes toward the target using inventory position:

```text
inventory position = on hand + on order - backorders
```

### Critical ratio

The critical-ratio policy uses:

```text
critical ratio = shortage cost / (shortage cost + holding cost)
```

The ratio is converted into a normal-demand quantile instead of being multiplied directly by a reorder point.

### Echelon base-stock heuristic

For each echelon, downstream demand means and variances are aggregated. The target uses cumulative lead-time exposure from the selected node through the downstream end of the serial network. This captures more multi-echelon structure than independent node calculations while remaining simple enough to inspect and test.

It is a heuristic, not an exact globally optimal multi-echelon solution.

## Simulation assumptions

The simulator uses the following assumptions:

- discrete daily periods,
- independent normally distributed external demand at each node,
- demand is truncated at zero,
- integer lead times,
- replenishment orders arrive after the configured lead time,
- shortages are backordered,
- orders replenish from an unconstrained upstream source,
- each node is simulated with its own stochastic demand stream,
- serial-network demand aggregation is used for echelon target calculation, not for material-flow coupling between nodes.

The last assumption is important. This repository provides a useful multi-echelon planning reference model, but it does not yet model physical shipment constraints between adjacent echelons.

## Metrics

The simulation reports, among other fields:

- average on-hand inventory,
- average inventory position,
- average backorders,
- demand units,
- filled units,
- fill rate,
- cycle service level,
- stockout periods,
- holding cost,
- shortage cost,
- ordering cost,
- total cost.

## Scope and limitations

Use this project for education, prototyping, comparative experiments, and as a basis for more advanced research implementations.

For production multi-echelon optimization, additional features would normally be required, including correlated demand, variable lead times, capacity constraints, shipment coupling, service-level constraints, lot sizing, network topology beyond a serial chain, parameter estimation, and exact or approximate stochastic optimization algorithms.

## License

This repository is source-available for non-commercial use only. See `LICENSE` for the terms.
