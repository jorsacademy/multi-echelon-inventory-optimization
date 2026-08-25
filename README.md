# Multi-Echelon Inventory Optimization

A compact Python reference implementation for serial multi-echelon inventory planning and discrete-time stochastic simulation.

This repository is intended for educational, research, and non-commercial use. It is not a production ERP, MRP, or supply-chain optimization engine.

## What this project does

The package models a serial supply chain whose nodes are ordered from the most upstream echelon to the customer-facing echelon. It includes:

- reorder-point and base-stock calculations,
- lead-time demand uncertainty,
- cycle-service-level-based safety stock,
- a newsvendor-style critical-ratio policy,
- an echelon-aware base-stock heuristic,
- stochastic external customer demand,
- inventory-position-based replenishment ordering,
- physical shipment coupling between adjacent echelons,
- replenishment lead-time pipelines,
- optional per-period shipment capacities,
- backorders,
- holding, shortage, and ordering costs,
- fill-rate and cycle-service-level reporting,
- deterministic random seeds for reproducible experiments,
- automated tests through GitHub Actions.

## Important modeling note

The original prototype used a square-root expression based on mean demand, holding cost, and stockout cost and labeled the result as echelon safety stock. That expression is not a standard multi-echelon safety-stock optimization formula. It also generated stock levels directly from a normal distribution instead of simulating material flows.

The current implementation replaces those assumptions with explicit inventory dynamics. Replenishment orders propagate upstream, physical shipments consume upstream on-hand inventory, shipments can be capacity constrained, and goods arrive downstream after the receiving node's lead time.

The included echelon base-stock policy is deliberately transparent. It propagates the customer-facing demand process upstream and increases cumulative lead-time exposure for higher echelons. It is a heuristic, not an exact Clark-Scarf dynamic-programming solution.

## Project structure

```text
multi-echelon-inventory-optimization/
├── .github/
│   └── workflows/
│       └── tests.yml
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
        demand_mean=20,
        demand_std=6,
        lead_time=10,
        holding_cost=0.10,
        shortage_cost=100,
        ordering_cost=25,
        service_level=0.95,
        shipment_capacity=80,
    ),
    InventoryNode(
        name="Manufacturer",
        initial_on_hand=750,
        demand_mean=20,
        demand_std=6,
        lead_time=5,
        holding_cost=0.20,
        shortage_cost=200,
        ordering_cost=20,
        service_level=0.95,
        shipment_capacity=60,
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

## Serial material-flow convention

Nodes must be supplied in upstream-to-downstream order:

```text
External Source -> Supplier -> Manufacturer -> Distributor -> Customer
```

Only the final node receives exogenous stochastic customer demand. Upstream demand is generated endogenously by replenishment orders from the immediate downstream node.

For example, a replenishment order placed by the Distributor becomes demand for the Manufacturer. The Manufacturer can ship only what it physically has on hand and, when configured, no more than its `shipment_capacity` for that period. That shipment then enters the Distributor's inbound pipeline and arrives after the Distributor's configured `lead_time`.

The first node orders from an unconstrained external source and receives those orders after its own lead time.

## Policy calculations

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

The ratio is converted to a standard-normal quantile and then applied to lead-time demand. The ratio is not multiplied directly by the reorder point.

### Echelon base-stock heuristic

In a serial chain, demand should not be summed across echelons because the same material flow propagates upstream as replenishment orders. The heuristic therefore uses the customer-facing demand mean and variance and increases cumulative lead-time exposure for upstream echelons.

This is more consistent with a serial material-flow model than independently aggregating every node's demand process. It remains a heuristic rather than an exact globally optimal multi-echelon policy.

## Simulation sequence

For each period, the simulator:

1. receives shipments whose lead times have expired,
2. attempts to clear previously backordered internal and customer demand,
3. generates new stochastic demand at the customer-facing node,
4. propagates replenishment decisions from downstream to upstream,
5. ships material from upstream nodes subject to on-hand inventory and shipment capacity,
6. places the most-upstream node's replenishment order with the external source,
7. records inventory, service, flow, and cost metrics.

## Output metrics

The period history includes demand, immediate demand fill, arrivals, shipments, replenishment orders, on-hand inventory, on-order inventory, backorders, inventory position, policy target, and costs.

The node-level summary includes:

- average on-hand inventory,
- average inventory position,
- average backorders,
- demand units,
- immediately filled units,
- fill rate,
- cycle service level,
- stockout periods,
- holding cost,
- shortage cost,
- ordering cost,
- total cost.

## Assumptions and limitations

Current assumptions include discrete periods, normally distributed customer demand, optional truncation of demand at zero, integer deterministic lead times, backordering, a serial topology, one immediate upstream supplier per node, and an unconstrained external source for the most-upstream node.

Production multi-echelon optimization would normally require additional features such as correlated or non-normal demand, stochastic lead times, order capacities in addition to shipment capacities, lot sizing, minimum order quantities, production transformations and yields, multiple products, bill-of-material relationships, branching networks, allocation rules among multiple downstream nodes, parameter estimation, service-level constraints, and exact or approximate stochastic optimization algorithms.

## Testing

Run the local test suite with:

```bash
pytest
```

GitHub Actions runs the same test suite automatically on pushes and pull requests to `main` across the configured Python versions.

## License

This repository is source-available for non-commercial use only. See `LICENSE` for the complete terms.
