# Multi-Echelon Inventory Optimization

A compact Python reference implementation for serial multi-echelon inventory planning, capacity-constrained material flow, and discrete-time stochastic simulation.

This repository is intended for educational, research, and non-commercial use. It is not a production ERP, MRP, or supply-chain optimization engine.

## What this project does

The package models a serial supply chain whose nodes are ordered from the most upstream echelon to the customer-facing echelon. It includes:

- reorder-point and base-stock calculations,
- demand and lead-time uncertainty,
- cycle-service-level-based safety stock,
- a newsvendor-style critical-ratio policy,
- an echelon-aware base-stock heuristic,
- stochastic external customer demand,
- stochastic non-negative integer lead times,
- physical shipment coupling between adjacent echelons,
- inventory-position-based replenishment ordering,
- open upstream replenishment orders,
- per-period replenishment order capacity,
- per-period production/fulfillment capacity,
- per-period shipment/transport capacity,
- replenishment pipelines,
- backorders,
- holding, shortage, and ordering costs,
- fill-rate and cycle-service-level reporting,
- Monte Carlo policy comparison,
- common random-number seeds across policy replications,
- deterministic random seeds for reproducible experiments,
- automated tests through GitHub Actions.

## Important modeling note

The original prototype used a square-root expression based on mean demand, holding cost, and stockout cost and labeled the result as echelon safety stock. That expression is not a standard multi-echelon safety-stock optimization formula. It also generated stock levels directly from a normal distribution instead of simulating material flows.

The current implementation replaces those assumptions with explicit inventory dynamics. Replenishment orders propagate upstream, physical shipments consume upstream on-hand inventory, shipments and production can be capacity constrained, and goods arrive downstream through deterministic or stochastic lead-time pipelines.

The included echelon base-stock policy is deliberately transparent. It propagates the customer-facing demand process upstream and increases cumulative replenishment exposure for higher echelons. It is a heuristic, not an exact Clark-Scarf dynamic-programming solution.

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
│       ├── monte_carlo.py
│       ├── policies.py
│       └── simulation.py
└── tests/
    ├── test_models.py
    ├── test_monte_carlo.py
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
    compare_policies,
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

comparison = compare_policies(
    system,
    runs=100,
    periods=365,
    base_seed=42,
)
print(comparison.policy_summary)
```

A complete runnable example is available in `examples/three_echelon_example.py`.

## Serial material-flow convention

Nodes must be supplied in upstream-to-downstream order:

```text
External Source -> Supplier -> Manufacturer -> Distributor -> Customer
```

Only the final node receives exogenous stochastic customer demand. Upstream demand is generated endogenously by replenishment orders from the immediate downstream node.

A downstream replenishment order becomes demand for the immediate upstream node. If the upstream node cannot fill that order because of insufficient stock, shipment capacity, or production capacity, the unfilled quantity remains an upstream backorder. That open upstream order is included in the downstream node's inventory position, preventing repeated ordering of the same outstanding quantity.

The first node orders from an unconstrained external source and receives accepted replenishment orders through its own stochastic inbound lead-time process.

## Stochastic lead times

`lead_time` is the mean inbound lead time in periods. `lead_time_std` controls lead-time uncertainty. When `lead_time_std > 0`, each physical inbound movement samples a lead time from a normal distribution, rounds it to the nearest integer, and truncates it at zero.

For planning calculations, lead-time demand variance uses the standard independent-demand/lead-time approximation:

```text
Var(D during L) = E[L] * Var(D) + E[D]^2 * Var(L)
```

For fixed lead times, this reduces to:

```text
lead-time demand std = demand_std * sqrt(lead_time)
```

The echelon heuristic adds independent lead-time variances across the cumulative replenishment path.

## Capacity constraints

Three distinct optional capacity parameters are available on each `InventoryNode`:

```text
order_capacity       maximum replenishment order placed per period
production_capacity  maximum physical outbound fulfillment per period
shipment_capacity    maximum physical outbound transport/dispatch per period
```

Physical outbound flow is constrained by on-hand inventory and by the tighter of `production_capacity` and `shipment_capacity` when both are configured.

These parameters are intentionally separate: an echelon may be able to request more material than it can produce or dispatch, and a transport bottleneck can be tighter than a production bottleneck.

## Policy calculations

### Base stock

For mean demand `mu`, demand standard deviation `sigma`, mean lead time `E[L]`, lead-time standard deviation `sigma_L`, and target service level `alpha`:

```text
lead-time mean demand = mu * E[L]
lead-time variance    = E[L] * sigma^2 + mu^2 * sigma_L^2
safety stock          = z(alpha) * sqrt(lead-time variance)
base-stock target     = lead-time mean demand + safety stock
```

### Critical ratio

The critical-ratio policy uses:

```text
critical ratio = shortage cost / (shortage cost + holding cost)
```

The ratio is converted to a standard-normal quantile and applied to lead-time demand. It is not multiplied directly by a reorder point.

### Echelon base-stock heuristic

In a serial chain, the same material flow propagates upstream. The heuristic therefore uses the customer-facing demand process rather than summing installation demand across echelons. Higher echelons receive greater cumulative lead-time exposure, including cumulative lead-time variance.

This remains a transparent heuristic rather than an exact globally optimal multi-echelon policy.

## Simulation sequence

For each period, the simulator:

1. receives inbound shipments whose sampled lead times have expired,
2. attempts to clear existing internal backorders subject to outbound capacity,
3. attempts to clear existing external customer backorders,
4. generates new stochastic customer demand,
5. fulfills current customer demand subject to stock and outbound capacity,
6. calculates replenishment orders from downstream to upstream,
7. caps new orders by `order_capacity`,
8. ships internal material subject to stock, production capacity, and shipment capacity,
9. samples inbound lead times for dispatched material,
10. places the most-upstream node's replenishment order with the external source,
11. records inventory, flow, service, lead-time, and cost metrics.

## Monte Carlo policy comparison

`compare_policies` repeatedly simulates selected policies and returns a `MonteCarloResult` with three DataFrames:

```text
run_results      one system-level row per policy and replication
node_results     node-level KPIs for every policy and replication
policy_summary   aggregated policy statistics across replications
```

The policy summary includes mean total cost, total-cost standard deviation, P05/P50/P95 cost quantiles, mean customer fill rate, mean customer cycle service level, mean network on-hand inventory, and mean network backorders.

Each replication shares the same seed across policies. Customer-demand random streams are therefore common across policies, which reduces comparison noise. Lead-time streams use the same seed family but can diverge when different policies create different shipment event paths.

Example:

```python
comparison = compare_policies(
    system,
    runs=500,
    periods=365,
    base_seed=123,
    policies=[
        PolicyType.BASE_STOCK,
        PolicyType.CRITICAL_RATIO,
        PolicyType.ECHELON_BASE_STOCK,
    ],
)

print(comparison.policy_summary)
```

## Output metrics

Period-level history includes demand, immediate demand fill, arrivals, shipments, replenishment orders, on-hand inventory, on-order inventory, open upstream orders, backorders, inventory position, policy target, sampled/scheduled inbound lead time, and costs.

The node-level summary includes average on-hand inventory, average inventory position, average backorders, demand units, immediately filled units, fill rate, cycle service level, stockout periods, holding cost, shortage cost, ordering cost, and total cost.

## Assumptions and limitations

Current assumptions include discrete periods, normally distributed customer demand, optional truncation of demand at zero, normally distributed lead-time samples truncated at zero, backordering, a serial topology, one immediate upstream supplier per node, and an unconstrained external source for the most-upstream node.

`production_capacity` is modeled as a per-period outbound fulfillment constraint. The model does not yet represent explicit multi-stage production transformations, yields, bills of material, setup times, batch production, or work-in-process queues.

Production-grade multi-echelon optimization would normally require additional capabilities such as correlated or non-normal demand, empirical lead-time distributions, minimum order quantities, lot sizing, multiple products, bill-of-material relationships, branching networks, allocation rules among multiple downstream nodes, parameter estimation, service-level constraints, and exact or approximate stochastic optimization algorithms.

## Testing

Run the local test suite with:

```bash
pytest
```

GitHub Actions runs the same test suite automatically on pushes and pull requests to `main` across the configured Python versions.

## License

This repository is source-available for non-commercial use only. See `LICENSE` for the complete terms.
