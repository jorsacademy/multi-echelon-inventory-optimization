import pandas as pd

from multi_echelon_inventory import (
    InventoryNode,
    MultiEchelonSystem,
    PolicyType,
    SimulationConfig,
)


def make_system() -> MultiEchelonSystem:
    return MultiEchelonSystem(
        [
            InventoryNode(
                name="Retailer",
                initial_on_hand=100,
                demand_mean=20,
                demand_std=4,
                lead_time=2,
                holding_cost=1.0,
                shortage_cost=9.0,
                ordering_cost=2.0,
                service_level=0.95,
            )
        ]
    )


def test_simulation_is_reproducible_for_fixed_seed() -> None:
    system = make_system()
    config = SimulationConfig(periods=30, seed=123)

    first = system.simulate(config, PolicyType.BASE_STOCK)
    second = system.simulate(config, PolicyType.BASE_STOCK)

    pd.testing.assert_frame_equal(first.history, second.history)
    pd.testing.assert_frame_equal(first.summary, second.summary)


def test_simulation_produces_expected_number_of_rows() -> None:
    result = make_system().simulate(SimulationConfig(periods=15, seed=1))
    assert len(result.history) == 15
    assert len(result.summary) == 1


def test_fill_rate_is_bounded() -> None:
    result = make_system().simulate(SimulationConfig(periods=60, seed=7))
    fill_rate = float(result.summary.loc[0, "Fill Rate"])
    assert 0.0 <= fill_rate <= 1.0


def test_internal_orders_become_upstream_demand() -> None:
    system = MultiEchelonSystem(
        [
            InventoryNode("Supplier", 100, 5, 0, 1, 1.0, 10.0),
            InventoryNode("Retailer", 0, 5, 0, 1, 1.0, 10.0),
        ]
    )
    result = system.simulate(
        SimulationConfig(periods=3, seed=1),
        PolicyType.BASE_STOCK,
    )
    supplier = result.history[result.history["Node"] == "Supplier"]
    assert supplier["Demand"].sum() > 0


def test_shipment_capacity_limits_physical_flow() -> None:
    system = MultiEchelonSystem(
        [
            InventoryNode(
                "Supplier", 100, 20, 0, 1, 1.0, 10.0, shipment_capacity=3
            ),
            InventoryNode("Retailer", 0, 20, 0, 1, 1.0, 10.0),
        ]
    )
    result = system.simulate(
        SimulationConfig(periods=5, seed=1),
        PolicyType.BASE_STOCK,
    )
    supplier = result.history[result.history["Node"] == "Supplier"]
    assert (supplier["Shipments"] <= 3.0).all()


def test_production_capacity_limits_physical_flow() -> None:
    system = MultiEchelonSystem(
        [
            InventoryNode(
                "Supplier", 100, 20, 0, 1, 1.0, 10.0, production_capacity=4
            ),
            InventoryNode("Retailer", 0, 20, 0, 1, 1.0, 10.0),
        ]
    )
    result = system.simulate(
        SimulationConfig(periods=5, seed=1),
        PolicyType.BASE_STOCK,
    )
    supplier = result.history[result.history["Node"] == "Supplier"]
    assert (supplier["Shipments"] <= 4.0).all()


def test_order_capacity_limits_replenishment_orders() -> None:
    system = MultiEchelonSystem(
        [
            InventoryNode(
                "Retailer",
                0,
                20,
                0,
                2,
                1.0,
                10.0,
                order_capacity=3,
            )
        ]
    )
    result = system.simulate(
        SimulationConfig(periods=5, seed=1),
        PolicyType.BASE_STOCK,
    )
    assert (result.history["Order Quantity"] <= 3.0).all()


def test_stochastic_lead_time_generates_variable_samples() -> None:
    system = MultiEchelonSystem(
        [
            InventoryNode(
                "Retailer",
                0,
                20,
                0,
                3,
                1.0,
                10.0,
                lead_time_std=1.5,
                order_capacity=10,
            )
        ]
    )
    result = system.simulate(
        SimulationConfig(periods=30, seed=9),
        PolicyType.BASE_STOCK,
    )
    samples = result.history["Scheduled Lead Time"].dropna()
    assert len(samples) > 1
    assert samples.nunique() > 1
    assert (samples >= 0).all()


def test_open_upstream_orders_are_included_in_inventory_position() -> None:
    system = MultiEchelonSystem(
        [
            InventoryNode(
                "Supplier", 0, 20, 0, 1, 1.0, 10.0, production_capacity=1
            ),
            InventoryNode("Retailer", 0, 20, 0, 2, 1.0, 10.0),
        ]
    )
    result = system.simulate(
        SimulationConfig(periods=3, seed=1),
        PolicyType.BASE_STOCK,
    )
    retailer = result.history[result.history["Node"] == "Retailer"]
    assert (retailer["Open Upstream Order"] >= 0).all()
    assert (retailer["Order Quantity"] <= retailer["Target"] + 1e-9).all()


def test_only_customer_facing_node_receives_external_stochastic_demand() -> None:
    system = MultiEchelonSystem(
        [
            InventoryNode("Supplier", 100, 20, 4, 1, 1.0, 10.0),
            InventoryNode("Retailer", 100, 20, 4, 1, 1.0, 10.0),
        ]
    )
    result = system.simulate(SimulationConfig(periods=2, seed=7))
    retailer = result.history[result.history["Node"] == "Retailer"]
    assert (retailer["Demand"] > 0).all()
