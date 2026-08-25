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
