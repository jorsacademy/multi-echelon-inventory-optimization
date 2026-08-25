import pandas as pd
import pytest

from multi_echelon_inventory import (
    InventoryNode,
    MultiEchelonSystem,
    PolicyType,
    compare_policies,
)


def make_system() -> MultiEchelonSystem:
    return MultiEchelonSystem(
        [
            InventoryNode(
                "Supplier",
                150,
                20,
                4,
                2,
                0.5,
                10.0,
                lead_time_std=0.5,
                order_capacity=60,
                production_capacity=50,
            ),
            InventoryNode(
                "Retailer",
                80,
                20,
                4,
                2,
                1.0,
                15.0,
                lead_time_std=0.5,
                order_capacity=40,
            ),
        ]
    )


def test_policy_comparison_returns_expected_shapes() -> None:
    result = compare_policies(
        make_system(),
        runs=4,
        periods=20,
        base_seed=123,
        policies=[PolicyType.BASE_STOCK, PolicyType.ECHELON_BASE_STOCK],
    )

    assert len(result.run_results) == 8
    assert len(result.node_results) == 16
    assert len(result.policy_summary) == 2
    assert set(result.policy_summary["Policy"]) == {
        PolicyType.BASE_STOCK.value,
        PolicyType.ECHELON_BASE_STOCK.value,
    }


def test_policy_comparison_is_reproducible() -> None:
    kwargs = {
        "runs": 3,
        "periods": 15,
        "base_seed": 99,
        "policies": [PolicyType.BASE_STOCK, PolicyType.CRITICAL_RATIO],
    }
    first = compare_policies(make_system(), **kwargs)
    second = compare_policies(make_system(), **kwargs)

    pd.testing.assert_frame_equal(first.run_results, second.run_results)
    pd.testing.assert_frame_equal(first.node_results, second.node_results)
    pd.testing.assert_frame_equal(first.policy_summary, second.policy_summary)


def test_same_replication_seed_is_shared_across_policies() -> None:
    result = compare_policies(
        make_system(),
        runs=5,
        periods=10,
        base_seed=7,
        policies=[PolicyType.BASE_STOCK, PolicyType.CRITICAL_RATIO],
    )
    seed_counts = result.run_results.groupby("Run")["Seed"].nunique()
    assert (seed_counts == 1).all()


def test_invalid_monte_carlo_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        compare_policies(make_system(), runs=0)
    with pytest.raises(ValueError):
        compare_policies(make_system(), periods=0)
    with pytest.raises(ValueError):
        compare_policies(make_system(), policies=[])
