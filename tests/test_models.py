import pytest

from multi_echelon_inventory import InventoryNode


def make_node(**overrides):
    values = {
        "name": "Retailer",
        "initial_on_hand": 100,
        "demand_mean": 20,
        "demand_std": 4,
        "lead_time": 5,
        "holding_cost": 1.0,
        "shortage_cost": 9.0,
        "service_level": 0.95,
    }
    values.update(overrides)
    return InventoryNode(**values)


def test_reorder_point_exceeds_mean_lead_time_demand() -> None:
    node = make_node()
    assert node.reorder_point() > node.lead_time_demand_mean()


def test_invalid_holding_cost_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_node(holding_cost=0)


def test_invalid_service_level_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_node(service_level=1.0)
