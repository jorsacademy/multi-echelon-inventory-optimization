import pytest

from robust_inventory import InventoryParameters


def test_default_parameters_are_valid():
    p = InventoryParameters()
    assert p.lead_time == 3
    assert p.max_shelf_life == 18
    assert p.service_level == pytest.approx(0.95)


def test_invalid_service_level_rejected():
    with pytest.raises(ValueError):
        InventoryParameters(service_level=1.0)
