import numpy as np

from robust_inventory import DemandScenarioGenerator, InventoryParameters


def test_scenarios_are_reproducible():
    params = InventoryParameters(periods=4)
    a, pa = DemandScenarioGenerator(params, seed=7).generate_scenarios(5)
    b, pb = DemandScenarioGenerator(params, seed=7).generate_scenarios(5)
    assert a == b
    assert np.array_equal(pa, pb)
    assert np.isclose(pa.sum(), 1.0)


def test_only_active_segments_are_generated():
    params = InventoryParameters(periods=2)
    scenarios, _ = DemandScenarioGenerator(params).generate_scenarios(1)
    assert ("Large_Teaching", "Rural") not in scenarios[0]
    assert ("Large_Teaching", "Metropolitan") in scenarios[0]
