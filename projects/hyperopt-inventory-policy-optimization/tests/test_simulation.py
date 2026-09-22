import pytest

from hyperopt_inventory.simulation import InventoryConfig, evaluate_policy, simulate_policy


def test_simulation_is_deterministic_for_fixed_seed():
    a = simulate_policy(4, 12, seed=7)
    b = simulate_policy(4, 12, seed=7)
    assert a == pytest.approx(b)


def test_evaluate_policy_averages_fixed_seed_runs():
    seeds = (1, 2, 3)
    expected = sum(simulate_policy(3, 10, seed=seed) for seed in seeds) / len(seeds)
    assert evaluate_policy(3, 10, seeds=seeds) == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"demand_rate": 0},
        {"initial_inventory": -1},
        {"holding_cost": -1},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        InventoryConfig(**kwargs)


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError):
        simulate_policy(-1, 10, seed=1)
    with pytest.raises(ValueError):
        simulate_policy(5, 5, seed=1)
    with pytest.raises(ValueError):
        evaluate_policy(3, 10, seeds=())
