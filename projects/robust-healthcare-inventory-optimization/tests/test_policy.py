import pytest

from robust_inventory import (
    InventoryParameters,
    InventoryPolicy,
    ReorderRule,
    RobustInventoryOptimizer,
)


def test_monthly_demand_is_not_divided_by_horizon_again():
    opt = RobustInventoryOptimizer(InventoryParameters(periods=3), num_scenarios=2, seed=1)
    seg = opt.generator.active_segments[0]
    manual = sum(s[seg][t] for s in opt.scenarios.values() for t in range(3)) / 6
    assert opt.expected_monthly_demand()[seg] == pytest.approx(manual)


def test_initial_policy_respects_budget_and_capacity():
    params = InventoryParameters(periods=4, budget=10_000, max_capacity=120)
    opt = RobustInventoryOptimizer(params, num_scenarios=5, seed=3)
    policy = opt.solve_heuristic()
    assert policy.total_initial_units <= 120
    assert policy.total_initial_units * params.unit_cost <= params.budget


def test_orders_arrive_only_after_lead_time():
    params = InventoryParameters(periods=3, lead_time=2, max_shelf_life=5, max_capacity=1000)
    opt = RobustInventoryOptimizer(params, num_scenarios=1, seed=2)
    seg = opt.generator.active_segments[0]
    for other in opt.generator.active_segments:
        opt.scenarios[0][other] = [0, 0, 0]
    opt.scenarios[0][seg] = [10, 0, 10]
    policy = InventoryPolicy(
        initial_inventory={},
        reorder_policy={s: ReorderRule(0, 10) for s in opt.generator.active_segments},
    )
    result = opt.simulate_policy_performance(policy)
    assert result.service_levels[seg] == pytest.approx(0.5)


def test_cost_breakdown_reconciles_to_expected_cost():
    opt = RobustInventoryOptimizer(InventoryParameters(periods=2), num_scenarios=3, seed=4)
    policy = opt.solve_heuristic()
    result = opt.simulate_policy_performance(policy)
    assert sum(result.expected_cost_breakdown.values()) == pytest.approx(result.expected_total_cost)


def test_cvar_rejects_invalid_alpha():
    opt = RobustInventoryOptimizer(InventoryParameters(periods=1), num_scenarios=1)
    with pytest.raises(ValueError):
        opt.cvar([1, 2], 1.0)
