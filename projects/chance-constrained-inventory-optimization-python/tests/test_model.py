import math

from scipy.stats import norm

from chance_constrained_inventory import (
    InventoryParameters,
    analytical_solution,
    chance_constraint_lower_bound,
    expected_cost,
    service_probability,
    solve_numerically,
    unconstrained_newsvendor_quantity,
)


def test_chance_constraint_lower_bound_matches_quantile():
    params = InventoryParameters()
    expected = params.mu + params.sigma * norm.ppf(params.alpha)
    assert math.isclose(
        chance_constraint_lower_bound(params),
        expected,
        rel_tol=0.0,
        abs_tol=1e-10,
    )


def test_baseline_service_bound_is_about_132_8971():
    params = InventoryParameters()
    assert math.isclose(
        chance_constraint_lower_bound(params),
        132.89707253902944,
        rel_tol=0.0,
        abs_tol=1e-8,
    )


def test_unconstrained_solution_uses_critical_fractile():
    params = InventoryParameters()
    critical_fractile = params.shortage_cost / (
        params.shortage_cost + params.holding_cost
    )
    expected = params.mu + params.sigma * norm.ppf(critical_fractile)
    assert math.isclose(
        unconstrained_newsvendor_quantity(params),
        expected,
        rel_tol=0.0,
        abs_tol=1e-10,
    )


def test_chance_constraint_binds_for_baseline_case():
    params = InventoryParameters()
    q = analytical_solution(params)
    assert math.isclose(
        q,
        chance_constraint_lower_bound(params),
        rel_tol=0.0,
        abs_tol=1e-10,
    )


def test_analytical_solution_satisfies_service_requirement():
    params = InventoryParameters()
    q = analytical_solution(params)
    assert service_probability(q, params) >= params.alpha - 1e-12


def test_numerical_solution_matches_analytical_solution():
    params = InventoryParameters()
    result = solve_numerically(params)
    q_numerical = float(result.x[0])
    q_analytical = analytical_solution(params)
    assert math.isclose(
        q_numerical,
        q_analytical,
        rel_tol=0.0,
        abs_tol=1e-5,
    )


def test_expected_cost_is_finite_and_positive():
    params = InventoryParameters()
    q = analytical_solution(params)
    cost = expected_cost(q, params)
    assert math.isfinite(cost)
    assert cost > 0
