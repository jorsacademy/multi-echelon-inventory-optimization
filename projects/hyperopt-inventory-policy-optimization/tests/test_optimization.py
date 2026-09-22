import pytest

from hyperopt_inventory.optimization import (
    compare_searches,
    objective,
    run_hyperopt,
    search_space,
)


def test_search_space_and_objective_contract():
    space = search_space()
    assert set(space) == {"reorder_point", "gap"}
    result = objective({"reorder_point": 3, "gap": 7})
    assert result["status"] == "ok"
    assert result["reorder_point"] == 3
    assert result["order_up_to"] == 10
    assert result["loss"] > 0


def test_tpe_runs_end_to_end_and_is_reproducible():
    a = run_hyperopt(algorithm="tpe", max_evals=12, seed=7)
    b = run_hyperopt(algorithm="tpe", max_evals=12, seed=7)
    assert a == b
    assert a.evaluations == 12
    assert a.order_up_to > a.reorder_point
    assert a.validation_loss > 0


def test_random_search_runs_with_same_budget():
    result = run_hyperopt(algorithm="random", max_evals=8, seed=3)
    assert result.algorithm == "random"
    assert result.evaluations == 8
    assert result.validation_loss > 0


def test_comparison_is_sorted_by_validation_loss():
    results = compare_searches(max_evals=8, seed=5)
    assert {result.algorithm for result in results} == {"tpe", "random"}
    assert results[0].validation_loss <= results[1].validation_loss


def test_invalid_search_inputs_are_rejected():
    with pytest.raises(ValueError):
        run_hyperopt(max_evals=0)
    with pytest.raises(ValueError):
        run_hyperopt(algorithm="missing", max_evals=2)
