from multi_echelon_inventory import (
    InventoryNode,
    MultiEchelonSystem,
    PolicyType,
    compare_policies,
    optimize_service_level,
    sensitivity_analysis,
)
from multi_echelon_inventory.policies import (
    economic_order_quantity,
    r_q_parameters,
    s_s_parameters,
)


def make_system() -> MultiEchelonSystem:
    return MultiEchelonSystem(
        [
            InventoryNode(
                "Supplier",
                100,
                10,
                2,
                2,
                0.5,
                8.0,
                ordering_cost=5.0,
                service_level=0.95,
                lead_time_std=0.5,
            ),
            InventoryNode(
                "Retailer",
                80,
                10,
                2,
                1,
                1.0,
                10.0,
                ordering_cost=4.0,
                service_level=0.95,
                lead_time_std=0.25,
            ),
        ]
    )


def test_ss_and_rq_parameters_are_well_formed() -> None:
    node = make_system().nodes[-1]
    s, S = s_s_parameters(node)
    R, Q = r_q_parameters(node)
    assert S > s >= 0
    assert R >= 0
    assert Q == economic_order_quantity(node)
    assert Q > 0


def test_monte_carlo_summary_contains_confidence_intervals() -> None:
    result = compare_policies(
        make_system(),
        runs=4,
        periods=20,
        base_seed=11,
        policies=[PolicyType.BASE_STOCK, PolicyType.S_S, PolicyType.R_Q],
    )
    assert len(result.policy_summary) == 3
    assert "Mean Total Cost CI Low" in result.policy_summary.columns
    assert "Mean Total Cost CI High" in result.policy_summary.columns
    assert (
        result.policy_summary["Mean Total Cost CI Low"]
        <= result.policy_summary["Mean Total Cost"]
    ).all()
    assert (
        result.policy_summary["Mean Total Cost"]
        <= result.policy_summary["Mean Total Cost CI High"]
    ).all()


def test_service_level_optimization_returns_feasible_candidate() -> None:
    result = optimize_service_level(
        make_system(),
        service_levels=[0.90, 0.95],
        runs=3,
        periods=25,
        base_seed=5,
        policies=[PolicyType.BASE_STOCK, PolicyType.S_S],
        min_customer_fill_rate=0.0,
    )
    assert result.feasible
    assert result.best_policy in {PolicyType.BASE_STOCK, PolicyType.S_S}
    assert result.best_service_level in {0.90, 0.95}
    assert len(result.candidates) == 4
    assert result.build_system().nodes[-1].service_level == result.best_service_level


def test_sensitivity_analysis_returns_all_scenarios() -> None:
    result = sensitivity_analysis(
        make_system(),
        parameter="demand_std",
        multipliers=[0.8, 1.0, 1.2],
        runs=2,
        periods=15,
        base_seed=7,
        policies=[PolicyType.BASE_STOCK, PolicyType.R_Q],
    )
    assert len(result.scenarios) == 3
    assert len(result.policy_results) == 6
    assert set(result.scenarios["Multiplier"]) == {0.8, 1.0, 1.2}
