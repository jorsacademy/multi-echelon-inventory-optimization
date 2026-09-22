import unittest
import numpy as np

from conformal_inventory import (
    ClassicalSigmaBox,
    DemandPredictor,
    InventoryParameters,
    MarginalResidualBox,
    SimultaneousConformalBox,
    box_vertices,
    finite_sample_quantile,
    generate_dataset,
    residual_scale_from_training,
    solve_box_robust_plan,
    solve_deterministic_plan,
    solve_saa_plan,
    trajectory_cost,
)


class ConformalInventoryTests(unittest.TestCase):
    def test_dataset_reproducible_and_nonnegative(self):
        a = generate_dataset(30, seed=10, horizon=5)
        b = generate_dataset(30, seed=10, horizon=5)
        np.testing.assert_array_equal(a.features, b.features)
        np.testing.assert_array_equal(a.demand, b.demand)
        self.assertTrue(np.all(a.demand >= 0))

    def test_finite_sample_quantile_hand_check(self):
        values = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
        self.assertEqual(finite_sample_quantile(values, 0.2), 8.0)

    def test_training_scale_is_positive(self):
        y = np.array([[1., 4.], [3., 8.], [5., 6.]])
        p = np.array([[2., 4.], [3., 6.], [4., 9.]])
        scale = residual_scale_from_training(y, p)
        self.assertEqual(scale.shape, (2,))
        self.assertTrue(np.all(scale >= 1.0))

    def test_simultaneous_conformal_box_contains_point_prediction(self):
        y_train = np.array([[10., 20.], [11., 18.], [9., 21.]])
        p_train = np.array([[10., 19.], [10., 19.], [10., 19.]])
        scale = residual_scale_from_training(y_train, p_train)
        y_cal = np.array([[11., 22.], [8., 20.], [10., 17.], [12., 19.]])
        p_cal = np.full_like(y_cal, [10., 19.])
        box = SimultaneousConformalBox.calibrate(y_cal, p_cal, alpha=0.25, scale=scale)
        pred = np.array([[10., 19.], [5., 7.]])
        lo, hi = box.bounds(pred)
        self.assertTrue(np.all(lo >= 0))
        self.assertTrue(np.all(lo <= pred + 1e-12))
        self.assertTrue(np.all(hi >= pred - 1e-12))

    def test_box_vertices_complete(self):
        lo = np.array([1., 3., 5.])
        hi = np.array([2., 4., 8.])
        vertices = box_vertices(lo, hi)
        self.assertEqual(vertices.shape, (8, 3))
        self.assertEqual(len({tuple(x) for x in vertices}), 8)

    def test_deterministic_plan_respects_order_capacity(self):
        params = InventoryParameters.default(3)
        plan = solve_deterministic_plan(np.array([40., 55., 60.]), params)
        self.assertEqual(plan.status, "OPTIMAL")
        self.assertTrue(np.all(plan.orders >= -1e-9))
        self.assertTrue(np.all(plan.orders <= params.order_capacity + 1e-8))

    def test_robust_lp_matches_independent_dense_grid_on_one_period(self):
        params = InventoryParameters(
            order_cost=np.array([2.0]),
            holding_cost=np.array([1.0]),
            shortage_cost=np.array([7.0]),
            order_capacity=np.array([20.0]),
            initial_inventory=1.0,
        )
        lo = np.array([4.0])
        hi = np.array([10.0])
        plan = solve_box_robust_plan(lo, hi, params)

        grid = np.linspace(0.0, 20.0, 20001)
        vals = []
        for q in grid:
            inv_cost = max(
                trajectory_cost(np.array([q]), np.array([lo[0]]), params)[4],
                trajectory_cost(np.array([q]), np.array([hi[0]]), params)[4],
            )
            vals.append(2.0*q + inv_cost)
        grid_best = float(np.min(vals))
        self.assertAlmostEqual(plan.objective, grid_best, delta=2e-3)

    def test_robust_plan_epigraph_matches_all_vertex_costs(self):
        params = InventoryParameters.default(3)
        lo = np.array([25., 35., 30.])
        hi = np.array([45., 52., 48.])
        plan = solve_box_robust_plan(lo, hi, params)
        vertices = box_vertices(lo, hi)
        inventory_costs = [trajectory_cost(plan.orders, d, params)[4] for d in vertices]
        self.assertAlmostEqual(plan.worst_case_inventory_cost, max(inventory_costs), places=6)
        full_worst = np.dot(params.order_cost, plan.orders) + max(inventory_costs)
        self.assertAlmostEqual(plan.objective, full_worst, places=6)

    def test_fill_rate_and_stockout_metrics_are_separate_from_cost(self):
        params = InventoryParameters(
            order_cost=np.array([1.0, 1.0]),
            holding_cost=np.array([1.0, 1.0]),
            shortage_cost=np.array([5.0, 5.0]),
            order_capacity=np.array([10.0, 10.0]),
            initial_inventory=0.0,
        )
        cost, fill, stockout, final_inv, _ = trajectory_cost(
            np.array([5.0, 5.0]),
            np.array([8.0, 2.0]),
            params,
        )
        self.assertGreater(cost, 0)
        self.assertAlmostEqual(fill, 0.7, places=8)
        self.assertAlmostEqual(stockout, 0.5, places=8)
        self.assertAlmostEqual(final_inv, 0.0, places=8)

    def test_saa_plan_is_optimal_against_a_fixed_alternative_on_same_scenarios(self):
        params = InventoryParameters.default(3)
        scenarios = np.array([
            [30., 42., 35.],
            [38., 36., 48.],
            [44., 50., 40.],
            [35., 46., 52.],
        ])
        saa = solve_saa_plan(scenarios, params)
        alternative = solve_deterministic_plan(scenarios.mean(axis=0), params)
        saa_cost = np.mean([trajectory_cost(saa.orders, d, params)[0] for d in scenarios])
        alt_cost = np.mean([trajectory_cost(alternative.orders, d, params)[0] for d in scenarios])
        self.assertAlmostEqual(saa.objective, saa_cost, places=6)
        self.assertLessEqual(saa_cost, alt_cost + 1e-7)

    def test_short_end_to_end_prediction_calibration_and_planning(self):
        train = generate_dataset(80, seed=80, horizon=4)
        cal = generate_dataset(30, seed=81, horizon=4)
        test = generate_dataset(8, seed=82, horizon=4)
        predictor = DemandPredictor.fit(train.features, train.demand, seed=3, n_estimators=50)
        pred_train = predictor.predict(train.features)
        pred_cal = predictor.predict(cal.features)
        pred_test = predictor.predict(test.features)
        scale = residual_scale_from_training(train.demand, pred_train)
        conformal = SimultaneousConformalBox.calibrate(cal.demand, pred_cal, alpha=0.10, scale=scale)
        marginal = MarginalResidualBox.calibrate(cal.demand, pred_cal, alpha=0.10)
        sigma = ClassicalSigmaBox.fit(train.demand, pred_train, sigma_multiplier=2.0)
        params = InventoryParameters.default(4)
        for builder in (conformal.bounds, marginal.bounds, sigma.bounds):
            lo, hi = builder(pred_test[0])
            plan = solve_box_robust_plan(lo, hi, params)
            cost, fill, stockout, _, _ = trajectory_cost(plan.orders, test.demand[0], params)
            self.assertTrue(np.isfinite(cost))
            self.assertTrue(0.0 <= fill <= 1.0)
            self.assertTrue(0.0 <= stockout <= 1.0)


if __name__ == "__main__":
    unittest.main()
