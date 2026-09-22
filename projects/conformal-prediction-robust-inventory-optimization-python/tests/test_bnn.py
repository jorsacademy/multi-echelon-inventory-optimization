import importlib.util
import unittest

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch optional dependency is not installed")
class BayesianDemandPredictorTests(unittest.TestCase):
    def test_analytic_kl_matches_torch_distribution(self):
        import torch
        from torch.distributions import Normal, kl_divergence

        from conformal_inventory.bnn import BayesianLinear

        layer = BayesianLinear(2, 3, prior_sigma=0.7).double()
        expected = sum(
            kl_divergence(
                Normal(mu, layer.scale(rho)),
                Normal(torch.zeros_like(mu), layer.prior_sigma),
            ).sum()
            for mu, rho in (
                (layer.weight_mu, layer.weight_rho),
                (layer.bias_mu, layer.bias_rho),
            )
        )
        torch.testing.assert_close(layer.kl_divergence(), expected)

    def test_elbo_averages_log_likelihood_over_weight_draws(self):
        import torch

        from conformal_inventory.bnn import negative_elbo

        draws = torch.tensor([[[-1.0]], [[1.0]]])
        target = torch.zeros(1, 1)
        actual = negative_elbo(
            draws,
            target,
            torch.tensor(4.0),
            dataset_size=2,
            noise_std=1.0,
        )
        expected = 0.5 * (1.0 + torch.log(torch.tensor(2.0 * torch.pi))) + 2.0
        torch.testing.assert_close(actual, expected)

    def test_small_bnn_fit_predicts_nonnegative_multi_period_demand(self):
        from conformal_inventory import generate_dataset
        from conformal_inventory.bnn import BayesianDemandPredictor

        train = generate_dataset(80, seed=123, horizon=4)
        test = generate_dataset(10, seed=124, horizon=4)

        predictor = BayesianDemandPredictor.fit(
            train.features,
            train.demand,
            seed=7,
            hidden_dims=(12,),
            steps=80,
            train_samples=2,
            prediction_samples=30,
        )
        pred = predictor.predict(test.features)
        pred_repeat = predictor.predict(test.features)
        epistemic = predictor.epistemic_std(test.features, num_samples=20)

        np.testing.assert_allclose(pred, pred_repeat)
        self.assertEqual(pred.shape, test.demand.shape)
        self.assertEqual(epistemic.shape, test.demand.shape)
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue(np.isfinite(epistemic).all())
        self.assertTrue(np.all(pred >= 0.0))
        self.assertTrue(np.all(epistemic >= 0.0))


if __name__ == "__main__":
    unittest.main()
