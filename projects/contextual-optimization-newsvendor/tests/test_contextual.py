import numpy as np
import torch

from contextual_opt.cost import critical_fractile, newsvendor_cost_numpy
from contextual_opt.data import generate_contextual_demand
from contextual_opt.policies import DirectDecisionPolicy, knn_saa, train_direct


def test_critical_fractile() -> None:
    assert np.isclose(critical_fractile(5.0, 1.0), 5.0 / 6.0)


def test_knn_saa_outputs_one_nonnegative_decision_per_context() -> None:
    train_x = np.array([[0.0], [1.0], [2.0]], dtype=float)
    train_y = np.array([5.0, 10.0, 15.0])
    test_x = np.array([[0.2], [1.8]])
    quantity = knn_saa(train_x, train_y, test_x, k=2)

    assert quantity.shape == (2,)
    assert np.all(quantity >= 0)
    assert np.isfinite(newsvendor_cost_numpy(quantity, np.array([6.0, 14.0]))).all()


def test_direct_policy_training_smoke() -> None:
    features, demand = generate_contextual_demand(80, seed=5)
    model = DirectDecisionPolicy(features.shape[1], hidden=8)
    train_direct(
        model,
        torch.tensor(features),
        torch.tensor(demand),
        epochs=3,
        learning_rate=1e-2,
    )
    with torch.no_grad():
        quantity = model(torch.tensor(features[:10]))
    assert quantity.shape == (10,)
    assert torch.isfinite(quantity).all()
    assert torch.all(quantity >= 0)
