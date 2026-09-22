import numpy as np

from demand_inventory.baselines import ConstantOrderPolicy, ForecastOrderUpToPolicy, SSInventoryPolicy
from demand_inventory.environment import InventoryConfig, InventoryControlEnv
from demand_inventory.evaluate import run_policy


def test_baseline_actions_are_valid():
    config = InventoryConfig(horizon=5)
    env = InventoryControlEnv(config)
    obs, _ = env.reset(seed=2)
    for policy in [ConstantOrderPolicy(config), SSInventoryPolicy(config), ForecastOrderUpToPolicy(config)]:
        action = policy.act(obs)
        assert env.action_space.contains(action)
        assert np.isfinite(action).all()


def test_baseline_evaluation_returns_finite_metrics():
    config = InventoryConfig()
    for name, policy in {
        "constant": ConstantOrderPolicy(config),
        "s_S": SSInventoryPolicy(config),
        "forecast": ForecastOrderUpToPolicy(config),
    }.items():
        frame = run_policy(name, policy, episodes=2, seed=10)
        assert len(frame) == 2
        assert np.isfinite(frame["total_cost"]).all()
        assert ((frame["service_level"] >= 0.0) & (frame["service_level"] <= 1.0 + 1e-9)).all()
