import numpy as np

from demand_inventory.environment import InventoryConfig, InventoryControlEnv


def test_seeded_reset_is_deterministic():
    env = InventoryControlEnv()
    obs1, _ = env.reset(seed=7)
    demand1 = env._demand.copy()
    obs2, _ = env.reset(seed=7)
    demand2 = env._demand.copy()
    np.testing.assert_allclose(obs1, obs2)
    np.testing.assert_allclose(demand1, demand2)


def test_inventory_and_backlog_remain_nonnegative():
    env = InventoryControlEnv(InventoryConfig(horizon=20))
    obs, _ = env.reset(seed=3)
    for _ in range(20):
        obs, reward, _, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert info["inventory"] >= 0.0
        assert info["backlog"] >= 0.0
        if truncated:
            break


def test_zero_order_eventually_creates_stockout_pressure():
    config = InventoryConfig(horizon=30, initial_inventory=10.0)
    env = InventoryControlEnv(config)
    env.reset(seed=11)
    final_info = {}
    for _ in range(config.horizon):
        _, _, _, truncated, final_info = env.step(np.array([0.0], dtype=np.float32))
        if truncated:
            break
    assert final_info["stockout_days"] > 0
    assert final_info["backlog"] > 0.0


def test_order_arrives_after_lead_time():
    config = InventoryConfig(horizon=8, lead_time=2, initial_inventory=0.0, base_demand=0.0, weekly_amplitude=0.0, demand_noise_std=0.0)
    env = InventoryControlEnv(config)
    env.reset(seed=1)
    _, _, _, _, info1 = env.step(np.array([1.0], dtype=np.float32))
    _, _, _, _, info2 = env.step(np.array([0.0], dtype=np.float32))
    _, _, _, _, info3 = env.step(np.array([0.0], dtype=np.float32))
    assert np.isclose(info1["inventory"], 0.0)
    assert np.isclose(info2["inventory"], 0.0)
    assert info3["inventory"] > 0.0
