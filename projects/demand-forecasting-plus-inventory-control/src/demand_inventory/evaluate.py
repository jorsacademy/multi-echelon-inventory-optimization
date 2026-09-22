"""Evaluate classical and learned inventory policies on identical seeded episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .baselines import ConstantOrderPolicy, ForecastOrderUpToPolicy, SSInventoryPolicy
from .environment import InventoryConfig, InventoryControlEnv


def run_policy(name: str, policy, episodes: int, seed: int) -> pd.DataFrame:
    rows = []
    for episode in range(episodes):
        env = InventoryControlEnv()
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        final_info = {}
        while not done:
            if hasattr(policy, "predict"):
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = policy.act(obs)
            obs, reward, terminated, truncated, final_info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rows.append(
            {
                "policy": name,
                "episode": episode,
                "return": total_reward,
                "total_cost": final_info["total_cost"],
                "service_level": final_info["service_level"],
                "fill_rate": final_info["fill_rate"],
                "stockout_days": final_info["stockout_days"],
                "final_inventory": final_info["inventory"],
                "final_backlog": final_info["backlog"],
                "total_ordered": final_info["total_ordered"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--ppo-model", type=Path)
    parser.add_argument("--sac-model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/comparison.csv"))
    args = parser.parse_args()

    config = InventoryConfig()
    policies = {
        "constant_order": ConstantOrderPolicy(config),
        "s_S": SSInventoryPolicy(config),
        "forecast_order_up_to": ForecastOrderUpToPolicy(config),
    }

    if args.ppo_model or args.sac_model:
        try:
            from stable_baselines3 import PPO, SAC
        except ImportError as exc:
            raise SystemExit("Install RL dependencies with: pip install -e '.[rl,dev]'") from exc
        if args.ppo_model:
            policies["ppo"] = PPO.load(args.ppo_model)
        if args.sac_model:
            policies["sac"] = SAC.load(args.sac_model)

    results = pd.concat(
        [run_policy(name, policy, args.episodes, args.seed) for name, policy in policies.items()],
        ignore_index=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    summary = results.groupby("policy").agg(
        mean_total_cost=("total_cost", "mean"),
        mean_service_level=("service_level", "mean"),
        mean_stockout_days=("stockout_days", "mean"),
        mean_final_inventory=("final_inventory", "mean"),
        mean_final_backlog=("final_backlog", "mean"),
    )
    print(summary.round(3))


if __name__ == "__main__":
    main()
