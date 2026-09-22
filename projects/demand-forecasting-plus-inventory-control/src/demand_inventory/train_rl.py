"""Train PPO or SAC replenishment policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from .environment import InventoryControlEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit("Install RL dependencies with: pip install -e '.[rl,dev]'") from exc

    env = Monitor(InventoryControlEnv())
    model_cls = PPO if args.algo == "ppo" else SAC
    kwargs = {"seed": args.seed, "verbose": 1}
    if args.algo == "sac":
        kwargs.update(
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
        )
    model = model_cls("MlpPolicy", env, **kwargs)
    model.learn(total_timesteps=args.timesteps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / f"{args.algo}_inventory_control")


if __name__ == "__main__":
    main()
