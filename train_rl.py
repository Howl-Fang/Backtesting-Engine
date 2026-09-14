"""Train a PPO trading agent and backtest the learned policy out-of-sample.

Example:
    python train_rl.py --timesteps 200000 --train-end 2019-12-31
"""

import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from data_loader import load_data
from rl_env import TradingEnv
from Engine import BacktesterEngine
from metrics import calculate_performance_metrics, print_metrics


def policy_to_signals(model, df, allow_short, window, env_kwargs=None):
    """Run a trained policy over `df` and return a `signal` DataFrame."""
    env_kwargs = env_kwargs or {}
    env = TradingEnv(df, window=window, allow_short=allow_short, **env_kwargs)
    obs, _ = env.reset()
    positions = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        action, _ = model.predict(obs, deterministic=True)
        positions[i] = env._action_to_position(action)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    return pd.DataFrame({"signal": positions}, index=df.index)


def main():
    parser = argparse.ArgumentParser(description="Train PPO trading agent")
    parser.add_argument("--ticker", default="000001.SS")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-08-31")
    parser.add_argument("--train-end", default="2019-12-31", help="last date of training set")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--allow-short", action="store_true", default=True)
    parser.add_argument("--no-short", dest="allow_short", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu", help="torch device: cpu, cuda, auto")
    args = parser.parse_args()

    df = load_data(args.ticker, args.start, args.end)
    train_df = df.loc[: args.train_end]
    test_df = df.loc[args.train_end :]

    env = TradingEnv(train_df, window=args.window, allow_short=args.allow_short, seed=args.seed)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        seed=args.seed,
        verbose=1,
        device=args.device,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save("ppo_trading")

    test_env = TradingEnv(test_df, window=args.window, allow_short=args.allow_short)
    obs, _ = test_env.reset()
    positions = np.empty(len(test_df), dtype=float)
    for i in range(len(test_df)):
        action, _ = model.predict(obs, deterministic=True)
        positions[i] = test_env._action_to_position(action)
        obs, _, terminated, _, _ = test_env.step(action)
        if terminated:
            break
    signals = pd.DataFrame({"signal": positions}, index=test_df.index)

    engine = BacktesterEngine(test_df, signals, allow_short=args.allow_short)
    portfolio = engine.run_backtest()

    bench = test_df["Close"].pct_change().fillna(0.0)
    metrics = calculate_performance_metrics(portfolio, benchmark_returns=bench)
    print_metrics(metrics, "PPO Out-of-Sample")


if __name__ == "__main__":
    main()
