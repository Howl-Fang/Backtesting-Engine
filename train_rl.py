"""Train a PPO trading agent and backtest the learned policy out-of-sample.

Example:
    python train_rl.py --timesteps 500000 --train-end 2019-12-31 \
        --action-type continuous --reward-mode differential_sharpe --turnover-penalty 1e-3
"""

import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from data_loader import load_data
from rl_env import TradingEnv, FEATURE_COLS
from rl_policy import AttentionExtractor
from Engine import BacktesterEngine
from metrics import calculate_performance_metrics, print_metrics


def build_env(df, args):
    return TradingEnv(
        df,
        window=args.window,
        allow_short=args.allow_short,
        action_type=args.action_type,
        reward_mode=args.reward_mode,
        turnover_penalty=args.turnover_penalty,
        dsr_eta=args.dsr_eta,
        seed=args.seed,
    )


def policy_to_signals(model, env):
    """Run a trained policy over `env` and return a `signal` DataFrame."""
    obs, _ = env.reset()
    positions = np.empty(env.n, dtype=float)
    for i in range(env.n):
        action, _ = model.predict(obs, deterministic=True)
        positions[i] = env._action_to_position(action)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    return pd.DataFrame({"signal": positions}, index=env.df.index)


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
    parser.add_argument("--action-type", choices=["discrete", "continuous"], default="continuous")
    parser.add_argument("--reward-mode", choices=["return", "log_return", "differential_sharpe"], default="return")
    parser.add_argument("--turnover-penalty", type=float, default=0.0)
    parser.add_argument("--dsr-eta", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--features-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu", help="torch device: cpu, cuda, auto")
    args = parser.parse_args()

    df = load_data(args.ticker, args.start, args.end)
    train_df = df.loc[: args.train_end]
    test_df = df.loc[args.train_end :]

    env = build_env(train_df, args)

    policy_kwargs = dict(
        features_extractor_class=AttentionExtractor,
        features_extractor_kwargs=dict(
            window=args.window,
            n_feat=len(FEATURE_COLS),
            d_model=args.d_model,
            nhead=args.nhead,
            features_dim=args.features_dim,
        ),
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        seed=args.seed,
        verbose=1,
        device=args.device,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save("ppo_trading")

    test_env = build_env(test_df, args)
    signals = policy_to_signals(model, test_env)

    engine = BacktesterEngine(test_df, signals, allow_short=args.allow_short)
    portfolio = engine.run_backtest()

    bench = test_df["Close"].pct_change().fillna(0.0)
    metrics = calculate_performance_metrics(portfolio, benchmark_returns=bench)
    print_metrics(metrics, "PPO Out-of-Sample")


if __name__ == "__main__":
    main()
