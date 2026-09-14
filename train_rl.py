"""Train a PPO trading agent and backtest the learned policy out-of-sample.

The console shows a live, in-place panel of detailed training metrics (loss,
KL, entropy, explained variance, ...) refreshed a few times per second, plus an
occasional Sharpe-style evaluation. Full logs are written to `--log-dir`
(SB3 metrics in progress.csv / log.txt, evaluations in eval.log).

Example:
    python train_rl.py --timesteps 500000 --train-end 2019-12-31
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from data_loader import load_data
from rl_env import TradingEnv, FEATURE_COLS
from rl_policy import AttentionExtractor
from Engine import BacktesterEngine
from metrics import calculate_performance_metrics, print_metrics

TRADING_DAYS = 252
RISK_FREE = 0.02


class TradingEvalCallback(BaseCallback):
    """Live training panel + periodic out-of-sample evaluation.

    The panel overwrites itself in place (ANSI cursor movement) instead of
    scrolling. Evaluations run every `eval_freq` timesteps and are written to
    `log_path`; the latest evaluation is also shown in the panel.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 40960,
        log_path: str = "logs/eval.log",
        refresh_sec: float = 0.5,
    ):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.refresh_sec = refresh_sec

        self._last_eval = 0
        self._last_render = 0.0
        self._panel_height = 0
        self._last_eval_metrics = None
        self._start_time = None
        self._live = sys.stdout.isatty() or bool(os.environ.get("RL_FORCE_LIVE"))

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self._fh = open(self.log_path, "w")
        self._fh.write(
            "timesteps\treturn\tsharpe\tsortino\tvol\tmax_drawdown\tturnover\texposure\n"
        )
        self._fh.flush()
        self._start_time = time.time()

    def _on_training_end(self):
        self._fh.close()
        if self._live:
            self._render(force=True)
            print()

    def _evaluate(self):
        obs, _ = self.eval_env.reset()
        nets, turnovers, exposures = [], [], []
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _, info = self.eval_env.step(action)
            if "net_return" in info:
                nets.append(info["net_return"])
                turnovers.append(info["turnover"])
                exposures.append(abs(self.eval_env.position) > 1e-9)
            if done:
                break

        nets = np.asarray(nets, dtype=float)
        n = len(nets)
        if n == 0:
            return {"return": 0.0, "sharpe": 0.0, "sortino": 0.0, "vol": 0.0,
                    "max_drawdown": 0.0, "turnover": 0.0, "exposure": 0.0}

        equity = np.cumprod(1.0 + nets)
        total_return = equity[-1] - 1.0
        ann_return = (1.0 + total_return) ** (TRADING_DAYS / n) - 1.0
        vol = nets.std() * np.sqrt(TRADING_DAYS)
        sharpe = (ann_return - RISK_FREE) / vol if vol > 0 else 0.0
        downside = nets[nets < 0].std() * np.sqrt(TRADING_DAYS)
        sortino = (ann_return - RISK_FREE) / downside if downside > 0 else 0.0
        cummax = np.maximum.accumulate(equity)
        max_drawdown = float(((cummax - equity) / cummax).max())

        return {
            "return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "vol": vol,
            "max_drawdown": max_drawdown,
            "turnover": float(np.mean(turnovers)),
            "exposure": float(np.mean(exposures)),
        }

    @staticmethod
    def _box(rows) -> list:
        key_w, val_w = 30, 9
        border = "-" * (key_w + val_w + 7)
        lines = [border]
        for key, value in rows:
            value = "" if value is None else value
            lines.append(f"| {key:<{key_w}} | {value:>{val_w}} |")
        lines.append(border)
        return lines

    def _panel_lines(self) -> list:
        v = self.model.logger.name_to_value

        def fmt(key, spec, default="n/a"):
            value = v.get(key)
            return format(value, spec) if value is not None else default

        elapsed = max(time.time() - self._start_time, 1e-6)
        fps = int(self.num_timesteps / elapsed)
        iteration = self.num_timesteps // max(self.model.n_steps, 1)

        ep_rew, ep_len = "n/a", "n/a"
        buf = self.model.ep_info_buffer
        if buf:
            ep_rew = f"{np.mean([ep['r'] for ep in buf]):.4f}"
            ep_len = f"{np.mean([ep['l'] for ep in buf]):.1f}"

        rows = [
            ("time/", None),
            ("   fps", str(fps)),
            ("   iterations", str(iteration)),
            ("   time_elapsed", str(int(elapsed))),
            ("   total_timesteps", str(self.num_timesteps)),
            ("rollout/", None),
            ("   ep_rew_mean", ep_rew),
            ("   ep_len_mean", ep_len),
            ("train/", None),
            ("   approx_kl", fmt("train/approx_kl", ".4f")),
            ("   clip_fraction", fmt("train/clip_fraction", ".3f")),
            ("   entropy_loss", fmt("train/entropy_loss", ".4f")),
            ("   explained_variance", fmt("train/explained_variance", ".3f")),
            ("   learning_rate", fmt("train/learning_rate", ".4f")),
            ("   loss", fmt("train/loss", ".4f")),
            ("   n_updates", fmt("train/n_updates", ".0f")),
            ("   policy_gradient_loss", fmt("train/policy_gradient_loss", ".4f")),
            ("   value_loss", fmt("train/value_loss", ".4f")),
            ("eval/", None),
        ]

        m = self._last_eval_metrics
        if m is None:
            for label in ("return", "sharpe", "sortino", "max_drawdown", "turnover"):
                rows.append((f"   {label}", "n/a"))
        else:
            rows.append(("   return", f"{m['return']:.2%}"))
            rows.append(("   sharpe", f"{m['sharpe']:.2f}"))
            rows.append(("   sortino", f"{m['sortino']:.2f}"))
            rows.append(("   max_drawdown", f"{m['max_drawdown']:.2%}"))
            rows.append(("   turnover", f"{m['turnover']:.2%}"))
        return self._box(rows)

    def _render(self, force: bool = False):
        if not self._live:
            return
        now = time.time()
        if not force and now - self._last_render < self.refresh_sec:
            return
        self._last_render = now

        lines = self._panel_lines()
        if self._panel_height:
            sys.stdout.write(f"\033[{self._panel_height}A")
        for line in lines:
            sys.stdout.write("\033[K" + line + "\n")
        self._panel_height = len(lines)
        sys.stdout.flush()

    def _on_step(self):
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._last_eval = self.num_timesteps
            m = self._evaluate()
            self._last_eval_metrics = m
            self._fh.write(
                f"{self.num_timesteps}\t{m['return']:.4f}\t{m['sharpe']:.3f}\t"
                f"{m['sortino']:.3f}\t{m['vol']:.4f}\t{m['max_drawdown']:.4f}\t"
                f"{m['turnover']:.4f}\t{m['exposure']:.4f}\n"
            )
            self._fh.flush()
            if not self._live:
                print(
                    f"[eval @ {self.num_timesteps}] ret {m['return']:.2%} | "
                    f"sharpe {m['sharpe']:.2f} | sortino {m['sortino']:.2f} | "
                    f"vol {m['vol']:.2%} | maxDD {m['max_drawdown']:.2%} | "
                    f"turn {m['turnover']:.2%} | exp {m['exposure']:.2%}"
                )
        self._render()
        return True


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
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--allow-short", action="store_true", default=True)
    parser.add_argument("--no-short", dest="allow_short", action="store_false")
    parser.add_argument("--action-type", choices=["discrete", "continuous"], default="continuous")
    parser.add_argument("--reward-mode", choices=["return", "log_return", "differential_sharpe"], default="differential_sharpe")
    parser.add_argument("--turnover-penalty", type=float, default=1e-3)
    parser.add_argument("--dsr-eta", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--features-dim", type=int, default=128)
    parser.add_argument("--eval-freq", type=int, default=40960, help="evaluate every N timesteps")
    parser.add_argument("--log-dir", default="logs")
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
        verbose=0,
        device=args.device,
        policy_kwargs=policy_kwargs,
    )

    # Full training metrics (loss/kl/entropy/...) are saved to --log-dir.
    model.set_logger(configure(args.log_dir, ["csv", "log"]))

    eval_env = build_env(test_df, args)
    callback = TradingEvalCallback(
        eval_env, eval_freq=args.eval_freq, log_path=os.path.join(args.log_dir, "eval.log")
    )

    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
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
