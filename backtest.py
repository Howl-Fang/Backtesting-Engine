"""High-level helpers to run strategies, compare them and do walk-forward tests."""

import numpy as np
import pandas as pd

from Engine import BacktesterEngine
from metrics import calculate_performance_metrics


def run_strategy(
    strategy,
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    **engine_kwargs,
):
    """Run a strategy and return (portfolio, metrics)."""
    signals = strategy.generate_signals(df)
    engine = BacktesterEngine(df, signals, **engine_kwargs)
    portfolio = engine.run_backtest()

    bench_returns = None
    if benchmark_df is not None:
        bench_returns = benchmark_df["Close"].pct_change().reindex(df.index).fillna(0.0)
    else:
        bench_returns = df["Close"].pct_change().fillna(0.0)

    metrics = calculate_performance_metrics(portfolio, benchmark_returns=bench_returns)
    return portfolio, metrics


def compare_strategies(
    strategies: dict,
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    **engine_kwargs,
) -> pd.DataFrame:
    """Run several strategies and return a summary table of key metrics."""
    rows = []
    for name, strategy in strategies.items():
        _, metrics = run_strategy(strategy, df, benchmark_df, **engine_kwargs)
        rows.append({"Strategy": name, **metrics})
    return pd.DataFrame(rows).set_index("Strategy")


def walk_forward(
    df: pd.DataFrame,
    strategy_factory,
    n_splits: int = 5,
    train_frac: float = 0.6,
    **engine_kwargs,
):
    """Rolling walk-forward backtest.

    `strategy_factory(train_end)` returns a fresh strategy fitted on data up to
    `train_end`. Each fold trains on a fixed-size trailing window and predicts
    only the immediately following out-of-sample segment, then the segments are
    stitched together and run through the engine once.
    """
    n = len(df)
    split_points = np.linspace(int(n * train_frac), n, n_splits + 1).astype(int)

    signals = pd.DataFrame(index=df.index, columns=["signal"], dtype=float)
    for k in range(n_splits):
        train_end = df.index[split_points[k] - 1]
        test_start, test_end = split_points[k], split_points[k + 1]

        strategy = strategy_factory(train_end)
        all_signal = strategy.generate_signals(df)
        segment = all_signal.iloc[test_start:test_end]
        signals.loc[segment.index, "signal"] = segment["signal"]

    engine = BacktesterEngine(df, signals, **engine_kwargs)
    portfolio = engine.run_backtest()
    bench_returns = df["Close"].pct_change().fillna(0.0)
    metrics = calculate_performance_metrics(portfolio, benchmark_returns=bench_returns)
    return portfolio, metrics
