"""Performance metrics for a backtest portfolio, with optional benchmark."""

import numpy as np
import pandas as pd


def calculate_performance_metrics(
    portfolio: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
) -> dict:
    returns = portfolio["strategy_returns"]
    equity = portfolio["equity_curve"]

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_days = len(equity)
    annualized_return = (1 + total_return) ** (trading_days / n_days) - 1

    annualized_vol = returns.std() * np.sqrt(trading_days)

    sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0.0

    downside = returns[returns < 0].std() * np.sqrt(trading_days)
    sortino = (annualized_return - risk_free_rate) / downside if downside > 0 else 0.0

    cum_max = equity.cummax()
    drawdown = (cum_max - equity) / cum_max
    max_drawdown = drawdown.max()
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    exposed = portfolio["position"] != 0 if "position" in portfolio else returns != 0
    win_rate = (returns[exposed] > 0).mean()

    metrics = {
        "Total Return": f"{total_return:.2%}",
        "Annualized Return": f"{annualized_return:.2%}",
        "Annualized Volatility": f"{annualized_vol:.2%}",
        "Sharpe Ratio": round(sharpe, 3),
        "Sortino Ratio": round(sortino, 3),
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Calmar Ratio": round(calmar, 3),
        "Win Rate (daily)": f"{win_rate:.2%}",
    }

    if "turnover" in portfolio:
        metrics["Avg Turnover (daily)"] = f"{portfolio['turnover'].mean():.2%}"

    if "position" in portfolio:
        metrics["Market Exposure"] = f"{(portfolio['position'] != 0).mean():.2%}"

    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(returns.index).fillna(0.0)
        bench_total = (1 + bench).prod() - 1
        bench_ann = (1 + bench_total) ** (trading_days / n_days) - 1
        bench_vol = bench.std() * np.sqrt(trading_days)
        bench_sharpe = (bench_ann - risk_free_rate) / bench_vol if bench_vol > 0 else 0.0

        excess_s = returns - risk_free_rate / trading_days
        excess_b = bench - risk_free_rate / trading_days
        beta = np.cov(excess_s, excess_b)[0, 1] / np.var(excess_b) if np.var(excess_b) > 0 else 0.0
        alpha = (excess_s.mean() - beta * excess_b.mean()) * trading_days

        metrics["Benchmark Return"] = f"{bench_total:.2%}"
        metrics["Benchmark Sharpe"] = round(bench_sharpe, 3)
        metrics["Alpha (annualized)"] = f"{alpha:.2%}"
        metrics["Beta"] = round(beta, 3)

    return metrics


def print_metrics(metrics: dict, title: str = "Performance Metrics") -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in metrics.items():
        print(f"{k}: {v}")
