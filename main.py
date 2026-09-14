import pandas as pd

from data_loader import load_data
from strategies import (
    BuyAndHoldStrategy,
    SmaCrossStrategy,
    MacdStrategy,
    MomentumStrategy,
    BollingerMeanReversionStrategy,
    RsiMeanReversionStrategy,
    LightGBMStrategy,
)
from backtest import compare_strategies, walk_forward
from metrics import print_metrics

TICKER = "000001.SS"
START = "2010-01-01"
END = "2026-08-31"

df = load_data(TICKER, START, END)

strategies = {
    "Buy & Hold": BuyAndHoldStrategy(),
    "SMA 20/50": SmaCrossStrategy(20, 50),
    "MACD 12/26/9": MacdStrategy(),
    "Momentum 60d": MomentumStrategy(60),
    "Bollinger MR": BollingerMeanReversionStrategy(),
    "RSI MR": RsiMeanReversionStrategy(),
}

table = compare_strategies(strategies, df)
print("=== Strategy Comparison (benchmark = Buy & Hold) ===")
print(table.to_string())

portfolio, metrics = walk_forward(
    df,
    strategy_factory=lambda train_end: LightGBMStrategy(train_end),
    n_splits=6,
)
print_metrics(metrics, "LightGBM Walk-Forward (out-of-sample)")
