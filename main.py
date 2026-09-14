import numpy as np
import pandas as pd

from data_loader import *
from signals import *
from Engine import *
from metrics import *

df = load_data(ticker="000001.SS", start="2000-01-01", end="2026-08-31")

signals = generate_sma_signals(df)

engine = BacktesterEngine(df, signals)

portfolio = engine.run_backtest()

metrics = calculate_performance_metrics(portfolio)

print("Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")