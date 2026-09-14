import yfinance as yf
import pandas as pd
import numpy as np


def load_data(ticker="QQQ", start="2020-01-01", end="2025-12-31"):
    """Download OHLCV data and return a DataFrame with single-level columns."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        multi_level_index=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker {ticker!r}")

    # yfinance may still return a MultiIndex; normalize to single-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]]
    df["Returns"] = df["Close"].pct_change()
    return df.dropna()
