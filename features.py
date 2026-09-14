"""Feature engineering helpers shared by strategies (mainly ML-based ones)."""

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False).mean()
    return line, signal_line, line - signal_line


def bollinger(close: pd.Series, window=20, num_std=2.0):
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame of technical features aligned to `df.index`.

    All features use information up to and including the current bar, so a
    model can safely predict the *next* bar's return without lookahead bias.
    """
    close = df["Close"]
    feats = pd.DataFrame(index=df.index)

    for lag in (1, 2, 3, 5, 10, 20):
        feats[f"ret_{lag}"] = close.pct_change(lag)

    feats["vol_5"] = close.pct_change().rolling(5).std()
    feats["vol_20"] = close.pct_change().rolling(20).std()

    feats["sma_10"] = close.rolling(10).mean() / close - 1
    feats["sma_50"] = close.rolling(50).mean() / close - 1
    feats["sma_200"] = close.rolling(200).mean() / close - 1

    feats["rsi_14"] = rsi(close, 14) / 100.0

    macd_line, macd_signal, macd_hist = macd(close)
    feats["macd_hist"] = macd_hist / close
    feats["macd_line"] = macd_line / close

    mid, upper, lower = bollinger(close)
    width = (upper - lower).replace(0.0, np.nan)
    feats["bb_pos"] = (close - mid) / width
    feats["bb_width"] = width / mid

    feats["vol_change"] = df["Volume"].pct_change(5)

    return feats
