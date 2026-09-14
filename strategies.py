"""Trading strategies.

Every strategy implements `generate_signals(df)` and returns a DataFrame
indexed by `df.index` with a `signal` column.

`signal` is the *target weight* decided at the close of bar `t` and applied
from the next bar onward (T+1). The engine is responsible for shifting the
signal and for modelling transaction costs, so strategies never need to worry
about execution.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from features import compute_features


class Strategy(ABC):
    """Base class for all strategies."""

    def __init__(self, allow_short: bool = True):
        self.allow_short = allow_short

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with a `signal` column (-1/0/+1 or continuous)."""

    def _to_signal(self, long_mask: pd.Series) -> pd.Series:
        """Map a boolean 'long' mask to a signal respecting the short policy."""
        long_mask = long_mask.astype(float)
        if self.allow_short:
            values = np.where(long_mask > 0.5, 1.0, -1.0)
        else:
            values = np.where(long_mask > 0.5, 1.0, 0.0)
        return pd.Series(values, index=long_mask.index)

    @staticmethod
    def _empty_signals(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=df.index, columns=["signal"], dtype=float)


class BuyAndHoldStrategy(Strategy):
    """Always fully long. Serves as a benchmark."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        sig = self._empty_signals(df)
        sig["signal"] = 1.0
        return sig


class SmaCrossStrategy(Strategy):
    """Long when the short MA is above the long MA, otherwise short/flat."""

    def __init__(self, short_win=20, long_win=50, allow_short=True):
        super().__init__(allow_short)
        self.short_win = short_win
        self.long_win = long_win

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        sig = self._empty_signals(df)
        short_ma = df["Close"].rolling(self.short_win).mean()
        long_ma = df["Close"].rolling(self.long_win).mean()
        sig["signal"] = self._to_signal(short_ma > long_ma)
        sig["signal"] = sig["signal"].fillna(0.0)
        return sig


class MacdStrategy(Strategy):
    """Long when the MACD line is above the signal line."""

    def __init__(self, fast=12, slow=26, signal=9, allow_short=True):
        super().__init__(allow_short)
        self.fast, self.slow, self.signal = fast, slow, signal

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        from features import macd

        sig = self._empty_signals(df)
        line, signal_line, _ = macd(df["Close"], self.fast, self.slow, self.signal)
        sig["signal"] = self._to_signal(line > signal_line)
        sig["signal"] = sig["signal"].fillna(0.0)
        return sig


class MomentumStrategy(Strategy):
    """Long when trailing N-day return is positive, else short/flat."""

    def __init__(self, lookback=60, allow_short=True):
        super().__init__(allow_short)
        self.lookback = lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        sig = self._empty_signals(df)
        mom = df["Close"].pct_change(self.lookback)
        sig["signal"] = self._to_signal(mom > 0)
        sig["signal"] = sig["signal"].fillna(0.0)
        return sig


class BollingerMeanReversionStrategy(Strategy):
    """Buy when price drops below the lower band, sell when above the upper."""

    def __init__(self, window=20, num_std=2.0, allow_short=True):
        super().__init__(allow_short)
        self.window, self.num_std = window, num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        from features import bollinger

        sig = self._empty_signals(df)
        mid, upper, lower = bollinger(df["Close"], self.window, self.num_std)
        long_mask = df["Close"] < lower
        short_mask = df["Close"] > upper

        base = self._to_signal(long_mask)  # long vs short/flat
        neutral = pd.Series(0.0, index=df.index)
        signal = base.where(long_mask | short_mask, neutral)
        sig["signal"] = signal.fillna(0.0)
        return sig


class RsiMeanReversionStrategy(Strategy):
    """Long when RSI is oversold, short/flat when overbought."""

    def __init__(self, window=14, oversold=30, overbought=70, allow_short=True):
        super().__init__(allow_short)
        self.window, self.oversold, self.overbought = window, oversold, overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        from features import rsi

        sig = self._empty_signals(df)
        r = rsi(df["Close"], self.window)
        long_mask = r < self.oversold
        short_mask = r > self.overbought
        base = self._to_signal(long_mask)
        neutral = pd.Series(0.0, index=df.index)
        signal = base.where(long_mask | short_mask, neutral)
        sig["signal"] = signal.fillna(0.0)
        return sig


class LightGBMStrategy(Strategy):
    """Predict next-day return direction with LightGBM (supervised baseline).

    Trains on bars up to `train_end` (inclusive) and emits predictions for the
    remaining bars. Features use only past information, so there is no lookahead.
    """

    def __init__(self, train_end, allow_short=True, threshold=0.5, params=None, seed=42):
        super().__init__(allow_short)
        self.train_end = pd.Timestamp(train_end)
        self.threshold = threshold
        self.seed = seed
        self.params = params or {}
        self.feature_names = None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        import lightgbm as lgb

        sig = self._empty_signals(df)
        feats = compute_features(df)

        next_ret = df["Close"].pct_change().shift(-1)
        y = (next_ret > 0).astype(int)

        train_mask = feats.index <= self.train_end
        valid = feats.notna().all(axis=1)

        X_train = feats.loc[train_mask & valid]
        y_train = y.loc[X_train.index]
        X_test = feats.loc[~train_mask & valid]

        if X_train.empty or X_test.empty:
            sig["signal"] = 0.0
            return sig

        default_params = {
            "objective": "binary",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbosity": -1,
            "random_state": self.seed,
        }
        default_params.update(self.params)
        model = lgb.LGBMClassifier(**default_params)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        self.feature_names = list(feats.columns)

        predicted_long = pd.Series(proba > self.threshold, index=X_test.index)
        sig.loc[X_test.index, "signal"] = self._to_signal(predicted_long)
        sig["signal"] = sig["signal"].fillna(0.0)
        return sig
