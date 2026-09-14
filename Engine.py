"""Event-driven (bar-by-bar) backtesting engine with realistic trading costs.

The engine consumes a DataFrame of OHLCV data and a `signal` series (target
weight decided at the close of bar `t`, applied from bar `t+1`), and simulates
cash, share positions, commissions, slippage, stamp duty and short-borrow fees.

Assumptions / conventions
-------------------------
* The signal at bar `t` is rebalanced at the close of bar `t`; the resulting
  position earns the return of bar `t+1` onward (T+1 execution, no lookahead).
* `commission` and `slippage` apply to both sides of a trade.
* `stamp_tax` applies to the sell side only (A-share style).
* `min_commission` is a per-trade floor (A-share ¥5 floor).
* `short_borrow_fee` is an annualised fee charged daily on short exposure.
"""

import numpy as np
import pandas as pd


class BacktesterEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,
        init_cap: float = 1e5,
        commission: float = 2.5e-4,
        min_commission: float = 5.0,
        slippage: float = 5e-4,
        stamp_tax: float = 5e-4,
        short_borrow_fee: float = 0.02,
        allow_short: bool = True,
    ):
        self.df = df
        self.signals = signals
        self.init_cap = init_cap
        self.commission = commission
        self.min_commission = min_commission
        self.slippage = slippage
        self.stamp_tax = stamp_tax
        self.short_borrow_fee = short_borrow_fee
        self.allow_short = allow_short

    def run_backtest(self) -> pd.DataFrame:
        prices = self.df["Close"].astype(float)
        target = self.signals["signal"].reindex(self.df.index).astype(float)
        if not self.allow_short:
            target = target.clip(lower=0.0)

        n = len(prices)
        cash = float(self.init_cap)
        shares = 0.0
        prev_signal = 0.0

        equity = np.zeros(n)
        position_weight = np.zeros(n)
        turnover = np.zeros(n)

        for i in range(n):
            price = prices.iloc[i]

            equity_pre = cash + shares * price
            position_weight[i] = shares * price / equity_pre if equity_pre != 0 else 0.0

            sig = target.iloc[i]
            delta = 0.0
            if abs(sig - prev_signal) > 1e-9:
                desired_shares = sig * equity_pre / price
                delta = desired_shares - shares

                if abs(delta) > 1e-12:
                    notional = abs(delta) * price
                    commission = max(notional * self.commission, self.min_commission)
                    slippage = notional * self.slippage
                    stamp = abs(delta) * price * self.stamp_tax if delta < 0 else 0.0

                    cash -= delta * price
                    cash -= commission + slippage + stamp
                    shares += delta

                prev_signal = sig

            if shares < 0:
                cash -= abs(shares) * price * (self.short_borrow_fee / 252)

            equity[i] = cash + shares * price
            turnover[i] = abs(delta) * price / equity[i] if equity[i] != 0 else 0.0

        portfolio = pd.DataFrame(
            {
                "equity_curve": equity,
                "position": position_weight,
                "turnover": turnover,
            },
            index=self.df.index,
        )
        portfolio["strategy_returns"] = portfolio["equity_curve"].pct_change().fillna(0.0)
        return portfolio
