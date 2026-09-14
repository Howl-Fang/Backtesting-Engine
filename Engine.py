import numpy as np
import pandas as pd


class BacktesterEngine:
     def __init__(self, df: pd.DataFrame, signals: pd.DataFrame, init_cap = 1e5, commission = 1e-3, slippage = 5e-4):
          self.df = df
          self.signals = signals
          self.init_cap = init_cap
          self.commission = commission
          self.slippage = slippage
          # print(self.signals['position'].shape, self.signals['position'].__class__)
     
     def run_backtest(self):
          portfolio = pd.DataFrame(index=self.df.index)
          
          asset_returns = self.df['Close'].pct_change().fillna(0)
          asset_returns = asset_returns.squeeze()
          # print(asset_returns.shape)
          
          position_diff = self.signals['position'].diff().abs().fillna(0)
          trade_costs = position_diff * (self.commission + self.slippage)
          
          # return = position * asset_returns - trade_costs
          # print(((self.signals['position'] * asset_returns) - trade_costs).shape)
          portfolio['strategy_returns'] = (self.signals['position'] * asset_returns) - trade_costs
          
          # Equity Curve
          portfolio['equity_curve'] = self.init_cap * (1 + portfolio['strategy_returns']).cumprod()
          
          return portfolio