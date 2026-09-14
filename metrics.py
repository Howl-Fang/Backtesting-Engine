import numpy as np
import pandas as pd

def calculate_performance_metrics(portfolio: pd.DataFrame, risk_free_rate=0.02, traded_days_per_year=252):
    returns = portfolio['strategy_returns']
    equity = portfolio['equity_curve']
    
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    n_days = len(equity)
    annualized_return = (1 + total_return) ** (traded_days_per_year / n_days) - 1
    
    annualized_volatility = returns.std() * np.sqrt(traded_days_per_year)
    
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
    
    cum_max = equity.cummax()
    drawdown = (cum_max - equity) / cum_max
    max_drawdown = drawdown.max()
    
    return {
        "Annualized Return": f"{annualized_return:.2%}",
        "Annualized Volatility": f"{annualized_volatility:.2%}",
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Max Drawdown": f"{max_drawdown:.2%}"
    }