import yfinance as yf
import pandas as pd
import numpy as np

def load_data(ticker="QQQ", start="2020-01-01", end="2025-12-31"):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df['Returns'] = df['Close'].pct_change()
    return df.dropna()

#print(load_data())