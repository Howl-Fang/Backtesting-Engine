import pandas as pd
import numpy as np

def generate_sma_signals(df: pd.DataFrame, short_win=20, long_win=50):
     signals = pd.DataFrame(index=df.index)

     signals['short_mavg'] = df['Close'].rolling(window=short_win, min_periods=1).mean()
     signals['long_mavg'] = df['Close'].rolling(window=long_win, min_periods=1).mean()
     
     # Buy when short>long term, and otherwise.
     signals['raw_signal'] = np.where(signals['short_mavg'] > signals['long_mavg'], 1.0, -1.0)
     
     # applies in T+1
     signals['position'] = signals['raw_signal'].shift(1).fillna(0)
     
     return signals