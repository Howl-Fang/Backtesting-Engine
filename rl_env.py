"""Gymnasium environment that wraps a price DataFrame for reinforcement learning.

The agent observes a window of technical features and outputs a discrete action:
    0 -> short (-1), 1 -> flat (0), 2 -> long (+1)
The chosen position is held during the *next* bar (T+1), so there is no
lookahead. Rewards are daily PnL net of transaction costs, mirroring
`BacktesterEngine`.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from features import compute_features

FEATURE_COLS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_5",
    "vol_20",
    "rsi_14",
    "macd_hist",
    "bb_pos",
]


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df,
        window: int = 20,
        commission: float = 2.5e-4,
        slippage: float = 5e-4,
        stamp_tax: float = 5e-4,
        allow_short: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.df = df
        self.window = window
        self.commission = commission
        self.slippage = slippage
        self.stamp_tax = stamp_tax
        self.allow_short = allow_short

        self.returns = df["Close"].pct_change().fillna(0.0).to_numpy(dtype=np.float32)
        feats = compute_features(df)[FEATURE_COLS]
        self.features = feats.fillna(0.0).to_numpy(dtype=np.float32)
        self.n = len(df)
        self.n_feat = self.features.shape[1]

        obs_dim = window * self.n_feat + 1  # +1 for current position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.reset(seed=seed)

    def _action_to_position(self, action) -> float:
        if self.allow_short:
            return float(action - 1)  # 0->-1, 1->0, 2->+1
        return 1.0 if action == 2 else 0.0

    def _get_obs(self) -> np.ndarray:
        day = min(self._day, self.n - 1)
        start = day - self.window + 1
        if start < 0:
            win = np.zeros((self.window, self.n_feat), dtype=np.float32)
            win[-day - 1 :] = self.features[0 : day + 1]
        else:
            win = self.features[start : day + 1]
        return np.concatenate([win.flatten(), np.array([self.position], dtype=np.float32)])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._day = 0
        self.position = 0.0
        return self._get_obs(), {}

    def step(self, action):
        target = self._action_to_position(int(action))

        turnover = abs(target - self.position)
        sell = max(self.position - target, 0.0)
        cost = turnover * (self.commission + self.slippage) + sell * self.stamp_tax

        self.position = target
        self._day += 1

        if self._day >= self.n:
            return self._get_obs(), 0.0, True, False, {}

        reward = float(target * self.returns[self._day] - cost)
        return self._get_obs(), reward, False, False, {}
