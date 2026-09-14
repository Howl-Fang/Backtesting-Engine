"""Gymnasium environment that wraps a price DataFrame for reinforcement learning.

The agent observes a window of technical features plus its current position and
outputs an action. The chosen position is held during the *next* bar (T+1), so
there is no lookahead. Rewards are transaction-cost-adjusted PnL, optionally
shaped by a turnover penalty and a risk-adjusted reward mode.
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

REWARD_MODES = ("return", "log_return", "differential_sharpe")


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
        action_type: str = "continuous",
        reward_mode: str = "return",
        turnover_penalty: float = 0.0,
        dsr_eta: float = 0.01,
        seed: int | None = None,
    ):
        super().__init__()
        assert action_type in ("discrete", "continuous")
        assert reward_mode in REWARD_MODES

        self.df = df
        self.window = window
        self.commission = commission
        self.slippage = slippage
        self.stamp_tax = stamp_tax
        self.allow_short = allow_short
        self.action_type = action_type
        self.reward_mode = reward_mode
        self.turnover_penalty = turnover_penalty
        self.dsr_eta = dsr_eta

        self.returns = df["Close"].pct_change().fillna(0.0).to_numpy(dtype=np.float32)
        feats = compute_features(df)[FEATURE_COLS]
        self.features = feats.fillna(0.0).to_numpy(dtype=np.float32)
        self.n = len(df)
        self.n_feat = self.features.shape[1]

        obs_dim = window * self.n_feat + 1  # +1 for current position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        if action_type == "discrete":
            self.action_space = spaces.Discrete(3)
        else:
            low = 0.0 if not allow_short else -1.0
            self.action_space = spaces.Box(low=low, high=1.0, shape=(1,), dtype=np.float32)

        self.reset(seed=seed)

    def _action_to_position(self, action) -> float:
        if self.action_type == "discrete":
            a = int(action)
            if self.allow_short:
                return float(a - 1)  # 0->-1, 1->0, 2->+1
            return 1.0 if a == 2 else 0.0

        x = float(np.asarray(action).reshape(-1)[0])
        x = np.clip(x, -1.0, 1.0)
        if not self.allow_short:
            x = max(x, 0.0)
        return x

    def _get_obs(self) -> np.ndarray:
        day = min(self._day, self.n - 1)
        start = day - self.window + 1
        if start < 0:
            win = np.zeros((self.window, self.n_feat), dtype=np.float32)
            win[-day - 1 :] = self.features[0 : day + 1]
        else:
            win = self.features[start : day + 1]
        return np.concatenate([win.flatten(), np.array([self.position], dtype=np.float32)])

    def _shaped_reward(self, net: float) -> float:
        if self.reward_mode == "log_return":
            return float(np.log1p(max(net, -0.999)))
        if self.reward_mode == "differential_sharpe":
            return self._differential_sharpe(net)
        return net

    def _differential_sharpe(self, net: float) -> float:
        eta = self.dsr_eta
        a, b = self._dsr_a, self._dsr_b
        da = eta * (net - a)
        db = eta * (net * net - b)
        denom = (b - a * a) ** 1.5
        dsr = (b * da - 0.5 * a * db) / denom if denom > 1e-8 else 0.0
        self._dsr_a = a + da
        self._dsr_b = b + db
        return float(dsr)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._day = 0
        self.position = 0.0
        self._dsr_a = 0.0
        self._dsr_b = 0.0
        return self._get_obs(), {}

    def step(self, action):
        target = self._action_to_position(action)

        turnover = abs(target - self.position)
        sell = max(self.position - target, 0.0)
        cost = turnover * (self.commission + self.slippage) + sell * self.stamp_tax

        self.position = target
        self._day += 1

        if self._day >= self.n:
            return self._get_obs(), 0.0, True, False, {}

        gross = float(target * self.returns[self._day])
        net = gross - cost - self.turnover_penalty * turnover
        reward = self._shaped_reward(net)
        return self._get_obs(), reward, False, False, {}
