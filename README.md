# Backtesting

一个面向 A 股/ETF 的日线策略回测与研究工作台：内置真实交易成本建模、多种技术面与机器学习策略、以及基于 PPO + 注意力网络的强化学习智能体。提供基准对比、样本外 walk-forward 验证和完整的风险/收益指标体系。

> 深入的方法论、公式与实验结果见 [TECHNICAL_BRIEF.md](TECHNICAL_BRIEF.md)。

## 功能特性

**数据层**
- 通过 `yfinance` 下载 OHLCV 行情，自动归一化 MultiIndex 列
- 因果技术特征工程（收益率、波动率、均线偏离、RSI、MACD、布林带、量能）

**策略层**
- `Strategy` 抽象基类，统一 `generate_signals(df) -> signal` 接口，可插拔扩展
- 技术策略：双均线 / MACD / 动量 / 布林带均值回归 / RSI 均值回归 / Buy & Hold
- 监督学习：LightGBM 预测次日涨跌方向
- 强化学习：PPO 智能体，支持注意力策略网络、离散或连续仓位、多种奖励设计

**回测引擎**
- 逐 bar 模拟现金、持仓与权益曲线，支持做空与空仓
- 交易成本：手续费（含最低佣金）、滑点、印花税（卖出侧）、融券利息
- 仅在信号变化时交易，避免隐性每日再平衡成本；严格 T+1，无未来函数

**评估体系**
- 指标：总/年化收益、波动、Sharpe、Sortino、最大回撤、Calmar、胜率、换手率、市场暴露度
- 基准对比：基准收益/Sharpe、年化 Alpha、Beta
- 样本外验证：滚动 walk-forward，防止过拟合

**强化学习训练**
- Gymnasium 交易环境，MDP 建模与回测引擎口径一致
- 自定义注意力特征提取器（multi-head attention pooling）
- 终端实时方形面板（原地覆盖刷新）+ 完整训练日志落盘 + 周期性 Sharpe 评估

## 项目结构

```
├── data_loader.py     # 数据获取与预处理
├── features.py        # 技术特征工程（供 ML / RL 使用）
├── strategies.py      # Strategy 基类与技术/ML 策略
├── Engine.py          # 逐 bar 回测引擎（交易成本建模）
├── metrics.py         # 绩效指标 + 基准对比
├── backtest.py        # 策略对比 + walk-forward 验证
├── main.py            # 策略对比示例入口
├── rl_env.py          # Gymnasium 交易环境（RL）
├── rl_policy.py       # 注意力策略网络（自定义 features extractor）
├── train_rl.py        # PPO 训练、实时面板与样本外回测
└── TECHNICAL_BRIEF.md # 详细技术简报
```

## 安装

需要 Python >= 3.10，使用 [uv](https://docs.astral.sh/uv/) 管理依赖：

```bash
uv sync
```

核心依赖：`numpy`、`pandas`、`yfinance`；机器学习：`lightgbm`、`scikit-learn`；强化学习：`stable-baselines3`、`torch`、`gymnasium`。

## 快速开始

```bash
# 1) 策略对比 + LightGBM 样本外 walk-forward
uv run python main.py

# 2) 训练 PPO 智能体并做样本外回测（默认推荐配置）
uv run python train_rl.py
```

## 使用方式

### 回测单个策略

```python
from data_loader import load_data
from strategies import SmaCrossStrategy
from backtest import run_strategy

df = load_data("000001.SS", "2010-01-01", "2026-08-31")
strategy = SmaCrossStrategy(short_win=20, long_win=50, allow_short=False)
portfolio, metrics = run_strategy(strategy, df)
print(metrics)
```

### 对比多个策略

```python
from strategies import SmaCrossStrategy, MacdStrategy, MomentumStrategy
from backtest import compare_strategies

table = compare_strategies(
    {
        "SMA 20/50": SmaCrossStrategy(20, 50),
        "MACD": MacdStrategy(),
        "Momentum 60d": MomentumStrategy(60),
    },
    df,
)
print(table.to_string())
```

### 交易成本配置

成本在 `BacktesterEngine` 中配置，默认值贴合 A 股：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `commission` | `2.5e-4` | 佣金率（双边） |
| `min_commission` | `5.0` | 单笔最低佣金 |
| `slippage` | `5e-4` | 滑点（双边） |
| `stamp_tax` | `5e-4` | 印花税（仅卖出） |
| `short_borrow_fee` | `0.02` | 年化融券利率（按日计提） |
| `allow_short` | `True` | `False` 时仅做多（A 股受限市场） |

### 强化学习训练

```bash
uv run python train_rl.py \
  --timesteps 500000 \
  --train-end 2019-12-31 \
  --action-type continuous \
  --reward-mode differential_sharpe \
  --turnover-penalty 1e-3 \
  --eval-freq 40960
```

训练过程在终端以一个固定高度的方形面板原地刷新，展示完整训练指标（loss / KL / entropy / explained variance / lr / n_updates）和周期性评估结果（return / sharpe / sortino / max drawdown / turnover）。完整日志写入 `--log-dir`：

- `logs/progress.csv`、`logs/log.txt`：SB3 全量训练指标
- `logs/eval.log`：周期性样本外评估历史
- `ppo_trading.zip`：训练好的模型（已 gitignore）

主要参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--action-type` | `continuous` | `continuous` 连续仓位 `[-1,1]`，`discrete` 满仓多/空/平 |
| `--reward-mode` | `differential_sharpe` | `return` / `log_return` / `differential_sharpe` |
| `--turnover-penalty` | `1e-3` | 奖励中的换手惩罚系数 |
| `--eval-freq` | `40960` | 每隔多少 timesteps 做一次样本外评估 |
| `--no-short` | - | 仅做多 |
| `--device` | `cpu` | torch 设备（MlpPolicy 建议 CPU） |

## 策略信号约定

每个策略返回 `signal` 列，表示在 `t` 日收盘决定的、从 `t+1` 日起生效的目标仓位（`+1` 满仓多 / `0` 空仓 / `-1` 满仓空，或连续权重）。回测引擎内部处理 T+1 与交易成本，策略无需关心执行细节，且不存在未来函数。

## 当前结果

**基线策略对比**（`000001.SS`，2010-01-01 ~ 2026-08-31，含交易成本）

| 策略 | 总收益 | 年化 | Sharpe | 最大回撤 | 日均换手 |
|---|---|---|---|---|---|
| Buy & Hold（基准） | +20.43% | +1.17% | -0.04 | 52.32% | 0.02% |
| SMA 20/50 | -40.52% | -3.19% | -0.27 | 61.84% | 4.42% |
| MACD 12/26/9 | +13.30% | +0.78% | -0.06 | 61.32% | 15.57% |
| Momentum 60d | -20.96% | -1.46% | -0.19 | 74.47% | 11.15% |
| Bollinger MR | -69.74% | -7.18% | -1.08 | 72.76% | 10.35% |
| RSI MR | -39.51% | -3.09% | -0.54 | 45.31% | 6.33% |
| LightGBM（样本外 walk-forward） | -73.51% | -18.71% | -1.25 | 74.11% | 82.52% |

> 上证指数该区间本身年化仅约 +1%，各朴素策略均未跑赢买入持有。LightGBM 因换手过高（每日翻仓）被交易成本侵蚀——其毛收益实为 +7%，说明**换手成本是主要杀手**。

**PPO 强化学习**（注意力网络 + 连续仓位 + 差分 Sharpe + 换手惩罚，训练 100k 步；样本外 2020–2026）

| 指标 | PPO | Buy & Hold（同期） |
|---|---|---|
| 总收益 | +13.34% | +29.57% |
| Sharpe | ≈ 0.00 | 0.128 |
| 最大回撤 | 31.06% | — |
| 日均换手 | 0.27% | ~0 |
| 年化 Alpha | -1.25% | — |

换手约束成功（0.27%），但策略退化为近似满仓持有，最终跑输基准；且训练中各检查点样本外收益在 -11% ~ +45% 间大幅波动，反映 RL 的不稳定性。完整的实验设置、检查点数据与讨论见 [TECHNICAL_BRIEF.md](TECHNICAL_BRIEF.md)。

## 已知限制

- 日线级别，暂不支持盘中/高频数据与分钟级撮合
- 交易成本按固定比例 + 最低佣金近似，未建模随成交量变化的市场冲击
- 单资产研究，未引入基本面/宏观特征，也未做多资产组合
- RL 策略在单条价格路径上训练，样本效率低、方差大，尚不稳定

## 路线图

- [ ] 并行训练环境（`SubprocVecEnv`）+ `VecNormalize`，提升速度与稳定性
- [ ] 多种子训练取平均，报告均值 ± 标准差
- [ ] 换手/风险约束下的策略搜索与超参优化（Optuna）
- [ ] 多资产组合与截面策略
