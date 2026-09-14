# Backtesting

一个能够处理日线数据、模拟真实市场交易成本的策略回测系统。支持多种技术面策略与机器学习策略，提供基准对比、样本外（walk-forward）验证和完整的绩效指标。

## 功能特性

- **数据层**：通过 `yfinance` 下载 OHLCV，自动归一化 MultiIndex 列
- **策略层**：`Strategy` 抽象基类，内置双均线 / MACD / 动量 / 布林带均值回归 / RSI / Buy&Hold / LightGBM 机器学习策略，可插拔扩展
- **回测引擎**：逐 bar 模拟现金与持仓，支持做空，建模手续费、最低佣金、滑点、印花税（卖出侧）、融券利息
- **绩效指标**：年化收益、波动、Sharpe、Sortino、最大回撤、Calmar、胜率、换手率、市场暴露度，以及相对基准的 Alpha/Beta
- **样本外验证**：滚动 walk-forward 回测，防止过拟合

## 项目结构

```
├── data_loader.py   # 数据获取与预处理
├── features.py      # 技术特征工程（供 ML 策略使用）
├── strategies.py    # 策略基类与各策略实现
├── Engine.py        # 逐 bar 回测引擎（交易成本建模）
├── metrics.py       # 绩效指标 + 基准对比
├── backtest.py      # 策略对比与 walk-forward 验证
└── main.py          # 示例入口
```

## 快速开始

```bash
# 安装依赖（需要 Python >= 3.10）
uv sync

# 运行示例（策略对比 + LightGBM 样本外验证）
uv run python main.py
```

## 使用方式

```python
from data_loader import load_data
from strategies import SmaCrossStrategy
from backtest import run_strategy

df = load_data("000001.SS", "2010-01-01", "2026-08-31")
strategy = SmaCrossStrategy(short_win=20, long_win=50, allow_short=False)
portfolio, metrics = run_strategy(strategy, df)
print(metrics)
```

- 交易成本在 `BacktesterEngine` 中配置：`commission`（佣金率）、`min_commission`（最低佣金）、`slippage`（滑点）、`stamp_tax`（印花税，卖出侧）、`short_borrow_fee`（年化融券利率）
- 策略通过 `allow_short=False` 可切换为仅做多（A 股等做空受限市场）
- `compare_strategies` 一键对比多个策略；`walk_forward` 做滚动样本外验证

## 策略信号约定

每个策略返回 `signal` 列，表示在 `t` 日收盘决定的、从 `t+1` 日起生效的目标仓位（+1 满仓多 / 0 空仓 / -1 满仓空）。回测引擎内部处理 T+1 与交易成本，策略无需关心执行细节，且不存在未来函数。

## 已知限制

- 日线级别，暂不支持盘中/高频数据与分钟级撮合
- 交易成本按固定比例 + 最低佣金近似，未建模市场冲击（随成交量变化）
- 机器学习策略为单资产日线方向预测的基线，未引入基本面/宏观特征
