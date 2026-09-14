# Backtesting

构建一个能够处理日线/高频数据、模拟真实市场交易的策略回测系统。不仅能计算收益率，还能模拟交易成本（手续费、滑点）与资金占用。

## 功能特性

- 数据获取：通过 `yfinance` 下载标的历史行情（OHLCV）
- 信号生成：双均线（SMA）交叉策略，T+1 信号执行
- 回测引擎：按持仓变动计算交易成本（手续费 + 滑点），输出每日策略收益与资金曲线
- 绩效指标：年化收益、年化波动、Sharpe 比率、最大回撤

## 项目结构

```
├── data_loader.py   # 数据获取与预处理
├── signals.py       # 交易信号生成（双均线策略）
├── Engine.py        # 回测引擎
├── metrics.py       # 绩效指标计算
└── main.py          # 主入口，串联回测流程
```

## 快速开始

```bash
# 安装依赖（需要 Python >= 3.10）
uv sync

# 运行回测
uv run python main.py
```

默认对 `000001.SS`（上证指数）在 `2000-01-01` 至 `2026-08-31` 区间，使用 20/50 日双均线策略回测。

## 使用方式

1. 在 `main.py` 中修改 `ticker`、`start`、`end` 以切换标的与时间区间
2. 在 `signals.py` 的 `generate_sma_signals` 中调整 `short_win` / `long_win` 修改均线窗口
3. 在 `Engine.py` 的 `BacktesterEngine` 中调整 `init_cap`、`commission`、`slippage`

## 已知限制

- 仓位仅支持满仓多头 / 满仓空头（±1），无现金管理
- 未模拟做空成本、印花税、最低佣金
- 无基准对比、无样本外验证
- 数据层依赖 `yfinance` 返回格式，需显式处理 MultiIndex 列
