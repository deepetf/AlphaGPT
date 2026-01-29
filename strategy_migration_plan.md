# 可转债策略实盘改造计划 (Strategy Migration Plan)

## 0. 目标
将现有的 Crypto Event-Driven 策略引擎改造为适配 **AlphaGPT 可转债 (CB)** 的 **日频轮动 (Daily Rotation)** 策略引擎。

## 1. 核心差异对比

| 特性 | Crypto Runtime (`runner.py`) | CB Runtime (`cb_runner.py`) |
| :--- | :--- | :--- |
| **触发机制** | 7x24h 轮询 (Loop sleep) | 每日定时触发 (Cron/Schedule at 14:50) |
| **仓位逻辑** | 独立开平 (StopLoss/TP) | 组合调仓 (Rebalance Top-K) |
| **资金管理** | 单笔 Risk 计算 | 满仓等权 (1/TopK) |
| **数据流** | 实时 RPC/K线接口 | 日线更新 (Data Pipeline) |
| **执行接口** | Solana DEX SDK | A股柜台 (QMT/XtQuant/CTP) |

## 2. 改造方案: 新增 `cb_runner.py`

不修改原有代码，新建独立运行脚本。

### 2.1 核心类: `CBStrategyRunner`

```python
class CBStrategyRunner:
    def __init__(self):
        self.top_k = 10
        self.loader = CBDataLoader()  # 复用现有的 Loader
        self.portfolio = PortfolioManager("cb_portfolio.json")
        self.vm = StackVM()
        # 加载最佳公式
        self.formula = load_best_formula()

    def run_daily_rebalance(self):
        """
        每日调仓主逻辑 (建议在收盘前 14:50 执行)
        """
        # 1. 更新今日数据 (假定 Parquet 已更新)
        self.loader.load_data()
        
        # 2. 计算全市场因子分
        # [Assets]
        scores = self.calculate_scores()
        
        # 3. 选出 Top-K 目标池
        # 需应用与回测一致的过滤器: 
        # - 排除停牌 (Vol > 0)
        # - 排除临期 (Left Years > 0.5)
        # - 排除涨停/跌停 (可选)
        target_codes = self.select_top_k(scores)
        
        # 4. 生成调仓指令
        current_codes = self.portfolio.get_positions()
        
        sell_list = set(current_codes) - set(target_codes)
        buy_list = set(target_codes) - set(current_codes)
        
        print(f"Plan: SELL {len(sell_list)} -> BUY {len(buy_list)}")
        
        # 5. 执行交易 (Mock 或 真实接口)
        self.execute_rebalance(sell_list, buy_list)
```

### 2.2 交易接口抽象 (`execution/cb_trader.py`)

由于 A 股接口多样（QMT, XtQuant, 甚至手动），建议定义标准接口：

```python
class BaseCBTrader:
    def get_positions(self): pass
    def get_assets(self): pass
    def buy(self, code, amount/cash): pass
    def sell(self, code, amount/pct): pass
```

并提供一个 `FileTrader` 实现，仅生成 `orders.csv` 供检查。

### 2.3 组合管理 (`strategy_manager/portfolio.py`)
- 现有 `PortfolioManager` 基于 JSON，可直接复用。
- 需确认 `symbol` 格式兼容 (如 `110088.SH`)。

## 3. 稳健性一致性 (Consistency with V2.1)
确保 **实盘筛选逻辑** 与 **V2.1 回测逻辑** 严格一致：
- **配置文件**: 直接引用 `model_core.config.RobustConfig`，无需硬编码参数。
- **Active Ratio 保护**: 在生成订单前计算 `Active Ratio`。如果 `< MIN_ACTIVE_RATIO` (0.5)，**触发熔断**（本日不调仓，防止由数据缺失导致的错误换仓）。
- **New Factors**: 确保 `CBDataLoader` 加载了 `IV`, `VOL_STK_60`, `PREM_Z`，防止模型因缺列报错。
- **数据对齐**: 实盘必须使用与训练时完全相同的特征计算方式（特别是 `Rolling` 等时序算子，需确保历史数据长度足够）。

## 4. 实施步骤
1. [ ] 创建 `strategy_manager/cb_runner.py` 框架。
2. [ ] 实现 `calculate_scores` 与 Top-K 筛选（复用 `engine.py` 逻辑）。
3. [ ] 集成 `RobustConfig` 和 `Active Ratio` 熔断保护。
4. [ ] 实现 `rebalance` 生成买卖单。
5. [ ] 添加 `FileTrader` 输出交易计划文件。
6. [ ] 编写简单的 `run_cb_strategy.bat` 脚本用于每日调度。
