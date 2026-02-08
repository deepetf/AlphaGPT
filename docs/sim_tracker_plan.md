# 可转债模拟盘跟踪系统 (CB Simulation Tracker) - V2

## 1. 目标与参数确认

| 参数 | 值 |
|------|-----|
| 运行频率 | 日频，每日 14:50 触发 |
| 止盈逻辑 | 复用 `backtest.py` 向量化止盈 (需 open, high) |
| 止损逻辑 | 不启用 |
| 初始资金 | 1,000,000 元 |
| 实时行情 | Mini QMT 接口 (用户提供) |
| CB 特性数据 | 本地 SQL 数据库 (用户提供) |
| 历史数据 | 用户提供接口 (用于因子计算) |

---

## 2. 系统架构

```
┌─────────────────┐   ┌─────────────────┐
│  Mini QMT API   │   │  Local SQL DB   │
│  (实时行情)      │   │  (CB特性数据)    │
└────────┬────────┘   └────────┬────────┘
         │                     │
         v                     v
    ┌────────────────────────────────┐
    │     RealtimeDataProvider       │
    │   (统一数据适配层)               │
    └────────────────┬───────────────┘
                     │
                     v
    ┌────────────────────────────────┐
    │        SimulationRunner        │
    │   14:50 定时触发                │
    │   - 因子计算 (StackVM)          │
    │   - Top-K 筛选                  │
    │   - 止盈检测                    │
    │   - 调仓决策                    │
    └────────────────┬───────────────┘
                     │
         ┌───────────┴───────────┐
         v                       v
┌─────────────────┐     ┌─────────────────┐
│   SimTrader     │     │  NavTracker     │
│  (模拟成交)      │     │  (净值记录)      │
└─────────────────┘     └─────────────────┘
         │                       │
         v                       v
┌─────────────────┐     ┌─────────────────┐
│ CBPortfolio     │     │ nav_history.json│
│ Manager         │     │ (绩效归档)       │
└─────────────────┘     └─────────────────┘
```

---

## 3. 数据字段映射 (CB_DATA 表)

> [!IMPORTANT]
> 请核对并补全下表中的 `CB_DATA 字段名` 列，确保与数据库一致。

| 序号 | 因子名 (INPUT_FEATURE) | 中文含义 | CB_DATA 字段名 |
|------|------------------------|----------|----------------|
| 1 | `CLOSE` | 收盘价 | `close` |
| 2 | `VOL` | 成交量 | `amount` |
| 3 | `PREM` | 转股溢价率 | |
| 4 | `DBLOW` | 双低值 | |
| 5 | `REMAIN_SIZE` | 剩余规模 | `remain_cap` |
| 6 | `PCT_CHG` | 涨跌幅 | |
| 7 | `PCT_CHG_5` | 五日涨跌幅 | |
| 8 | `VOLATILITY_STK` | 正股波动率 | |
| 9 | `PCT_CHG_STK` | 正股涨跌幅 | |
| 10 | `PCT_CHG_5_STK` | 正股五日涨跌幅 | |
| 11 | `PURE_VALUE` | 纯债价值 | |
| 12 | `ALPHA_PCT_CHG_5` | 五日涨跌幅差 | |
| 13 | `CAP_MV_RATE` | 转债市值占比 | |
| 14 | `TURNOVER` | 换手率 | |
| 15 | `IV` | 隐含波动率 | |
| 16 | `VOL_STK_60` | 正股60日波动率 | |
| 17 | `PREM_Z` | 溢价率 Z-Score | |

**附加字段** (止盈逻辑需要):
| 用途 | CB_DATA 字段名 |
|------|----------------|
| 开盘价 (open) | `open` |
| 最高价 (high) | `high` |
| 交易日期 | `trade_date` |
| 转债代码 | `code` |
| 转债名称 | `name` |

---

## 4. 新增模块

### A. `data_pipeline/realtime_provider.py`
统一封装 Mini QMT 和 SQL 数据源。

```python
class RealtimeDataProvider:
    def __init__(self, qmt_client, sql_engine):
        self.qmt = qmt_client  # 用户注入
        self.sql = sql_engine  # 用户注入

    def get_quotes(self) -> pd.DataFrame:
        """返回 [code, open, high, close, volume, ...]"""
        pass

    def get_cb_features(self) -> pd.DataFrame:
        """返回 [code, pure_value, remain_size, prem, iv, ...]"""
        pass

    def build_feat_tensor(self, history_loader) -> torch.Tensor:
        """合并行情+特性，构建因子计算所需的 feat_tensor"""
        pass
```

---

### B. `strategy_manager/sim_runner.py`
模拟盘主控逻辑。

**核心流程**:
1. 加载公式 & 持仓
2. 拉取实时行情 + CB 特性
3. 构建 `feat_tensor` → `StackVM` 执行
4. **止盈检测**: 对持仓标的，检查 `high >= entry_price * (1 + TP_RATIO)`
5. Top-K 筛选 → 调仓决策
6. 模拟成交 → 更新持仓
7. 净值快照 → 归档

---

### C. `execution/sim_trader.py`
模拟成交器 (不操作真实柜台)。

```python
class SimTrader:
    def execute(self, orders: List[Order], prices: Dict[str, float]) -> List[TradeRecord]:
        """以 prices 字典中的价格模拟成交，返回成交记录"""
        pass
```

---

### D. `strategy_manager/nav_tracker.py`
净值与绩效追踪。

**输出文件**: `execution/portfolio/nav_history.json`
```json
{
  "initial_capital": 1000000.0,
  "records": [
    {"date": "2026-02-07", "nav": 1005000.0, "daily_ret": 0.005, "cum_ret": 0.005, "mdd": 0.0}
  ]
}
```

---

## 4. 修改模块

### `strategy_manager/cb_portfolio.py`
- 增加 `calculate_nav(prices)` 方法
- 增加 `get_entry_prices()` 用于止盈检测

---

## 5. 止盈逻辑 (复用 backtest.py)

每日 14:50 检测持仓标的：
1. 获取当日 `open`, `high` (来自 Mini QMT)
2. 对于每个持仓 `pos`：
   - **跳空止盈**: `open >= pos.avg_cost * (1 + TP_RATIO)` → 以 `open` 卖出
   - **盘中止盈**: `high >= pos.avg_cost * (1 + TP_RATIO)` → 以 `avg_cost * (1 + TP_RATIO)` 卖出
3. 止盈后释放资金，参与当日 Top-K 再投资

---

## 6. 待用户提供

> [!IMPORTANT]
> 实施前需要您提供以下接口适配信息：

1.  **Mini QMT 接口**:
    - 如何初始化连接？
    - 获取全市场快照的方法签名是什么？
    - 返回的 DataFrame 包含哪些列？

2.  **本地 SQL 数据库**:
    - 连接字符串 (SQLAlchemy 格式)?
    - 表名 & 字段名？
    - 是否需要按日期过滤？

3.  **历史数据接口** (可选，如因子需要 T-60):
    - 是否与 SQL 同源？
    - 查询方式？

---

## 7. 实施步骤

| 阶段 | 任务 | 预计文件 |
|------|------|----------|
| 1 | 创建 `RealtimeDataProvider` 骨架 | `data_pipeline/realtime_provider.py` |
| 2 | 实现 `SimTrader` 模拟成交 | `execution/sim_trader.py` |
| 3 | 实现 `NavTracker` 净值追踪 | `strategy_manager/nav_tracker.py` |
| 4 | 扩展 `CBPortfolioManager` | `strategy_manager/cb_portfolio.py` |
| 5 | 实现 `SimulationRunner` 主控 | `strategy_manager/sim_runner.py` |
| 6 | 编写单元测试 | `tests/test_sim_runner.py` |
| 7 | 集成测试 (Mock 数据) | - |
| 8 | 对接真实 Mini QMT + SQL | - |
