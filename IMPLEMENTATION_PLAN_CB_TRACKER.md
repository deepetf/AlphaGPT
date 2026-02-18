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
    │   (跨策略共享数据适配层)          │
    └────────────────┬───────────────┘
                     │
                     v
    ┌────────────────────────────────┐
    │       MultiSimRunner           │
    │   (多策略主控: 调度与隔离)        │
    └───────────┬───────────┬────────┘
                │           │
                v           v
    ┌────────────────┐ ┌────────────────┐
    │SimulationRunner│ │SimulationRunner│
    │  [策略 A]      │ │  [策略 B]      │
    └──────┬─────────┘ └──────┬─────────┘
           │                  │
    ┌──────┴──────┐    ┌──────┴──────┐
    │  SimTrader  │    │  SimTrader  │
    │  NavTracker │    │  NavTracker │
    └─────────────┘    └─────────────┘
```

---

## 3. 数据字段映射 (CB_DATA 表)

> [!IMPORTANT]
> 请核对并补全下表中的 `CB_DATA 字段名` 列，确保与数据库一致。

| 序号 | 因子名 (INPUT_FEATURE) | 中文含义 | CB_DATA 字段名 |
|------|------------------------|----------|----------------|
| 1 | `CLOSE` | 收盘价 | `close` |
| 2 | `VOL` | 成交量 | `vol` |
| 3 | `PREM` | 转股溢价率 | `conv_prem` |
| 4 | `DBLOW` | 双低值 | `dblow` |
| 5 | `REMAIN_SIZE` | 剩余规模 | `remain_size` |
| 6 | `PCT_CHG` | 涨跌幅 | `pct_chg` |
| 7 | `PCT_CHG_5` | 五日涨跌幅 | `pct_chg_5` |
| 8 | `VOLATILITY_STK` | 正股波动率 | `volatility_stk` |
| 9 | `PCT_CHG_STK` | 正股涨跌幅 | `pct_chg_stk` |
| 10 | `PCT_CHG_5_STK` | 正股五日涨跌幅 | `pct_chg_5_stk` |
| 11 | `PURE_VALUE` | 纯债价值 | `pure_value` |
| 12 | `ALPHA_PCT_CHG_5` | 五日涨跌幅差 | `alpha_pct_chg_5` |
| 13 | `CAP_MV_RATE` | 转债市值占比 | `cap_mv_rate` |
| 14 | `TURNOVER` | 换手率 | `turnover` |
| 15 | `IV` | 隐含波动率 | `IV` |
| 16 | `VOL_STK_60` | 正股60日波动率 | `stock_vol60d` |
| 17 | `PREM_Z` | 溢价率 Z-Score | `convprem_zscore` |

**附加字段** (止盈/过滤逻辑需要):
| 用途 | CB_DATA 字段名 |
|------|----------------|
| 开盘价 (open) | `open` |
| 最高价 (high) | `high` |
| 交易日期 | `trade_date` |
| 转债代码 | `code` |
| 转债名称 | `name` |
| 剩余年限 | `left_years` |

---

## 4. 新增模块

### A. `data_pipeline/realtime_provider.py`
统一封装 Mini QMT (`xtquant.xtdata`) 和 SQL 数据源。

**核心职责**:
1. **连接管理**: `xtdata` 本质上与本地运行的 Mini QMT 执行通讯。
2. **行情获取**: 使用 `xtdata.subscribe_quote` 订阅并使用 `xtdata.get_market_data_ex` 获取实时快照。
3. **数据对齐**: 将 QMT 行情与 SQL 中的 CB 特效数据按 `code` 合并。

```python
from xtquant import xtdata

class RealtimeDataProvider:
    def __init__(self, sql_engine):
        self.sql = sql_engine

    def download_data(self, code_list: List[str]):
        """下载/更新历史数据 (用于因子计算)"""
        for code in code_list:
            xtdata.download_history_data(code, period='1d', incrementally=True)

    def get_realtime_quotes(self, code_list: List[str]) -> pd.DataFrame:
        """从 Mini QMT 获取实时行情 (open, high, close, vol)"""
        for code in code_list:
            xtdata.subscribe_quote(code, period='1d', count=-1)
        
        # 获取快照
        data = xtdata.get_market_data_ex([], code_list, period='1d')
        # data 为字典，key 为标的代码，value 为 DataFrame 或 Series
        return self._format_qmt_data(data)

    def get_cb_features(self) -> pd.DataFrame:
        """从本地 SQL 数据库获取最新特数据 (pure_value, prem, etc.)"""
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

### E. `strategy_manager/multi_sim_runner.py`
多策略主控类，负责并行管理多个 `SimulationRunner`。

**核心职责**:
1. **配置加载**: 解析 `strategies_config.json`。
2. **实例隔离**: 为每个策略创建独立的目录及组件。
3. **统一调度**: 每日 14:50 依次执行所有启用策略。

---

### F. `strategy_manager/strategies_config.json`
存储多策略定义的 JSON 配置文件。支持 `formula_path` 和内联 `formula` 两种定义方式。

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

## 7. 分步实施与单元测试计划

我们将按照“数据驱动 -> 状态管理 -> 模拟撮合 -> 逻辑闭环”的顺序分步推进。

### 第一阶段：数据适配器 (`data_pipeline/realtime_provider.py`)
**目标**: 实现 Mini QMT 与 SQL 数据的整合，为因子计算提供张量。
- **实施**:
  - 实现 `RealtimeDataProvider.__init__` (SQL 连接)。
  - 实现 `download_data` 和 `get_realtime_quotes` (Mini QMT)。
  - 实现 `build_feat_tensor` (对齐行情与特性)。
- **单元测试**: `tests/test_realtime_provider.py`
  - 使用 `unittest.mock` 模拟 `xtdata` 的返回。
  - 验证获取的全市场行情 DataFrame 的列名和数据类型是否正确。
  - 验证合并 QMT 和 SQL 数据后，最终生成的 `feat_tensor` 维度是否符合 `(Assets, Features)`。

### 第二阶段：组合管理与绩效追踪 (`strategy_manager/cb_portfolio.py` & `nav_tracker.py`)
**目标**: 实现模拟盘的“账本”功能。
- **实施**:
  - 扩展 `CBPortfolioManager` 支持 `calculate_nav`。
  - 实现 `NavTracker` 记录每日绩效指标。
- **单元测试**: `tests/test_nav_tracker.py`
  - 构造模拟持仓和价格。
  - 验证 `NAV` (净值) 计算逻辑是否正确计算了持仓市值+剩余现金。
  - 验证 `mdd` (最大回撤) 在净值下跌时的计算逻辑。

### 第三阶段：模拟成交引擎 (`execution/sim_trader.py`)
**目标**: 实现无风险模拟成交。
- **实施**:
  - 实现 `SimTrader.execute`，根据传入的实时价直接更新持仓。
- **单元测试**: `tests/test_sim_trader.py`
  - 验证买入指令后，`Portfolio` 里的 `shares` 和 `avg_cost` 正确更新。
  - 验证卖出指令后，现金正确增加。

### 第四阶段：策略主控逻辑 (`strategy_manager/sim_runner.py`)
**目标**: 串联整个调仓与止盈流程。
- **实施**:
  - 实现 `SimulationRunner.run_daily`。
  - 复用 `backtest.py` 逻辑实现 14:50 止盈检测。
- **单元测试**: `tests/test_sim_runner.py` (集成测试/Mock)
  - 使用 Mock 数据模拟 14:50 行情。
  - 验证：如果 `high` 触发止盈，是否优先执行了止盈卖出。
  - 验证：Top-K 选股结果是否正确转化为 `Order` 列表。

- 连接真实 Mini QMT 客户端。
- 整合 `RealtimeDataProvider` 的所有接口。

### 第六阶段：多策略并行扩展 (`strategy_manager/multi_sim_runner.py`)
**目标**: 实现多个 Alpha 因子同时跟踪。
- **实施**:
  - 实现 `StrategyConfig` 配置加载器。
  - 重构 `SimulationRunner` 支持通过配置初始化。
  - 实现 `MultiSimRunner` 调度逻辑。
- **单元测试**: `tests/test_strategy_config.py`
  - 验证 JSON 配置解析。
  - 验证内联公式加载。
  - 验证不同策略在各自目录（`execution/portfolio/{id}/`）的数据隔离。
