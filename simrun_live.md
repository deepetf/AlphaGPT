# sim_run 实时模拟盘实施计划（SQL版）

## 1. 目标与范围

### 1.1 总目标
- `sim_run` 支持实时模拟盘：每日基于最新 SQL 数据计算最新持仓，用实时价格执行调仓，计算组合当日涨跌幅与净值等关键指标。
- `holdings`、`nav_history`、`trade_history` 统一入库（不再依赖 JSON 作为主存储）。

### 1.2 对齐原则
- 策略逻辑与 `verify_strategy.py`、`model_core/backtest.py` 保持同口径（选股、调仓、止盈、收益计算定义一致）。
- 回放模式与实时模式共享核心调仓与记账路径，避免双轨逻辑漂移。

### 1.3 本次边界
- 重点实现 `sim_run` 的 SQL 化状态持久化与实时可运行框架。
- 不改动 `verify_strategy.py` 的数据源路径（仍可保留本地回测输入）。

## 2. 数据要求

### 2.1 输入数据（已有）
- 市场与因子数据来源：`CB_HISTORY.CB_DATA`（通过 `Config.CB_DB_DSN` + SQLAlchemy 访问）。
- 交易日历：可从 `CB_DATA.trade_date` 去重得到。
- 实时价格：优先使用实时行情源；无实时源时可回退到 SQL 当日最新价（EOD 或近实时快照）。

### 2.2 输出数据（新增落库）
- 每日净值：`sim_nav_history`
- 每日持仓快照：`sim_daily_holdings`
- 逐笔交易流水：`sim_trade_history`

### 2.3 关键业务字段
- `strategy_id`：支持多策略并行隔离。
- `trade_date`：交易业务日期。
- `side`：支持 `BUY` / `SELL` / `SELL-TP`。

## 3. 运行逻辑（T日）

1. 读取 T 日可用股票池与因子输入（SQL）。
2. 计算目标持仓（Top-K + 风控约束）。
3. 先执行止盈检查（可触发 `SELL-TP`），再执行调仓（卖出旧仓、买入新仓）。
4. 用成交结果更新现金与持仓。
5. 基于“交易后持仓 + 收盘/最新估值价”计算当日 NAV 与收益指标。
6. 按日写入三类状态表（交易流水、持仓快照、净值）。

## 4. 与回测一致性的核心口径

- 调仓顺序：先卖后买，保持资金约束真实有效。
- 止盈口径：与模拟盘一致，触发时 `side='SELL-TP'`，并进入当日收益路径。
- 收益口径：以当日最终持仓与估值计算 `daily_ret`、`cum_ret`、`mdd`。
- 状态口径：每日快照唯一（`strategy_id + trade_date`），避免重复写入导致净值漂移。

## 5. 数据库表结构定义（MySQL 8）

```sql
-- 1) 每日净值
CREATE TABLE IF NOT EXISTS sim_nav_history (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  strategy_id VARCHAR(64) NOT NULL COMMENT '策略ID，如 default',
  trade_date DATE NOT NULL COMMENT '交易日',
  nav DECIMAL(20,4) NOT NULL COMMENT '净值=现金+持仓市值',
  cash DECIMAL(20,4) NOT NULL COMMENT '现金余额',
  holdings_value DECIMAL(20,4) NOT NULL COMMENT '持仓市值',
  holdings_count INT NOT NULL COMMENT '持仓数量',
  daily_ret DECIMAL(18,8) NOT NULL COMMENT '当日收益率',
  cum_ret DECIMAL(18,8) NOT NULL COMMENT '累计收益率',
  mdd DECIMAL(18,8) NOT NULL COMMENT '回撤',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_nav_strategy_date (strategy_id, trade_date),
  KEY idx_nav_trade_date (trade_date),
  KEY idx_nav_strategy (strategy_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2) 每日持仓快照（EOD）
CREATE TABLE IF NOT EXISTS sim_daily_holdings (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  strategy_id VARCHAR(64) NOT NULL COMMENT '策略ID',
  trade_date DATE NOT NULL COMMENT '交易日',
  code VARCHAR(32) NOT NULL COMMENT '转债代码',
  name VARCHAR(128) NOT NULL COMMENT '名称',
  shares BIGINT NOT NULL COMMENT '持仓张数',
  avg_cost DECIMAL(20,6) NOT NULL COMMENT '持仓均价',
  last_price DECIMAL(20,6) NOT NULL COMMENT '当日用于估值价格',
  entry_date DATE NULL COMMENT '首次建仓日',
  market_value DECIMAL(20,4) NOT NULL COMMENT 'shares*last_price',
  pnl DECIMAL(20,4) NOT NULL COMMENT '浮盈亏',
  pnl_pct DECIMAL(18,8) NOT NULL COMMENT '浮盈亏比例',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_holding_strategy_date_code (strategy_id, trade_date, code),
  KEY idx_holding_trade_date (trade_date),
  KEY idx_holding_code_date (code, trade_date),
  KEY idx_holding_strategy (strategy_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3) 交易流水（逐笔）
CREATE TABLE IF NOT EXISTS sim_trade_history (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  strategy_id VARCHAR(64) NOT NULL COMMENT '策略ID',
  trade_date DATE NOT NULL COMMENT '交易日',
  trade_time DATETIME NOT NULL COMMENT '成交时间',
  code VARCHAR(32) NOT NULL COMMENT '转债代码',
  name VARCHAR(128) NOT NULL COMMENT '名称',
  side VARCHAR(16) NOT NULL COMMENT 'BUY/SELL/SELL-TP',
  shares BIGINT NOT NULL COMMENT '成交张数',
  price DECIMAL(20,6) NOT NULL COMMENT '成交价',
  amount DECIMAL(20,4) NOT NULL COMMENT 'BUY为总成本, SELL为净回款(与当前代码一致)',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_trade_strategy_date (strategy_id, trade_date),
  KEY idx_trade_code_date (code, trade_date),
  KEY idx_trade_side_date (side, trade_date),
  KEY idx_trade_time (trade_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

## 6. 写入策略与事务约定

- `sim_trade_history`：append-only，逐笔 `INSERT`。
- `sim_daily_holdings`：按日覆盖写入：
  - `DELETE FROM sim_daily_holdings WHERE strategy_id=? AND trade_date=?`
  - 再批量 `INSERT` 当日全部持仓。
- `sim_nav_history`：`INSERT ... ON DUPLICATE KEY UPDATE` 幂等更新。
- 建议单日事务边界：
  1. 写交易流水
  2. 写持仓快照
  3. 写净值
  4. `COMMIT`

## 7. 预计代码改造清单

- 新增：`strategy_manager/sql_state_store.py`
  - 封装三张表的读写（按策略、按日期）。
  - 提供 `load_latest_state(strategy_id)` 初始化当日运行状态。
- 修改：`strategy_manager/run_sim.py`
  - 增加 SQL 存储开关与默认启用逻辑。
  - 回放/实时路径统一调用 SQL 状态层。
- 修改：`strategy_manager/sim_runner.py`
  - 将 `portfolio/nav/trader` 的 JSON 持久化改为 SQL 持久化接口。
  - 保留内存对象作为运行态，落库由 `sql_state_store` 负责。
- 修改：`execution/sim_trader.py`
  - 成交记录构造保留，落库由外层统一批量提交。
- 可选迁移脚本：`infra/migrations/xxx_create_sim_tables.sql`

## 8. 单元测试与验证计划

### 8.1 单元测试
- `tests/test_sql_state_store.py`
  - `sim_nav_history` upsert 幂等性
  - `sim_daily_holdings` 覆盖写正确性
  - `sim_trade_history` append 正确性
- `tests/test_sim_runner_sql_mode.py`
  - 单日运行后 DB 三表数据完整性
  - `SELL-TP` 记录侧标正确性

### 8.2 回放一致性验证
- 区间：`2025-01-01 ~ 2025-12-31`
- 比较项：
  - 每日持仓集合重合度
  - `daily_ret` 序列 MAE / MaxAE / Corr
  - 累计收益与回撤差异
- 通过标准（建议）：
  - `corr > 0.99`
  - `MAE < 5e-4`
  - 持仓日级重合度显著高于改造前

## 9. 产出物

- 设计文档：`simrun_live.md`（本文件）
- 建表 SQL：`infra/migrations/*_create_sim_tables.sql`
- SQL 状态层代码：`strategy_manager/sql_state_store.py`
- `sim_run` SQL 化改造代码与测试用例
- SQL 回放验收脚本：`tests/verify_sim_sql_replay.py`

## 11. 回放验收命令

```bash
# 先跑回放
python strategy_manager/run_sim.py --start-date 2026-01-01 --end-date 2026-02-13 --state-backend sql --replay-strict --replay-source sql_eod

# 再跑 SQL 产出一致性验收
python tests/verify_sim_sql_replay.py --strategy-id default --start-date 2026-01-01 --end-date 2026-02-13 --top-k 10
```

验收结果输出到 `tests/artifacts`：
- `sim_sql_nav_check_*.csv`
- `sim_sql_nav_identity_fail_*.csv`
- `sim_sql_holdings_count_fail_*.csv`
- `sim_sql_suspicious_days_*.csv`
- `sim_sql_candidate_missing_*.csv`
- `sim_sql_replay_report_*.md`

## 10. 风险与注意事项

- 同日重复运行：必须依赖唯一键与 upsert，避免重复快照/净值。
- 行情时间戳差异：实时价与 EOD 价混用时需明确估值口径（建议在表中保留来源字段，可后续追加）。
- 多进程并发：同 `strategy_id` 建议串行运行，避免竞争写入。
