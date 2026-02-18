# sim_run 实时与回放实施文档（SQL版）

## 1. 目标
- 仅保留两条执行路径：`live` 与 `strict_replay`。
- `live` 与 `strict_replay` 共享同一套选股/调仓核心逻辑，减少口径漂移。
- 模拟盘状态写入 SQL，且按数据集隔离：`replay` 与 `live` 各自独立三张表（NAV/持仓/交易）。

## 2. 当前实现状态
- 已完成 `run_sim` 双模式入口：
  - `--mode live`
  - `--mode strict_replay`
- 已移除旧普通 replay 分支（非 strict）。
- 已支持 `dataset` 维度映射到不同 SQL 表名。
- 已新增 live 三张表迁移脚本：
  - `infra/migrations/20260215_create_simrun_live_dataset_tables.sql`
- `live` 已支持 `--live-quote-source dummy`（占位）与 `qmt`（保留接口）。
- 为兼容旧命令，保留参数：
  - `--replay-strict`（兼容别名）
  - `--replay-source`（废弃，仅提示）

## 3. 运行模式与数据源
### 3.1 strict_replay
- 目标：历史回放（EOD口径）。
- 数据：SQL EOD + strict loader。
- 状态表：`sim_*`（replay 数据集）。

### 3.2 live
- 目标：实盘时段模拟运行。
- 数据：SQL 特征 + 实时行情（当前可用 dummy，占位后续接 API）。
- 状态表：`sim_live_*`（live 数据集）。

## 4. SQL 表映射
- replay：
  - `sim_nav_history`
  - `sim_daily_holdings`
  - `sim_trade_history`
- live：
  - `sim_live_nav_history`
  - `sim_live_daily_holdings`
  - `sim_live_trade_history`

## 5. 命令示例
### 5.1 strict_replay 区间回放
```bash
python strategy_manager/run_sim.py \
  --mode strict_replay \
  --start-date 2026-01-01 \
  --end-date 2026-02-13 \
  --state-backend sql
```

### 5.2 strict_replay 单日
```bash
python strategy_manager/run_sim.py \
  --mode strict_replay \
  --date 2026-02-13 \
  --state-backend sql \
  --strategy-id king_factor_v1
```

### 5.3 live 单日（dummy 行情）
```bash
python strategy_manager/run_sim.py \
  --mode live \
  --date 2026-02-13 \
  --state-backend sql \
  --live-quote-source dummy \
  --strategy-id king_factor_v1
```

## 6. 关键代码位置
- 入口与模式分发：`strategy_manager/run_sim.py`
- 策略集合运行器：`strategy_manager/multi_sim_runner.py`
- 单策略执行核心：`strategy_manager/sim_runner.py`
- SQL 状态存取：`strategy_manager/sql_state_store.py`
- 实时数据提供器（含 dummy）：`data_pipeline/realtime_provider.py`

## 7. 下一阶段
- 对接真实实时行情 API，替换 dummy。
- 增加 live vs strict_replay 对比脚本（持仓重合率、NAV 偏差、交易差异归因）。
- 增加针对 SQLStateStore 与双模式路径的单元测试。
