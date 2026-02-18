# sim_run Live 实施计划（基线版）

## 1. 目标
- 仅保留两条路径：`live` 与 `strict_replay`。
- `live` 与 `strict_replay` 共用同一套选股/调仓主逻辑，减少口径漂移。
- 模拟盘状态统一落 SQL，并按 dataset 隔离：`replay` / `live`。

## 2. strict_replay 数据来源规则
- `strict_replay` 的数据来源由 `replay_source` 指定。
- 当前阶段为保证与回测强对齐，仅允许：`replay_source=sql_eod`。
- 若传入其他值（如 `parquet`），应直接报错，不允许 fallback。

## 3. SQL 状态存储
- replay 表：
  - `sim_nav_history`
  - `sim_daily_holdings`
  - `sim_trade_history`
- live 表：
  - `sim_live_nav_history`
  - `sim_live_daily_holdings`
  - `sim_live_trade_history`

## 4. 入口与参数
- 统一入口：`strategy_manager/run_sim.py`
- 关键参数：
  - `--mode {live,strict_replay}`
  - `--state-backend {sql,json}`
  - `--live-quote-source {dummy,qmt}`
  - `--replay-source {sql_eod,parquet}`（当前 strict_replay 仅允许 sql_eod）

## 5. 实施要点
- `run_sim` 将参数下传到 `MultiSimRunner`。
- `MultiSimRunner` 支持本次运行级别的 `replay_source` 覆盖。
- `SimulationRunner._ensure_backtest_context()` 对 strict_replay 做来源硬校验。
- `SQLStateStore` 使用 dataset 映射表名读写。

## 6. 验证项
- 语法验证：`py_compile` 覆盖 run_sim/multi_sim_runner/sim_runner/sql_state_store/realtime_provider。
- 行为验证：
  - strict_replay + `--replay-source sql_eod` 正常运行。
  - strict_replay + `--replay-source parquet` 立即报错。
  - live 模式写入 `sim_live_*` 三张表。

## 7. 后续
- 对接真实实时行情 API，替换 dummy。
- 增加 live vs strict_replay 差异报告脚本（持仓、交易、NAV）。
