# AlphaGPT Changelog

## [V5.95] - 2026-03-09

### Added
- **Top-Factor Post Selection**: 新增 `model_core/select_top_factors.py` 与 `model_core/top_factor_config.yaml`，支持从训练输出中合并候选、公式去重、统一重评估、硬过滤、加权排序与相似度去重，自动产出 Top-3 因子结果。
- **AI Factor Review**: 新增 `model_core/factor_ai_review.py`，支持对候选因子输出结构化金融含义评审，并可生成 Markdown 报告。
- **Windows Venv Bootstrap**: 新增 `setup_venv.ps1`，用于创建标准 Windows 虚拟环境 `.venv`、升级 `pip` 并安装项目依赖。

### Changed
- **Dependency Manifest Refresh**: 重新整理 `requirements.txt`，按当前项目实际依赖覆盖训练、verify、sim、dashboard、AI review 和测试所需三方库。
- **README Upgrade**: 更新 README 版本说明、环境初始化、训练后二次筛选与 AI 评审命令示例。

### Fixed
- **AI Review Fallback**: Gemini / OpenAI 评审失败时，二次筛选流程不再整体中断；会自动回退为占位评审并保留量化结果与报告输出。

## [V5.4] - 2026-02-17

### Changed
- **Live Selection Pipeline Alignment**: `sim_runner` 在 `live` 模式下改为构建 strict-aligned 选股上下文（`SQLStrictLoader` + 65 交易日窗口），将实时行情与选股特征口径解耦。
- **Top-K Selection Consistency**: `live` 模式选股调用显式传入 `valid_mask`，对齐 strict replay 的可交易样本过滤规则，降低持仓偏移。

### Added
- **As-Of SQL State Hydration**: `SimulationRunner._hydrate_state_from_sql` 新增 `as_of_date`，可按回放日期截止加载历史持仓/交易/NAV 状态。
- **Date-Scoped SQL Reset**: `SQLStateStore` 新增 `reset_strategy_date(strategy_id, trade_date)`，支持只清理单日状态数据。

### Fixed
- **Single-Day Strict Replay Resume**: `run_sim` 在 `strict_replay + --date` 场景下不再全量重置状态，改为“仅清理当日记录 + 续接历史状态”后再调仓与记账；区间回放（`--start-date/--end-date`）逻辑保持不变。

## [V4.1.2] - 2026-02-05

### Added
- **SimFailShare Metric**: Now correctly calculates the redundancy rate of *good* factors (SimReject / MetricPass), decoupling it from structural failures.
- **V4.X Architecture Specs**: Archived implementation plans for V4.0, V4.1, V4.1.1, V4.1.2 in `doc/`.

### Fixed
- **SimPass Logic**: Explicitly includes `SIM_REPLACE` (better factors replacing old ones) as a success event.
- **Code Cleanup**: Removed deprecated `beta_is_locked` variables to reduce confusion.

## [V4.1] - 2026-02-04 (Low Risk Online)

### Key Features
- **Rolling Window Controller**: Entropy beta now reacts to `RollingHardPassRate` (10-step window) instead of volatile single-step rates, preventing erratic resets.
- **3-Level Success Criteria**: Introduced `HardPass` (Valid), `MetricPass` (Profitable), and `SimPass` (Unique) to precisely diagnose pipeline bottlenecks.
- **Dynamic Penalty Lock**: If `LowVar` failure persists (>70% for 50 steps), penalty multiplier increases (x1.15) to force exploration.

### Performance
- **Zero Struct Failure**: Achieved <0.2% error rate via Grammar-Guided Masking.
- **Diversity**: Maintains ~5% healthy redundancy (`SimReject`) without collapsing into mode repetition.

## [V4.0] - 2026-02-01 (Robustness Overhaul)

### Key Features
- **Hierarchy of Failure**: Implemented tiered penalties (`EXEC` -10, `STRUCT` -8, `LOWVAR` -6.5, `METRIC` -4~-6) to create a learnable gradient for RL.
- **Safe LRU Cache**: 100k capacity `OrderedDict` to eliminate redundant backtests within and across batches.
- **Observability 2.0**: Added console logs for `Gap`, `RStd`, and `TopFail` distribution.
