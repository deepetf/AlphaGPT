# AlphaGPT: Agentic Quantitative Mining System

**An industrial-grade symbolic regression framework for Alpha factor mining, powered by Reinforcement Learning.**

Current Version: **V5.6: Live Quote Override & Warmup Alignment (Current)**

当前版本run_sim CLI:

  # 0) 无参数启动（默认 live + SQL 状态 + QMT 行情）
  python strategy_manager/run_sim.py

  # 1) SQL 严格回放（单日）
  python strategy_manager/run_sim.py --mode strict_replay --date 2025-12-01 --state-backend sql --replay-source sql_eod

  # 2) SQL 严格回放（区间）
  python strategy_manager/run_sim.py --mode strict_replay --start-date 2025-01-01 --end-date 2025-12-31 --state-backend sql --replay-source sql_eod

  # 3) Live（单日，dummy 行情）
  python strategy_manager/run_sim.py --mode live --date 2025-12-01 --state-backend sql --live-quote-source dummy

  # 4) Live（单日，QMT 行情）
  python strategy_manager/run_sim.py --mode live --date 2025-12-01 --state-backend sql --live-quote-source qmt

  可选定时跑 live：

  python strategy_manager/run_sim.py --mode live --schedule --hour 14 --minute 50 --state-backend sql --live-quote-source qmt
---

## 🧾 Version History
> 维护约定：从 V5.4 起，每次新增版本条目时，需同步补充“主要功能更新对应的示例命令行（可直接复制运行）”。

### **V5.6: Live Quote Override & Warmup Alignment (Current)**
*在 V5.5 基础上补强 live 行情接入稳定性，并完成训练/验证预热窗口口径升级。*
- **Live No-Arg Default Profile**: `run_sim.py` 在无参数启动时默认使用 `--mode live --state-backend sql --live-quote-source qmt`，便于实盘日常调度。
- **QMT Tick Snapshot Upgrade**: `RealtimeDataProvider.get_realtime_quotes` 改为 `get_full_tick` 路径，统一输出 `trade_date/open/high/low/close/vol/amount` 并解析 `timetag` 日期。
- **Live OHLC SQL Override**: live 模式下新增 `merge_live_ohlc_into_cb_features`，按 `code` 用 QMT 的 `open/high/low/close`（有效正值）覆盖 SQL 当日行情，增强盘中执行一致性。
- **Take-Profit Null-Safety**: `SimulationRunner` 增加 `_safe_float` 容错，修复 `float(None)` 导致的 live 止盈路径中断；价格字典与 TP 判定统一使用安全数值转换。
- **Top-K Input Shape Hardening**: `_select_top_k` 增加 `factor_values/valid_mask/assets` 维度与长度一致性校验，避免静默截断引起持仓漂移。
- **Warmup-Aware Loader Pipeline**: `CBDataLoader` 支持 `start_date` 与 `warmup_days` 前置加载，特征计算后裁剪预热段并清理无效资产；`FeatureEngineer` 支持 `warmup_rows` 与 `LOG_MONEYNESS` 派生特征。
- **Strict Loader Optional Columns**: `SQLStrictLoader` 支持可选原始列缺失（用于派生特征依赖），缺失时跳过并提示，提升跨库表结构兼容性。
- **Verify Warmup Start Backfill**: `verify_strategy` 自动从验证起点向前回补加载窗口（约 200 自然日），与滚动特征和 warmup 口径对齐。
- **示例命令（无参数默认 live 启动）**: `python strategy_manager/run_sim.py`
- **示例命令（live 单日 + QMT 实时行情）**: `python strategy_manager/run_sim.py --mode live --date 2026-02-27 --state-backend sql --live-quote-source qmt`
- **示例命令（训练端指定数据起始日）**: `python -m model_core.engine --data-start-date 2023-01-01`
- **示例命令（verify 预热对齐）**: `python tests/verify_strategy.py --start 2025-01-01 --end 2026-02-13 --strategy-id king_v1`

### **V5.5: Verify Visualization & Multi-Strategy Combo**
*在 V5.4 基础上继续强化 verify 诊断能力，新增基准对齐可视化与多策略组合产物，便于直接评估“降回撤、提夏普”。*
- **Verify Cash-Aware Rebalance (Verify-only)**: 在 `verify_strategy` 内新增“现金约束买单裁剪”，先卖后买、按预算与 10 张粒度裁剪买单；不修改 `sim run` 调仓逻辑。
- **Cash Trim Audit Traceability**: `daily_trades` 新增 `requested/submitted` 买卖单统计、`trimmed/skipped` 计数与 `buy_trim_events` 事件明细，便于定位持仓数不足原因。
- **Benchmark Integration (index_jsl)**: 基准改为读取 `data/index.pq` 的 `index_jsl`（日涨跌幅），按交易日 `t -> t+1` 对齐到 verify 收益序列；收益图与汇总表同时展示策略/基准/超额。
- **Return & Drawdown Visualization Upgrade**: 总收益图新增基准累计收益与“策略-基准累计超额曲线”；汇总表补充并统一展示最大回撤等核心 KPI。
- **Year/Month Switch + Heatmap Labels**: 分期收益图支持“按年/按月”切换，年/月均展示策略 vs 基准，并单独展示超额柱状图；月度热力图在矩形内直接标注当月收益率。
- **Turnover Definition Alignment**: 换手率口径统一为“日换手 = max(去重买入数, 去重卖出数)/Top-K（不含 TP 卖出）”；汇总换手 = 总调仓数/(Top-K*交易天数)。
- **Multi-Strategy Combined Artifacts**: 当 verify 全策略运行后，自动生成组合产物（`daily_returns/daily_trades/daily_holdings/verification_report`）与可视化报告，口径与单策略一致。
- **Combo-Specific Summary Metrics**: 组合汇总表新增子策略两两收益相关系数、两两持仓重合度（按日 Jaccard 均值）、各子策略独立年化收益与 Sharpe。
- **示例命令（单策略 verify + 可视化）**: `python tests/verify_strategy.py --start 2025-01-01 --end 2026-02-13 --strategy-id king_v1`
- **示例命令（全启用策略 verify + 自动生成组合产物）**: `python tests/verify_strategy.py --start 2025-01-01 --end 2026-02-13`
- **示例命令（关闭 verify 现金裁剪，做口径对比）**: `python tests/verify_strategy.py --start 2025-01-01 --end 2026-02-13 --strategy-id momen_supply --verify-no-cash-aware`

### **V5.4: SimRun Live/Replay State Alignment**
*在 V5.3 基础上继续强化 live 与 strict replay 的持仓一致性，并修复单日 strict 回放状态续跑。*
- **Live Selection Strict-Alignment**: `live` 模式选股上下文改为通过 `SQLStrictLoader` 按 65 交易日窗口构建，复用 strict replay 的特征/资产/valid_mask 口径；实时行情仅用于成交与止盈定价。
- **Top-K Valid Mask Consistency**: `live` 模式选股显式透传 `valid_mask` 到 `_select_top_k`，避免因可交易样本筛选差异导致持仓漂移。
- **Single-Day Replay Incremental Resume**: `strict_replay + --date` 改为“仅清理当日 holdings/trades/nav + 按 as_of_date 回灌历史状态”，不再默认全量清空状态。
- **SQL State Store As-Of Loading**: `SQLStateStore.load_runtime_state` 新增 `as_of_date` 截止加载能力，并支持按策略+日期粒度重置（`reset_strategy_date`）。
- **Verify Warmup Alignment**: `verify_strategy` 在 `--start` 验证区间前自动回补 65 个交易日预热窗口，并在该窗口内重算特征与收益标签，和 `run_sim strict_replay` 口径对齐。
- **示例命令（单日 strict 回放续跑）**: `python strategy_manager/run_sim.py --mode strict_replay --date 2025-02-16 --state-backend sql --replay-source sql_eod`
- **示例命令（live 单日仿真，运行全部启用策略）**: `python strategy_manager/run_sim.py --mode live --date 2025-02-16 --state-backend sql --live-quote-source dummy`
- **示例命令（区间 strict 回放，原逻辑不变）**: `python strategy_manager/run_sim.py --mode strict_replay --start-date 2025-01-01 --end-date 2025-02-16 --state-backend sql --replay-source sql_eod`
- **示例命令（verify 预热窗对齐 strict replay）**: `python tests/verify_strategy.py --start 2025-02-10 --end 2025-02-16 --strategy-id king_v1`

### **V5.3: SimRun + Verify Config Alignment**
*在 V5.2 基础上完成 sim_run 严格回放链路强化与 verify_strategy 配置驱动对齐。*
- **Strict Replay Windowing**: `run_sim` 在 `strict_replay` 下按“回放区间 + 预热窗(65交易日)”加载 SQL，避免全历史初始化卡顿，并新增阶段耗时日志。
- **SQL Strict Loader Range**: `SQLStrictLoader` 支持 `start_date/end_date` 区间查询，strict 回放初始化可控、可观测。
- **Live/Replay Dataset Isolation**: `SQLStateStore` 新增 `dataset` 维度，`live/replay` 映射到独立 `nav/holdings/trades` 表，避免状态互相污染。
- **Runner Orchestration Upgrade**: `run_sim`/`multi_sim_runner` 支持统一 `mode`、`live_quote_source`、`replay_source` 覆盖与 strict 区间透传。
- **Verify Config-Driven**: `verify_strategy` 支持 `strategies_config.json + strategy_id`；`top_k/fee_rate/take_profit/initial_capital` 与配置一致。
- **Verify Multi-Strategy by Default**: 不指定 `--strategy-id` 时自动验证全部 `enabled` 策略，产物按 `strategy_id` 分文件输出，避免覆盖。
- **Verify Output Readability**: 修复验证报告模板乱码，产出统一 UTF-8 可读。

### **V5.2: Config-Driven SimRun (Current)**
*完成 sim_run 策略配置驱动改造，统一策略入口并简化依赖。*
- **Config-Only Entry**: `run_sim` 移除 `--multi`，统一通过 `strategies_config.json` 加载单/多策略。
- **Strategy ID Hardening**: `SimulationRunner` 移除 `default` 兜底，`strategy_id` 必须来自配置文件。
- **Schema Validation**: `strategy_config` 增加严格校验（`id` 唯一、参数只允许在 `params`、公式必填）。
- **Self-Contained Formula**: 支持策略内嵌 `formula`，`strategies_config.json` 可与外部公式文件解耦。
- **Runner Refactor**: `MultiSimRunner` 支持按 `strategy_id` 过滤，便于单策略定向运行与验收。

### **V5.1: SQL Live State & Replay QA**
*完成 SQL 状态持久化、回放可观测性与一致性验收工具。*
- **SQL State Backend**: `run_sim` 支持 `--state-backend sql`，将 `nav/holdings/trades` 写入 `sim_nav_history`、`sim_daily_holdings`、`sim_trade_history`。
- **Strategy Isolation**: 单策略固定 `strategy_id=default`，多策略使用 `strategies_config.json` 的 `id` 对应数据库分区。
- **Replay Robustness**: 修复区间回放状态重置与运行时异常；完善日志编码与关键日志可读性。
- **Execution Reliability**: 调仓新增“现金预算裁剪 + 排名优先买入”，避免候选 10 只但落仓 9 只的静默失败。
- **Diagnostics**: 新增 `tests/verify_sim_sql_replay.py`，自动检查 NAV 恒等式、持仓计数一致性、异常日和候选缺失。
- **Docs & Migration**: 补充 `simrun_live.md` 与建表脚本 `infra/migrations/20260214_create_simrun_live_tables.sql`。

### **V5.0: Sim-Verify Alignment**
*SQL EOD 回放、模拟盘执行链路与验证口径对齐。*
- **SQL EOD Strict Replay**: 新增 `SQLStrictLoader`，`run_sim` 支持 `--replay-strict --replay-source sql_eod`，并支持 `--start-date/--end-date` 连续区间回放。
- **Simulation Infrastructure**: 新增 `SimulationRunner`、`SimTrader`、`NavTracker`、`StrategyConfig`、`MultiSimRunner` 等模块，完善单策略/多策略模拟盘框架。
- **Take-Profit Alignment**: TP 规则统一为 `prev_close*(1+tp)`，跳空按 `open`、盘中按 `tp_trigger`；交易历史支持 `SELL-TP` 标记。
- **Verify Alignment Fix**: `verify_strategy` 修复为“交易后净值记当日收益”，并将 TP 与手续费纳入当日收益口径。
- **Rebalance Alignment**: 模拟盘复用 `CBRebalancer` 与 `Top-K + valid_mask` 约束，提升与回测/验证的一致性。
- **Latest Alignment Metrics**:
  - 2024 区间：`Corr≈0.999`，`MAE≈5e-4`
  - 2025 区间：`Corr≈0.999`，`MAE≈4e-4`

### **V4.2: Alpha Efficiency (Current)**
*止盈逻辑与评价体系优化。*
- **Vectorized Take-Profit**: 全面向量化止盈逻辑，大幅提升训练速度；支持开盘跳空与盘中止盈。
- **Simplified Buy-Back**: 锁定止盈当日收益，仅对仍在 `Top-K` 的标的计算额外买回成本，平衡收益与换手。
- **Strict Price Filter**: 引入价格有效性检查（`0 < Price < 10000`），彻底消除脏数据引起的收益率污染。
- **IC/IR Alignment**: 修正因子评价对齐口径，对无效样本返回 `None`，防止 RL 指标偏置。

### **V4.1: Engineering Hardening**
*工程加固与口径真实性：确保每一条 Alpha 都经得起推敲。*
- **Hierarchy of Failure**: 建立（`EXEC` < `STRUCT` < `LOWVAR` < `METRIC` < `SIM`）惩罚阶梯，提供清晰的强化学习梯度。
- **Grammar-Guided Decoding**: 利用 Action Masking 技术将无效语法生成率从 99% 降至 **0%**。
- **Rolling Window Controller**: 引入 10 步滑动窗口熵控制，并结合 **3-Level Success (Hard/Metric/Sim)** 口径，减少训练震荡。
- **SimPass 2.0**: 重新定义多样性成功标准，认可“优胜劣汰”的替换行为（`SIM_REPLACE`）。

### **V3.5: Efficiency**
*效率革命：解决相似性冗余。*
- **Safe LRU Cache**: 100k 上限确定性缓存，消除冗余回测。
- **Adaptive Entropy 1.0**: 初步引入自适应熵控制。

### **V3.4: Grammar-Guided Decoding**
*引入算子语法约束，彻底解决无效公式生成问题。*

### **V3.3: Long-Run Optimization**
*Optimized for long-duration training stability and diversity.*
- **Training Stability**: Adjusted default `Entropy Beta` and `Train Steps` (2000) to prevent premature convergence.
- **Documentation**: Added "Playbook-style" tuning comments in `default_config.yaml`.
- **Dynamic Config**: `TRAIN_STEPS` is now dynamically configurable via YAML.

### **V3.2: Dynamic Configuration**
*Refactored configuration architecture for flexibility and ease of use.*
- **Dynamic Loading**:
    - 将 `INPUT_FEATURES` 和 `RobustConfig` 移至外部 YAML 配置文件（`default_config.yaml`）。
    - 支持通过命令行参数 `--config` 加载自定义配置文件，无需修改代码。
- **CLI Support**: `engine.py` 支持 `--config` 参数。
- **Backward Compatibility**: `config.py` 使用 Metaclass 保持 API 兼容性，确保旧代码无缝运行。

### **V3.1: Diversity & Anti-Stacking**
*Implemented advanced mechanisms to prevent "Formula Stacking" and boost diversity.*
- **Local Density Check**:
    - 在 `Formula Validator` 中引入滑动窗口密度检查（`Window=6, MaxTS=3`）。
    - 有效打击利用 `TS_MEAN` 堆叠刷高 Sharpe 的行为，同时保护合法的多因子逻辑。
- **Enhanced Exploration**:
    - **Entropy Regularization**: 引入线性衰减的熵正则项（`Beta: 0.04 -> 0.005`），在训练初期强制模型探索未知领域。
    - **Diversity Pool**: 维护一个 Top-50 多样性池，基于 **Jaccard Similarity**（阈值 `0.8`）过滤同质化公式。
- **Outcome**: 成功挖掘出 `DBLOW`（双低）、`VOL_STK`（正股波动）、`PREM`（溢价率）等多种不同逻辑的高性能因子。

### **V3.0: CB Migration & Perfect Simulation**
*Completed migration from Crypto to Convertible Bonds and achieved pixel-perfect simulation alignment.*
- **System Migration**:
    - 全面适配可转债（CB）市场特性（T+0、涨跌停、整手交易、债券属性）。
    - 因子工厂与特征工程针对 CB 结构进行了重构。
- **Perfect Verification**:
    - **Event-Driven Simulation** 与 **Vector Backtest** 达到 >99% 相关性（Correlation）。
    - 修复持仓系统双轨制 Bug，彻底消除模拟误差。
    - 验证策略在 2023-2026 四年间的稳健表现（年化 30%~60%）。
- **Simulation Suite**:
    - 升级 `verify_strategy.py`，支持命令行参数配置（初始资金、时间区间）。
    - 增强交易日志，支持名称显示与详细一致性检查（Return MAE、Jaccard）。
- **Configuration**:
    - 费率下调至实盘水平（万五，`0.0005`），释放高频策略潜力。

### **V2.3: Formula Structure Validator**
*防止进化搜索产生“分数坍缩”的公式结构，并增强收益风险平衡。*
- **Hard Filters**: 直接拒绝已知有害序列（如 `SIGN -> LOG`），避免所有资产得分相同。
- **Soft Penalties**: 对可疑结构（如连续 `LOG`、`TS_*` 过多）进行扣分而非直接拒绝。
- **Documentation**: 新增 `docs/dangerous_structures.md` 记录所有规则。
- **Integration**: 在 `_worker_eval` 回测前进行验证，减少无效计算。
- **Fee Rate Management**: 交易费率统一由 `RobustConfig.FEE_RATE` 管理（`0.001`，千分之一）。
- **Return Reward**: 评分体系新增**年化收益率奖励**（`RET_W = 5.0`），鼓励挖掘高回报策略。

### **V2.2: Performance & Operators**
*修复训练过程中的性能衰减，并扩充算子库。*
- **CPU Thread Fix**: 强制单进程模式下 Worker 使用单线程（`torch.set_num_threads(1)`），解决 CPU Oversubscription 问题。
- **I/O Optimization**: 引入 `MIN_SCORE_IMPROVEMENT` 阈值，减少无效 I/O。
- **New Operator**: 新增 `TS_BIAS5`（5 日乖离率）算子。

### **V2.1: Robustness Enhanced**
*专注于挖掘“稳健、可实盘”的因子，而非单纯追求高夏普。*
- **Split Validation**: 强制进行 **Train/Test 分段验证**；若 Valid 段表现差，直接淘汰。
- **Rolling Stability**: 引入 **Mean - k*Std** 评分机制，惩罚收益曲线剧烈波动的因子。
- **Drawdown Control**: 显式惩罚最大回撤（MDD）。
- **Tradability Constraints**:
    - **Active Ratio**: 剔除因停牌或数据缺失导致无法满仓的因子。
    - **Valid Days**: 剔除样本量不足的偶然高分因子。
- **Composite Score**: 综合 `(Train+Val) * Stability - MDD` 的多维评分体系。

### **V1.0: Foundation (No Lookahead)**
*专注于消除“未来函数”和“数据泄露”。*
- **Strict Causality**: 所有算子（`TS_DELAY`、`TS_MEAN`）经严格审计，杜绝 `torch.roll` 造成的循环泄露。
- **Robust Normalization**: 使用 `Rolling(60)` 做特征标准化，而非全局 Z-Score，避免全样本偏差。
- **Aligned Data Pipeline**: 统一 `CBDataLoader`，确保训练、验证、回测使用一致的数据流（`[Time, Assets]`）。

---

## 📊 Key Metrics Explained

| Metric | Definition | Good | Bad |
|--------|------------|------|-----|
| **Split Sharpe** | 训练集与验证集的夏普比率对比 | Train 与 Val 接近，且均 > 1.5 | Val < 0 或 Train >> Val（过拟合） |
| **Stability** | 滚动夏普均值 - 1.5 × 标准差 | > 0.5 | < 0（不稳定） |
| **Max Drawdown** | 历史最大回撤 | < 20% | > 40% |
| **Active Ratio** | 实际持仓数 / 目标 Top-K | > 90% | < 50%（不可交易） |

---

## 🧠 Core Modules

### 1. AlphaGPT Model
- 轻量级 Transformer Decoder。
- 学习算子语法（`ADD`、`SUB`、`TS_DELAY`...）与特征组合。
- 通过 RL（Policy Gradient）优化，根据回测 Reward 调整生成概率。

### 2. StackVM (Vectorized)
- 高性能 PyTorch 向量化栈虚拟机。
- 支持时序算子（`TS_*`）、截面算子（`CS_*`）和逻辑算子（`IF_POS`...）。
- **零未来函数设计**。

### 3. CBBacktest (Robust)
- Top-K 轮动策略回测器。
- 支持 `Transaction Fee` 与 `Turnover` 惩罚。
- 内置 V2.1 稳健性评估指标（`Split Sharpe`、`Stability`、`Active Ratio`）。

---

## 🔧 Usage

### 1. Training (Mining)
启动多进程挖掘：
```bash
python -m model_core.engine
```
*自动保存表现最好的 “Kings” 到 `model_core/verified_trades/` 和 `best_cb_formula.json`。*

### 2. Verification
验证特定因子（如 King #8）的稳健性：
```bash
python verify_kings.py --king 8
```
输出示例：
```text
✅ Backtest Result:
   Composite Score: 10.76
   Sharpe (Train/Val): 1.27 / 1.98  <-- 关键指标
   Max Drawdown: 14.9%
   Stability: -0.86
   Active Ratio: 100.0%
```

### 3. Real-time Simulation
模拟实盘选股（使用最近 70 天数据）：
```bash
python verify_king8_realtime.py
```

---

## ⚙️ Configuration

Configuration is now managed via `model_core/default_config.yaml`.

### 1. Default Configuration
You can directly edit `model_core/default_config.yaml` to adjust parameters:
- **Input Features**: List of factors used by the model.
- **Robustness**: Split date, rolling window, stability thresholds.
- **Scoring**: Weights for Sharpe, Stability, Returns, Drawdown.

### 2. Custom Configuration
Create a custom YAML file (e.g., `my_config.yaml`) and run:

```bash
python -m model_core.engine --config my_config.yaml
```

Example YAML override:
```yaml
robust_config:
  top_k: 20
  fee_rate: 0.0002
  min_sharpe_val: 0.5
```

---

**Disclaimer**: Quantitative trading involves significant risks. This code is for research purposes only.


