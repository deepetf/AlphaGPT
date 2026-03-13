# AlphaGPT 命令文档（V6.01）

本文档基于当前 `README.md`、`model_core/engine.py`、`tests/verify_strategy.py`、`strategy_manager/run_sim.py` 与相关配置加载逻辑整理，覆盖当前版本最常用的三类命令：

- 训练：`python -m model_core.engine`
- 验证：`python tests/verify_strategy.py`
- 模拟/严格回放：`python strategy_manager/run_sim.py`

V6.01 补充说明：

- `model_core/select_top_factors.py` 当前支持 `similarity_mode: hybrid`
- hybrid 去重同时参考公式 Jaccard、平均持仓 Jaccard、日收益相关性
- `factor_ai_review` 报告会单独输出 `Similarity rejected` 计数

## 1. 适用范围与命令入口

### 1.1 训练

用于符号回归因子挖掘、训练 AlphaGPT 模型，并输出当前最佳公式与模型权重。

命令入口：

```bash
python -m model_core.engine
```

主代码位置：

- `model_core/engine.py`
- `model_core/default_config.yaml`

### 1.2 VERIFY

用于对“事件驱动模拟”和“向量化回测”进行口径对齐验证，支持单策略与多策略验证。

命令入口：

```bash
python tests/verify_strategy.py
```

主代码位置：

- `tests/verify_strategy.py`
- `strategy_manager/strategies_config.json`
- `model_core/default_config.yaml`

### 1.3 SIM

用于模拟盘日常运行，支持：

- `live`：实时/准实时模拟
- `strict_replay`：基于 SQL EOD 数据的严格回放

命令入口：

```bash
python strategy_manager/run_sim.py
```

主代码位置：

- `strategy_manager/run_sim.py`
- `strategy_manager/strategies_config.json`
- `model_core/default_config.yaml`

## 2. 配置文件总览

### 2.1 模型配置（YAML）

所有训练、VERIFY、SIM 都会先加载 `model_core/default_config.yaml`，如果通过 `--config` 传入自定义 YAML，则会在默认配置基础上做深度合并覆盖。

当前常见模型配置文件：

- `model_core/default_config.yaml`：默认基线配置
- `model_core/config_batch1024.yaml`：大 batch 训练覆盖模板
- `model_core/config_slow_cs_replace.yaml`：slow feature replace 主实验配置
- `model_core/config_slow_cs_append.yaml`：slow feature append 对照配置
- `model_core/test_config.yaml`：测试用途配置

配置加载规则：

- 未传 `--config`：仅加载 `model_core/default_config.yaml`
- 传入 `--config xxx.yaml`：先加载默认配置，再用 `xxx.yaml` 覆盖
- `--config` 可用于训练、VERIFY、SIM，确保三者特征口径一致

### 2.2 策略配置（JSON）

VERIFY 与 SIM 使用策略 JSON 配置：

- `strategy_manager/strategies_config.json`：当前默认策略配置
- `strategy_manager/strategies_config.example.json`：示例模板

策略 JSON 结构包含三部分：

- `global`：全局设置，如数据源、日志级别
- `defaults`：策略默认参数
- `strategies`：策略列表

当前 `params` 支持字段：

- `initial_capital`
- `top_k`
- `take_profit_ratio`
- `fee_rate`
- `replay_strict`
- `replay_source`
- `state_backend`

每个策略必须提供以下其一：

- `formula`
- `formula_path`

## 3. 训练命令：`python -m model_core.engine`

### 3.1 功能说明

训练命令用于：

- 加载可转债历史数据与特征
- 让 AlphaGPT 采样公式
- 对候选公式执行向量化回测与稳健性评估
- 输出最佳公式、历史 king 公式与模型权重

主要输出：

- `model_core/best_cb_formula.json`
- `model_core/alphagpt_cb.pt`
- `model_core/king_trades/king_*.json`
- `artifacts/runs/<run_id>/manifest.json`
- `artifacts/runs/<run_id>/resolved_model_config.yaml`
- `artifacts/runs/<run_id>/train/best_cb_formula.json`
- `artifacts/runs/<run_id>/train/alphagpt_cb.pt`
- `artifacts/runs/<run_id>/train/king_trades/king_*.json`

### 3.2 命令格式

```bash
python -m model_core.engine [--config CONFIG_PATH] [--data-start-date YYYY-MM-DD] [--run-id RUN_ID] [--artifacts-root PATH]
```

### 3.3 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--config`, `-c` | `str` | `None` | 模型 YAML 配置路径；未传时使用 `model_core/default_config.yaml` |
| `--data-start-date` | `str` | `None` | 数据加载起始日，格式 `YYYY-MM-DD`；未传时走代码默认起点 |
| `--run-id` | `str` | `None` | 可选训练运行 ID；未传时自动生成并写入 `artifacts/runs/<run_id>/` |
| `--artifacts-root` | `str` | `None` | 可选训练产物根目录；默认写到 `artifacts/runs` |

### 3.4 对应配置文件

训练只使用模型配置 YAML，不使用策略 JSON。

重点配置文件：

- `model_core/default_config.yaml`
- `model_core/config_batch1024.yaml`
- `model_core/config_slow_cs_replace.yaml`
- `model_core/config_slow_cs_append.yaml`

重点配置项：

- 顶层训练参数：
  - `train_steps`
  - `batch_size`
  - `lr`
  - `weight_decay`
  - `grad_clip_norm`
  - `warmup_days`
  - `input_features`
  - `feature_normalization_overrides`
- `robust_config`：
  - `train_test_split_date`
  - `rolling_window`
  - `stability_k`
  - `min_sharpe_val`
  - `min_active_ratio`
  - `min_valid_days`
  - `min_valid_count`
  - `top_k`
  - `fee_rate`
  - `take_profit`
  - `signal_clean_enabled`
  - `signal_winsor_q`
  - `signal_clip`
  - `signal_rank_output`

### 3.5 典型用法

### 3.5.1 使用默认配置训练

```bash
python -m model_core.engine
```

适用场景：

- 跑当前默认基线实验
- 快速确认训练链路是否正常

### 3.5.2 显式使用默认配置训练

```bash
python -m model_core.engine --config model_core/default_config.yaml
```

适用场景：

- 明确记录实验配置来源
- 便于日志与实验复现

### 3.5.3 使用大 batch 配置训练

```bash
python -m model_core.engine --config model_core/config_batch1024.yaml
```

适用场景：

- 候选公式采样量更大
- 希望 reward/梯度更平滑

### 3.5.4 指定训练起始日

```bash
python -m model_core.engine --config model_core/config_batch1024.yaml --data-start-date 2022-08-01
```

适用场景：

- 复现实验窗口
- 控制样本起点

### 3.5.5 使用 slow feature replace 配置训练

```bash
python -m model_core.engine --config model_core/config_slow_cs_replace.yaml
```

适用场景：

- 训练 slow feature 主实验版本
- 让训练口径与后续 VERIFY/SIM 保持一致

### 3.5.6 指定 run_id 与产物目录

```bash
python -m model_core.engine --config model_core/config_slow_cs_replace.yaml --run-id slow_replace_exp01 --artifacts-root artifacts/runs
```

适用场景：

- 固定本次训练的实验编号
- 将训练产物标准化落到 run 目录，便于后续 straight-through 流程接入

### 3.5.7 slow replace 主实验（偏稳健基线）

```bash
python -m model_core.engine --config model_core/config_slow_cs_replace.yaml --run-id slow_replace_v601
```

适用场景：

- 使用当前仓库已跟踪的 slow feature replace 配置做主实验
- 与后续 `select / bundle / verify / sim` 保持特征口径一致
- 配合 `artifacts/runs/<run_id>/` 产物目录做 straight-through 续接

## 4. VERIFY 命令：`python tests/verify_strategy.py`

### 4.1 功能说明

VERIFY 用于验证以下口径是否一致：

- 事件驱动模拟路径
- 向量化回测路径
- Top-K 选择、止盈、手续费、净值计算

支持两种工作模式：

- 配置驱动模式：从 `strategy_manager/strategies_config.json` 读取策略
- legacy king 模式：从 `model_core/best_cb_formula.json` 读取历史 king 公式

主要输出目录：

- `tests/artifacts/`

常见输出文件：

- `tests/artifacts/daily_returns_*.csv`
- `tests/artifacts/daily_trades_*.json`
- `tests/artifacts/daily_holdings_*.json`
- `tests/artifacts/verification_report_*.md`

多策略模式下还会生成组合后的聚合产物。

### 4.2 命令格式

```bash
python tests/verify_strategy.py \
  [--start YYYY-MM-DD] \
  [--end YYYY-MM-DD] \
  [--strategies-config PATH] \
  [--config PATH] \
  [--bundle PATH] \
  [--strategy-id STRATEGY_ID] \
  [--king STEP] \
  [--top-k N] \
  [--fee-rate FLOAT] \
  [--initial-cash FLOAT] \
  [--take-profit FLOAT] \
  [--verify-cash-aware | --verify-no-cash-aware]
```

### 4.3 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--start` | `str` | `2024-01-01` | 验证起始日 |
| `--end` | `str` | `2024-12-31` | 验证结束日 |
| `--strategies-config` | `str` | `None` | 策略 JSON 路径；未传时使用 `strategy_manager/strategies_config.json` |
| `--config` | `str` | `None` | 模型 YAML 路径；未传时使用 `model_core/default_config.yaml` |
| `--bundle` | `str` | `None` | 由 `workflow.bundle_builder` 生成的策略 bundle；设置后 VERIFY 按 bundle 单策略运行 |
| `--strategy-id` | `str` | `None` | 只验证某一个策略；未传时验证配置中全部启用策略 |
| `--king` | `int` | `None` | legacy king 模式的 step id；不能与 `--strategy-id` 同时使用 |
| `--top-k` | `int` | `None` | 覆盖策略中的 `top_k` |
| `--fee-rate` | `float` | `None` | 覆盖单边费率 |
| `--initial-cash` | `float` | `None` | 覆盖初始资金；默认读取策略配置里的 `initial_capital` |
| `--take-profit` | `float` | `None` | 覆盖止盈比例，如 `0.06` 表示 6% |
| `--verify-cash-aware` | flag | 开启 | 启用按现金可用性裁剪买入 |
| `--verify-no-cash-aware` | flag | 关闭上项 | 关闭现金约束裁剪 |

### 4.4 配置文件与参数来源

VERIFY 同时依赖两类配置：

### 4.4.1 模型配置 YAML

来源：

- `--config`
- 默认：`model_core/default_config.yaml`

用途：

- 构建特征张量
- 控制 `input_features`
- 控制归一化与信号清洗
- 保证 VERIFY 与训练/SIM 使用同一特征语义

若传入 `--bundle`：

- 未显式传 `--config` 时，VERIFY 会自动读取 bundle 内记录的 `model_config_path`
- 若同时显式传 `--config`，则以 CLI 为准，并记录日志提示

### 4.4.2 策略配置 JSON

来源：

- `--strategies-config`
- 默认：`strategy_manager/strategies_config.json`

用途：

- 提供策略公式
- 提供 `top_k`
- 提供 `fee_rate`
- 提供 `take_profit_ratio`
- 提供 `initial_capital`

参数优先级：

1. CLI 显式传参
2. `strategies_config.json` 中该策略的 `params`
3. `defaults`
4. 代码默认值

### 4.5 典型用法

### 4.5.1 验证全部启用策略

```bash
python tests/verify_strategy.py --start 2025-01-01 --end 2025-12-31
```

适用场景：

- 对默认策略组合做全量对齐验证
- 自动生成多策略汇总产物

### 4.5.2 验证单个策略

```bash
python tests/verify_strategy.py --start 2025-01-01 --end 2025-12-31 --strategy-id king_mom
```

适用场景：

- 单独检查某一个策略的事件驱动与向量回测一致性

### 4.5.3 使用 slow feature replace 配置验证

```bash
python tests/verify_strategy.py --start 2025-01-01 --end 2025-12-31 --strategies-config strategy_manager/strategies_config.json --strategy-id king_slow --config model_core/config_slow_cs_replace.yaml
```

适用场景：

- 验证 slow feature 实验策略
- 避免 VERIFY 侧因为特征未接线出现 `UNKNOWN_TOKEN`

### 4.5.4 覆盖交易参数验证

```bash
python tests/verify_strategy.py --start 2025-01-01 --end 2025-03-31 --strategy-id king_mom --top-k 10 --take-profit 0.05 --fee-rate 0.001
```

适用场景：

- 快速测试参数敏感性
- 做局部口径复核

### 4.5.5 legacy king 模式验证

```bash
python tests/verify_strategy.py --start 2025-01-01 --end 2025-12-31 --king 9
```

适用场景：

- 验证 `model_core/best_cb_formula.json` 中某个历史 king 的表现

### 4.5.6 直接验证 strategy bundle

```bash
python tests/verify_strategy.py --bundle artifacts/runs/slow_replace_exp01/bundle/strategy_bundle.json --start 2025-01-01 --end 2025-12-31
```

适用场景：

- 直接验证 Phase 2 产出的标准策略 bundle
- 自动使用 bundle 内的模型配置、公式和默认交易参数

### 4.5.7 bundle + 参数覆盖验证

```bash
python tests/verify_strategy.py --bundle artifacts/runs/slow_replace_exp01/bundle/strategy_bundle.json --start 2025-01-01 --end 2025-03-31 --top-k 10 --take-profit 0.05
```

适用场景：

- 以 bundle 为基础做局部参数敏感性验证
- 不再依赖 `strategies_config.json`

## 5. SIM 命令：`python strategy_manager/run_sim.py`

### 5.1 功能说明

SIM 用于日常策略运行与回放，支持：

- `live`：模拟实时运行
- `strict_replay`：基于 SQL EOD 的严格回放
- 单日运行
- 区间连续回放
- 定时调度运行
- 单策略或多策略运行

当前代码中的无参数默认行为：

```bash
python strategy_manager/run_sim.py
```

等价于：

```bash
python strategy_manager/run_sim.py --mode live --state-backend sql --live-quote-source dummy
```

这是当前版本的最新默认口径。

### 5.2 命令格式

```bash
python strategy_manager/run_sim.py \
  [--mode {live,strict_replay}] \
  [--schedule] \
  [--strategies-config PATH] \
  [--config PATH] \
  [--bundle PATH] \
  [--strategy-id STRATEGY_ID] \
  [--date YYYY-MM-DD] \
  [--start-date YYYY-MM-DD] \
  [--end-date YYYY-MM-DD] \
  [--state-backend {sql,json}] \
  [--live-quote-source {dummy,qmt}] \
  [--replay-strict] \
  [--replay-source {sql_eod,parquet}] \
  [--hour H] \
  [--minute M]
```

### 5.3 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--mode` | `str` | `strict_replay` | 运行模式；可选 `live`、`strict_replay` |
| `--schedule` | flag | 关闭 | 启用内置定时调度模式 |
| `--strategies-config` | `str` | `None` | 策略 JSON 路径；默认 `strategy_manager/strategies_config.json` |
| `--config` | `str` | `None` | 模型 YAML 路径；默认 `model_core/default_config.yaml` |
| `--bundle` | `str` | `None` | 由 `workflow.bundle_builder` 生成的策略 bundle；设置后 SIM 按 bundle 单策略运行 |
| `--strategy-id` | `str` | `None` | 只运行指定策略；未传则运行全部启用策略 |
| `--date` | `str` | `None` | 单日运行日期；未传时单日模式默认取当天 |
| `--start-date` | `str` | `None` | 区间回放起始日 |
| `--end-date` | `str` | `None` | 区间回放结束日 |
| `--state-backend` | `str` | `None` | 覆盖策略状态存储后端，可选 `sql`、`json` |
| `--live-quote-source` | `str` | `dummy` | `live` 模式行情源，可选 `dummy`、`qmt` |
| `--replay-strict` | flag | 关闭 | 向后兼容别名；开启后强制 `mode=strict_replay` |
| `--replay-source` | `str` | `None` | 覆盖回放源，可选 `sql_eod`、`parquet`；但当前 `strict_replay` 只允许 `sql_eod` |
| `--hour` | `int` | `14` | 定时任务小时 |
| `--minute` | `int` | `50` | 定时任务分钟 |

### 5.4 模式约束

### 5.4.1 单日运行

当未提供 `--start-date/--end-date` 时，按单日模式运行。

- 使用 `--date`：跑指定交易日
- 不传 `--date`：默认取当前日期

### 5.4.2 区间运行

当提供 `--start-date` 或 `--end-date` 时：

- 必须同时提供 `--start-date` 和 `--end-date`
- 仅支持 `--mode strict_replay`

### 5.4.3 strict replay 约束

当前代码中：

- `strict_replay` 仅允许 `replay_source=sql_eod`
- 若传入 `--replay-source parquet` 并使用 `strict_replay`，会直接报错

### 5.4.4 schedule 模式

启用 `--schedule` 后，程序会循环等待到指定时间，并按单日模式每日执行一次。

### 5.5 配置文件与参数来源

SIM 同时依赖模型配置 YAML 和策略配置 JSON。

### 5.5.1 模型配置 YAML

来源：

- `--config`
- 默认：`model_core/default_config.yaml`

用途：

- 特征工程
- 输入特征集合
- 归一化与 masked CS 相关控制
- 信号清洗与 Top-K 选择口径

若传入 `--bundle`：

- 未显式传 `--config` 时，SIM 会自动读取 bundle 内记录的 `model_config_path`
- 若同时显式传 `--config`，则以 CLI 为准，并记录日志提示

### 5.5.2 策略配置 JSON

来源：

- `--strategies-config`
- 默认：`strategy_manager/strategies_config.json`

用途：

- 策略列表与启停
- 每个策略的公式
- 每个策略的：
  - `initial_capital`
  - `top_k`
  - `take_profit_ratio`
  - `fee_rate`
  - `replay_strict`
  - `replay_source`
  - `state_backend`

SIM 的状态与落盘通常位于：

- `execution/portfolio/<strategy_id>/holdings.json`
- `execution/portfolio/<strategy_id>/nav_history.json`
- `execution/portfolio/<strategy_id>/trade_history.json`
- `execution/portfolio/<strategy_id>/candidates_history.json`

若使用 SQL 状态后端，还会同步写入 SQL 状态表。

若传入 `--bundle`：

- `SIM` 会优先使用 bundle 同目录下的 `generated_strategy_config.json`
- `--strategies-config` 会被忽略，并记录日志提示

### 5.6 典型用法

### 5.6.1 无参数启动

```bash
python strategy_manager/run_sim.py
```

适用场景：

- 默认 live 日常运行
- SQL 状态 + dummy 行情

### 5.6.2 strict replay 单日

```bash
python strategy_manager/run_sim.py --mode strict_replay --date 2025-12-01 --state-backend sql --replay-source sql_eod
```

适用场景：

- 回放某一交易日
- 复核当日选股、成交、止盈与净值

### 5.6.3 strict replay 区间

```bash
python strategy_manager/run_sim.py --mode strict_replay --start-date 2025-01-01 --end-date 2025-12-31 --state-backend sql --replay-source sql_eod
```

适用场景：

- 连续区间回放
- 检查持仓延续、净值轨迹与状态累积

### 5.6.4 live 单日 + dummy 行情

```bash
python strategy_manager/run_sim.py --mode live --date 2025-12-01 --state-backend sql --live-quote-source dummy
```

适用场景：

- 本地环境没有 QMT
- 快速联调 live 流程

### 5.6.5 live 单日 + QMT 行情

```bash
python strategy_manager/run_sim.py --mode live --date 2025-12-01 --state-backend sql --live-quote-source qmt
```

适用场景：

- 接入 QMT 实时行情
- 更接近实盘运行环境

### 5.6.6 使用 slow feature replace 配置做 strict replay

```bash
python strategy_manager/run_sim.py --mode strict_replay --date 2025-12-01 --state-backend sql --replay-source sql_eod --config model_core/config_slow_cs_replace.yaml
```

适用场景：

- SIM 口径与 slow feature 训练配置保持一致

### 5.6.7 只运行单个策略

```bash
python strategy_manager/run_sim.py --mode strict_replay --date 2025-12-01 --strategy-id king_mom --state-backend sql --replay-source sql_eod
```

适用场景：

- 排查单策略问题
- 降低调试复杂度

### 5.6.8 定时运行 live

```bash
python strategy_manager/run_sim.py --mode live --schedule --hour 14 --minute 50 --state-backend sql --live-quote-source qmt
```

适用场景：

- 每日固定时间触发模拟盘运行

### 5.6.9 直接运行 bundle（strict replay）

```bash
python strategy_manager/run_sim.py --bundle artifacts/runs/slow_replace_exp01/bundle/strategy_bundle.json --mode strict_replay --date 2025-12-01
```

适用场景：

- 直接消费 Phase 2 产出的标准策略 bundle
- 自动使用 bundle 内的模型配置和生成策略配置

### 5.6.10 直接运行 bundle（live）

```bash
python strategy_manager/run_sim.py --bundle artifacts/runs/slow_replace_exp01/bundle/strategy_bundle.json --mode live --date 2025-12-01 --live-quote-source dummy
```

适用场景：

- 不再依赖手工整理 `strategies_config.json`
- 以 bundle 为单策略运行单元进行 live 仿真

## 6. 推荐命令组合

### 6.1 默认基线流程

```bash
python -m model_core.engine --config model_core/default_config.yaml
python tests/verify_strategy.py --start 2025-01-01 --end 2025-12-31 --strategies-config strategy_manager/strategies_config.json
python strategy_manager/run_sim.py --mode strict_replay --date 2025-12-01 --strategies-config strategy_manager/strategies_config.json --state-backend sql --replay-source sql_eod
```

适用场景：

- 默认特征集
- 默认策略集
- 默认训练/验证/模拟闭环

### 6.2 slow feature replace 闭环

```bash
python -m model_core.engine --config model_core/config_slow_cs_replace.yaml
python tests/verify_strategy.py --start 2025-01-01 --end 2025-12-31 --strategies-config strategy_manager/strategies_config.json --strategy-id king_slow --config model_core/config_slow_cs_replace.yaml
python strategy_manager/run_sim.py --mode strict_replay --date 2025-12-01 --strategies-config strategy_manager/strategies_config.json --strategy-id king_slow --state-backend sql --replay-source sql_eod --config model_core/config_slow_cs_replace.yaml
```

适用场景：

- 保证训练、VERIFY、SIM 使用同一组 slow feature 口径

### 6.3 Phase 1 + Phase 2 straight-through 闭环

```bash
python -m model_core.engine --config model_core/config_slow_cs_replace.yaml --run-id slow_replace_exp01
python -m model_core.select_top_factors --input artifacts/runs/slow_replace_exp01/train/best_cb_formula.json --output artifacts/runs/slow_replace_exp01/selection/top3_factors.json --report-output artifacts/runs/slow_replace_exp01/selection/top3_factors_report.md --config artifacts/runs/slow_replace_exp01/resolved_model_config.yaml
python -m workflow.bundle_builder --manifest artifacts/runs/slow_replace_exp01/manifest.json --selection-output artifacts/runs/slow_replace_exp01/selection/top3_factors.json --source selected --candidate-rank 1 --strategy-id slow_replace_top1
```

适用场景：

- 使用标准 run 目录承接训练、筛选、打包
- 减少人工从 `model_core/` 拷贝产物到策略配置的步骤

### 6.4 仅训练 best 直接打包闭环

```bash
python -m model_core.engine --config model_core/config_slow_cs_replace.yaml --run-id slow_replace_exp01
python -m workflow.bundle_builder --manifest artifacts/runs/slow_replace_exp01/manifest.json --source best --strategy-id slow_replace_best
```

适用场景：

- 训练完成后先不做二次筛选
- 直接把训练 best 结果转成可发布 bundle

## 7. 常见注意事项

- `VERIFY` 与 `SIM` 若涉及 slow feature、新增特征或替换特征集合，务必显式传入与训练一致的 `--config`
- `strict_replay` 当前应优先使用 `--replay-source sql_eod`
- `VERIFY` 中 `--king` 不能与 `--strategy-id` 同时使用
- `SIM` 的区间回放必须同时传 `--start-date` 和 `--end-date`
- `SIM` 的区间回放仅支持 `--mode strict_replay`
- `run_sim.py` 当前最新无参默认口径是 `live + sql + dummy`，不要参考 README 历史版本里的旧默认值
- 训练输出最佳公式后，若要投入 VERIFY 或 SIM，优先将公式整理进 `strategy_manager/strategies_config.json`，避免 legacy 路径和当前策略配置脱节
- 当前 `VERIFY` 已支持 `--bundle`
- 当前 `SIM` 已支持 `--bundle`
- 当前 `model_core.select_top_factors` 仍需要显式指定 `--output`，推荐直接输出到 `artifacts/runs/<run_id>/selection/`
- 如果你希望 straight-through 链路最顺，训练阶段建议总是显式传 `--run-id`

## 8. 当前建议优先使用的配置文件

### 8.1 常规基线实验

- 模型配置：`model_core/default_config.yaml`
- 策略配置：`strategy_manager/strategies_config.json`

### 8.2 大 batch 训练实验

- 模型配置：`model_core/config_batch1024.yaml`

### 8.3 slow feature 主实验

- 模型配置：`model_core/config_slow_cs_replace.yaml`
- 策略配置：`strategy_manager/strategies_config.json`

### 8.4 slow feature 对照实验

- 模型配置：`model_core/config_slow_cs_append.yaml`

## 9. Straight-Through 辅助命令

### 9.0 当前 straight-through 状态

当前已落地：

- Phase 1：训练 run 目录、`manifest.json`、配置快照
- Phase 2：`strategy_bundle.json`、`formula_top1.json`、`generated_strategy_config.json`
- Phase 3：`tests/verify_strategy.py --bundle`
- Phase 4（部分）：`strategy_manager/run_sim.py --bundle`
- 统一入口：`python -m workflow.pipeline`
- `e2e` 运行状态落盘：`pipeline_status.json`
- `e2e` 断点续跑：`--resume`
- `e2e` 指定阶段启动：`--from-stage`

当前尚未落地：

- 更细的失败重试策略与阶段级强制重跑

因此当前推荐使用方式是：

1. 用 `python -m model_core.engine --run-id ...` 生成标准 run 目录
2. 用 `python -m model_core.select_top_factors` 将筛选结果输出到该 run 的 `selection/`
3. 用 `python -m workflow.bundle_builder` 生成标准 bundle
4. 用 `python tests/verify_strategy.py --bundle ...` 直接完成验证
5. 用 `python strategy_manager/run_sim.py --bundle ...` 直接完成模拟

补充说明：

- `workflow.pipeline e2e` 现在会在 run 目录中写入：
  - `artifacts/runs/<run_id>/pipeline_status.json`
- 用于记录各阶段状态：
  - `train`
  - `select`
  - `bundle`
  - `verify`
  - `sim`
- 可配合：
  - `--resume`
  - `--from-stage`
  做断点续跑

### 9.0.1 统一入口：`python -m workflow.pipeline`

当前已支持以下子命令：

- `train`
- `select`
- `bundle`
- `verify`
- `sim`
- `e2e`

设计目标：

- 复用现有 train/select/bundle/verify/sim 脚本
- 通过统一入口减少人工串命令
- 默认围绕 `run_id` 和 `artifacts/runs/<run_id>/` 工作

推荐先查看帮助：

```bash
python -m workflow.pipeline --help
python -m workflow.pipeline e2e --help
```

其中 `e2e` 额外支持：

- `--resume`
  - 跳过 `pipeline_status.json` 中已完成的阶段
- `--from-stage {train,select,bundle,verify,sim}`
  - 从指定阶段开始执行

### 9.1 候选筛选：`python -m model_core.select_top_factors`

建议在 straight-through 流程中，将筛选输出直接写入对应 run 目录。

推荐命令格式：

```bash
python -m model_core.select_top_factors --input artifacts/runs/<run_id>/train/best_cb_formula.json --output artifacts/runs/<run_id>/selection/top3_factors.json --report-output artifacts/runs/<run_id>/selection/top3_factors_report.md --config artifacts/runs/<run_id>/resolved_model_config.yaml
```

关键说明：

- `--input` 推荐直接使用训练 run 目录中的 `train/best_cb_formula.json`
- `--config` 推荐直接使用训练 run 目录中的 `resolved_model_config.yaml`
- `--output` 与 `--report-output` 推荐直接落到 `artifacts/runs/<run_id>/selection/`
- `top_factor_config.yaml` 当前支持 `similarity_mode: hybrid`（公式 Jaccard + 持仓 Jaccard + 日收益相关性）
- hybrid 关键阈值：`jaccard_threshold`、`holding_jaccard_threshold`、`return_corr_threshold`

典型用法：

#### 9.1.1 在 run 目录内生成筛选结果

```bash
python -m model_core.select_top_factors --input artifacts/runs/slow_replace_exp01/train/best_cb_formula.json --output artifacts/runs/slow_replace_exp01/selection/top3_factors.json --report-output artifacts/runs/slow_replace_exp01/selection/top3_factors_report.md --config artifacts/runs/slow_replace_exp01/resolved_model_config.yaml
```

适用场景：

- 把训练结果完整沉淀到单个 run 目录
- 为后续 `workflow.bundle_builder` 提供标准输入

### 9.2 策略打包：`python -m workflow.bundle_builder`

用于把训练产物或筛选产物打包成标准策略 bundle，减少人工整理 `strategies_config.json`。

主要输出：

- `artifacts/runs/<run_id>/bundle/strategy_bundle.json`
- `artifacts/runs/<run_id>/bundle/formula_top1.json`
- `artifacts/runs/<run_id>/bundle/generated_strategy_config.json`

命令格式：

```bash
python -m workflow.bundle_builder --manifest PATH [--selection-output PATH] [--source {selected,best}] [--candidate-rank N] [--strategy-id ID]
```

参数说明：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--manifest` | `str` | 必填 | 训练阶段产出的 `manifest.json` |
| `--selection-output` | `str` | `None` | 候选筛选结果；默认尝试从 `run_dir/selection/` 自动发现 |
| `--source` | `str` | 自动判断 | 候选来源；有 selection 时优先 `selected`，否则回退 `best` |
| `--candidate-rank` | `int` | `1` | 选择第几个候选，1-based |
| `--strategy-id` | `str` | `None` | 生成的策略 ID；未传时自动用 `run_id_topN` |
| `--strategy-name` | `str` | `None` | 生成的策略名称 |
| `--top-k` | `int` | `None` | 覆盖 bundle 中的 `top_k` |
| `--fee-rate` | `float` | `None` | 覆盖 bundle 中的 `fee_rate` |
| `--take-profit` | `float` | `None` | 覆盖 bundle 中的 `take_profit_ratio` |
| `--initial-capital` | `float` | `None` | 覆盖 bundle 中的 `initial_capital` |
| `--state-backend` | `str` | `sql` | 生成策略的状态后端 |
| `--replay-source` | `str` | `sql_eod` | 生成策略的回放源 |
| `--replay-strict` | flag | 开启 | 生成策略默认开启 strict replay |
| `--no-replay-strict` | flag | 关闭上项 | 关闭 strict replay |
| `--output-dir` | `str` | `None` | 指定 bundle 输出目录；默认 `artifacts/runs/<run_id>/bundle/` |

典型用法：

#### 9.2.1 直接用训练最佳公式打包

```bash
python -m workflow.bundle_builder --manifest artifacts/runs/slow_replace_exp01/manifest.json --source best --strategy-id slow_replace_best
```

适用场景：

- 训练完成后先不做二次筛选
- 直接将 `best_cb_formula.json` 打包成标准策略 bundle

#### 9.2.2 用二次筛选 Top-1 候选打包

```bash
python -m workflow.bundle_builder --manifest artifacts/runs/slow_replace_exp01/manifest.json --selection-output artifacts/runs/slow_replace_exp01/selection/top3_factors.json --source selected --candidate-rank 1 --strategy-id slow_replace_top1
```

适用场景：

- 已运行 `model_core/select_top_factors.py`
- 需要把筛选后的 Top-1 候选转成标准策略文件

### 9.3 统一流水线：`python -m workflow.pipeline`

#### 9.3.1 训练

```bash
python -m workflow.pipeline train --config model_core/config_slow_cs_replace.yaml --run-id slow_replace_exp01
```

适用场景：

- 统一通过 pipeline 触发训练
- 自动生成标准 run 目录

#### 9.3.2 筛选

```bash
python -m workflow.pipeline select --run-id slow_replace_exp01
```

适用场景：

- 自动读取 `artifacts/runs/<run_id>/train/best_cb_formula.json`
- 自动把筛选结果输出到 `selection/`

#### 9.3.3 打包

```bash
python -m workflow.pipeline bundle --run-id slow_replace_exp01 --source selected --candidate-rank 1 --strategy-id slow_replace_top1
```

适用场景：

- 基于 run 目录直接生成标准 bundle
- 不再手工传 `manifest` 和 `selection-output`

#### 9.3.4 验证

```bash
python -m workflow.pipeline verify --run-id slow_replace_exp01 --start 2025-01-01 --end 2025-12-31
```

适用场景：

- 自动定位 `bundle/strategy_bundle.json`
- 通过 bundle 直连 VERIFY

#### 9.3.5 模拟

```bash
python -m workflow.pipeline sim --run-id slow_replace_exp01 --mode strict_replay --date 2025-12-01
```

适用场景：

- 自动定位 `bundle/strategy_bundle.json`
- 通过 bundle 直连 SIM

#### 9.3.6 一键贯通

```bash
python -m workflow.pipeline e2e --config model_core/config_slow_cs_replace.yaml --run-id slow_replace_exp01 --verify-start 2025-01-01 --verify-end 2025-12-31 --sim-date 2025-12-01
```

适用场景：

- 一次性执行 train → select → bundle → verify → sim
- 适合标准实验闭环和回归检查

关键说明：

- 若只传 `--sim-date`，表示执行**单日**模拟
- `e2e` 中 `--sim-mode` 默认值是 `strict_replay`
- 若不显式传 `--sim-mode`，上面的示例等价于附带：
  - `--sim-mode strict_replay`
- 执行过程中会写：
  - `artifacts/runs/slow_replace_exp01/pipeline_status.json`

#### 9.3.7 一键贯通（单日 strict replay）

```bash
python -m workflow.pipeline e2e --config model_core/config_slow_cs_replace.yaml --run-id slow_replace_exp01 --verify-start 2025-01-01 --verify-end 2025-12-31 --sim-date 2025-12-01 --sim-mode strict_replay
```

适用场景：

- 标准单日严格回放闭环
- 验证某个目标交易日的 bundle 选股与模拟执行

#### 9.3.8 一键贯通（单日 live）

```bash
python -m workflow.pipeline e2e --config model_core/config_slow_cs_replace.yaml --run-id slow_replace_exp01 --verify-start 2025-01-01 --verify-end 2025-12-31 --sim-date 2025-12-01 --sim-mode live --live-quote-source dummy
```

适用场景：

- 单日 live 仿真闭环
- 训练、筛选、打包完成后直接做 live 模式联调

#### 9.3.9 一键贯通（区间 strict replay）

```bash
python -m workflow.pipeline e2e --config model_core/config_slow_cs_replace.yaml --run-id slow_replace_exp01 --verify-start 2025-01-01 --verify-end 2025-12-31 --sim-start-date 2025-01-01 --sim-end-date 2025-12-31 --sim-mode strict_replay
```

适用场景：

- 区间连续严格回放
- 观察组合持仓延续、净值轨迹和 SQL 状态累积

关键说明：

- 传 `--sim-start-date` 和 `--sim-end-date` 表示执行**区间**模拟
- 区间模拟底层仍受 `run_sim.py` 约束，**仅支持**：
  - `--sim-mode strict_replay`
- 如果传了区间参数但使用 `--sim-mode live`，底层会报错

#### 9.3.10 失败后断点续跑

```bash
python -m workflow.pipeline e2e --config model_core/config_slow_cs_replace.yaml --run-id slow_replace_exp01 --verify-start 2025-01-01 --verify-end 2025-12-31 --sim-date 2025-12-01 --resume
```

适用场景：

- 上一次 `e2e` 中途失败
- 希望跳过已完成阶段，只从失败点往后继续

关键说明：

- `--resume` 会读取：
  - `artifacts/runs/<run_id>/pipeline_status.json`
- 若某阶段状态已经是 `completed`，则自动跳过

#### 9.3.11 从指定阶段开始执行

```bash
python -m workflow.pipeline e2e --run-id slow_replace_exp01 --from-stage verify --verify-start 2025-01-01 --verify-end 2025-12-31 --sim-date 2025-12-01
```

适用场景：

- 训练、筛选、打包都已经完成
- 只想从 `verify` 开始继续

关键说明：

- `--from-stage verify` 会跳过：
  - `train`
  - `select`
  - `bundle`
- 但前提是这些阶段所需产物已经存在

## 10. 最推荐的 Straight-Through 操作手册

这一节给出当前项目最推荐的 straight-through 实操方式，按实际使用场景分为：

- 研究实验闭环
- 单日复盘
- 区间回放

建议统一原则：

- 始终显式传 `--run-id`
- 优先使用 `python -m workflow.pipeline`
- 若涉及新特征或 slow feature，统一使用与训练一致的 `--config`

### 10.1 场景一：研究实验闭环（推荐默认）

适用场景：

- 新配置实验
- 想从训练一路跑到验证和模拟
- 希望所有产物都沉淀在 `artifacts/runs/<run_id>/`

推荐命令：

```bash
python -m workflow.pipeline e2e --config model_core/config_slow_cs_replace.yaml --run-id exp_slow_replace_v1 --verify-start 2025-01-01 --verify-end 2025-12-31 --sim-date 2025-12-01 --sim-mode strict_replay
```

会自动完成：

- 训练
- 候选筛选
- 策略打包
- VERIFY
- 单日 strict replay 模拟

推荐产物检查顺序：

1. `artifacts/runs/exp_slow_replace_v1/manifest.json`
2. `artifacts/runs/exp_slow_replace_v1/selection/top3_factors.json`
3. `artifacts/runs/exp_slow_replace_v1/bundle/strategy_bundle.json`
4. `tests/artifacts/verification_report_*.md`

### 10.2 场景二：先训练，再人工检查筛选结果，再打包

适用场景：

- 你想先看筛选报告再决定用哪个候选
- 不想一步到位自动跑完所有阶段

推荐命令：

```bash
python -m workflow.pipeline train --config model_core/config_slow_cs_replace.yaml --run-id exp_manual_review_v1
python -m workflow.pipeline select --run-id exp_manual_review_v1
python -m workflow.pipeline bundle --run-id exp_manual_review_v1 --source selected --candidate-rank 1 --strategy-id exp_manual_review_top1
python -m workflow.pipeline verify --run-id exp_manual_review_v1 --start 2025-01-01 --end 2025-12-31
python -m workflow.pipeline sim --run-id exp_manual_review_v1 --mode strict_replay --date 2025-12-01
```

推荐人工检查文件：

- `artifacts/runs/exp_manual_review_v1/selection/top3_factors_report.md`
- `artifacts/runs/exp_manual_review_v1/bundle/strategy_bundle.json`

### 10.3 场景三：训练 best 直接打包，不做二次筛选

适用场景：

- 快速试运行
- 想直接验证训练输出的 best 因子

推荐命令：

```bash
python -m workflow.pipeline train --config model_core/default_config.yaml --run-id exp_fast_best_v1
python -m workflow.pipeline bundle --run-id exp_fast_best_v1 --source best --strategy-id exp_fast_best
python -m workflow.pipeline verify --run-id exp_fast_best_v1 --start 2025-01-01 --end 2025-12-31
python -m workflow.pipeline sim --run-id exp_fast_best_v1 --mode strict_replay --date 2025-12-01
```

特点：

- 路径最短
- 不依赖 `top3_factors.json`
- 适合快速实验闭环

### 10.4 场景四：单日复盘（最常用）

适用场景：

- 已经有一个 run_id
- 只想复盘某一个交易日

推荐命令：

```bash
python -m workflow.pipeline sim --run-id exp_slow_replace_v1 --mode strict_replay --date 2025-12-01
```

如果你想重新做验证再复盘：

```bash
python -m workflow.pipeline verify --run-id exp_slow_replace_v1 --start 2025-01-01 --end 2025-12-31
python -m workflow.pipeline sim --run-id exp_slow_replace_v1 --mode strict_replay --date 2025-12-01
```

### 10.5 场景五：区间回放（连续持仓）

适用场景：

- 想看策略在一段区间内的连续表现
- 关注净值曲线、持仓延续和状态累积

推荐命令：

```bash
python -m workflow.pipeline sim --run-id exp_slow_replace_v1 --mode strict_replay --start-date 2025-01-01 --end-date 2025-12-31
```

如果希望一键从训练跑到区间回放：

```bash
python -m workflow.pipeline e2e --config model_core/config_slow_cs_replace.yaml --run-id exp_range_replay_v1 --verify-start 2025-01-01 --verify-end 2025-12-31 --sim-start-date 2025-01-01 --sim-end-date 2025-12-31 --sim-mode strict_replay
```

注意：

- 区间模拟仅支持 `strict_replay`

### 10.6 场景六：live 联调

适用场景：

- 验证 bundle 到模拟引擎的 live 侧接线
- 本地环境没有 QMT 时做 dummy 行情联调

推荐命令：

```bash
python -m workflow.pipeline sim --run-id exp_slow_replace_v1 --mode live --date 2025-12-01 --live-quote-source dummy
```

如果接 QMT：

```bash
python -m workflow.pipeline sim --run-id exp_slow_replace_v1 --mode live --date 2025-12-01 --live-quote-source qmt
```

### 10.7 场景七：只替换交易参数做敏感性检查

适用场景：

- 因子固定
- 只想调整 `top_k / fee_rate / take_profit`

推荐命令：

```bash
python -m workflow.pipeline bundle --run-id exp_slow_replace_v1 --source selected --candidate-rank 1 --strategy-id exp_slow_replace_top1_tuned --top-k 10 --take-profit 0.05 --fee-rate 0.001
python -m workflow.pipeline verify --run-id exp_slow_replace_v1 --start 2025-01-01 --end 2025-12-31 --top-k 10 --take-profit 0.05 --fee-rate 0.001
```

说明：

- 第一条命令会生成参数覆盖后的 bundle
- 第二条命令是在 verify 侧继续覆盖，适合做快速参数对比

### 10.8 当前最推荐的日常使用顺序

如果你是日常研究/复盘，建议采用以下顺序：

1. 新实验：
   - `workflow.pipeline e2e`
2. 看筛选报告：
   - `artifacts/runs/<run_id>/selection/top3_factors_report.md`
3. 看验证报告：
   - `tests/artifacts/verification_report_*.md`
4. 做单日复盘：
   - `workflow.pipeline sim --run-id <run_id> --mode strict_replay --date <date>`
5. 做区间回放：
   - `workflow.pipeline sim --run-id <run_id> --mode strict_replay --start-date <start> --end-date <end>`

一句话总结：

- **新实验用 `e2e`**
- **日常复盘用 `sim`**
- **人工挑候选时用 `train + select + bundle + verify + sim`**
