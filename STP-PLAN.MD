# AlphaGPT Straight-Through Pipeline 实施计划

## 1. 背景

当前项目已经具备完整的三段能力：

- 训练：`model_core/engine.py`
- 验证：`tests/verify_strategy.py`
- 模拟：`strategy_manager/run_sim.py`

但三段流程目前主要通过人工方式串联：

- 训练产出 `model_core/best_cb_formula.json`
- 人工挑选公式或运行 `model_core/select_top_factors.py`
- 人工整理 `strategy_manager/strategies_config.json` 或手动指定 `strategy_id`
- 人工记忆并传递 `--config`
- 再分别运行 VERIFY 和 SIM

这意味着当前系统已经完成了“研究能力建设”，但还没有完成“研究结果到策略发布”的标准化交付链路。

本文档目标是规划一条更 straight-through 的流程，使训练、验证、模拟能够围绕统一产物自动串联，减少人工搬运、手工改文件和配置错配风险。

## 2. 现状拆解

### 2.1 当前主链路

#### 训练阶段

- 入口：`model_core/engine.py`
- 输入：模型 YAML 配置，如 `model_core/default_config.yaml`
- 输出：
  - `model_core/best_cb_formula.json`
  - `model_core/alphagpt_cb.pt`
  - `model_core/king_trades/king_*.json`

#### 候选收敛阶段

- 入口：`model_core/select_top_factors.py`
- 输入：
  - `model_core/best_cb_formula.json`
  - `model_core/top_factor_config.yaml`
- 输出：
  - `model_core/top3_factors.json`
  - `model_core/top3_factors_report.md`
  - 可选 AI review 结果

#### 验证阶段

- 入口：`tests/verify_strategy.py`
- 当前支持两种来源：
  - legacy：`model_core/best_cb_formula.json`
  - config-driven：`strategy_manager/strategies_config.json + strategy_id`

#### 模拟阶段

- 入口：`strategy_manager/run_sim.py`
- 来源：`strategy_manager/strategies_config.json`
- 运行模式：
  - `live`
  - `strict_replay`

### 2.2 当前断点

现阶段主要断点如下：

#### 断点 A：训练产物不是标准策略产物

`best_cb_formula.json` 更偏研究输出，缺少发布信息：

- 使用的模型配置快照
- 数据窗口
- 训练时间
- 选择来源
- 推荐策略参数

#### 断点 B：策略配置需要人工发布

训练后的最佳公式不能直接被 verify/sim 消费为“标准策略包”，还需要：

- 人工挑候选
- 人工写或修改 `strategies_config.json`
- 人工同步 `top_k / fee_rate / take_profit`

#### 断点 C：模型配置和策略配置是两套源

当前：

- YAML 决定特征、归一化、信号清洗
- JSON 决定公式、交易参数、状态后端

这导致 verify/sim 虽然支持 `--config`，但仍依赖操作者手工保持一致。

#### 断点 D：缺少统一 run identity

现在没有统一 `run_id`，导致以下问题：

- 训练、筛选、验证、模拟产物分散
- 无法一眼追踪“某个策略来自哪次训练”
- 无法沉淀标准实验记录

#### 断点 E：缺少发布动作

当前没有“promote/publish”步骤，研究结果到可运行策略之间没有标准化发布过程。

## 3. 目标状态

目标不是把所有逻辑塞进一个超大脚本，而是建立一条统一的产物链：

`训练运行` → `候选收敛` → `策略打包` → `验证` → `模拟`

所有阶段围绕一个统一的 `run_id` 和标准化 `bundle` 工作。

### 3.1 目标原则

#### 原则 1：不重写核心算法

保留现有：

- `engine.py`
- `select_top_factors.py`
- `verify_strategy.py`
- `run_sim.py`

主要在外层补 orchestration 层和标准产物层。

#### 原则 2：让产物而不是人工成为接口

从“人记住应该传什么文件”改成“下游脚本只认上游标准产物”。

#### 原则 3：先兼容再替换

新能力应先与现有流程并存：

- 旧命令继续可用
- 新 straight-through 入口优先服务新流程

#### 原则 4：研究与发布分层

不要直接把研究产物写进正式策略配置。

应区分：

- research artifact
- candidate artifact
- strategy bundle
- promoted strategy

## 4. 目标流程设计

### 4.1 目标 E2E 流程

#### Step 1：训练运行

输入：

- 模型配置 YAML
- 可选 `data_start_date`

输出：

- `run_id`
- merged model config snapshot
- `best_cb_formula.json`
- 训练元数据 `manifest.json`

#### Step 2：候选收敛

输入：

- `manifest.json`
- `best_cb_formula.json`

输出：

- `top_candidates.json`
- `top_candidates_report.md`
- 可选 AI review 结果

#### Step 3：策略打包

输入：

- `manifest.json`
- 选中的候选公式
- 发布参数模板

输出：

- `strategy_bundle.json`
- `formula_top1.json`
- 可选 `generated_strategy_config.json`

#### Step 4：验证

输入：

- `strategy_bundle.json`

输出：

- verify 报告
- 校验状态
- 可视化与交易明细

#### Step 5：模拟

输入：

- `strategy_bundle.json`

输出：

- strict replay 结果
- live/sim 结果
- 执行状态与持仓历史

### 4.2 新的调用方式

理想状态下，下游不再要求操作者同时记住：

- `--config`
- `--strategies-config`
- `--strategy-id`
- `formula_path`

而是统一输入：

- `--manifest`
或
- `--bundle`

## 5. 建议新增的产物模型

### 5.1 Run Manifest

建议新增：

`artifacts/runs/<run_id>/manifest.json`

建议字段：

```json
{
  "run_id": "20260311_163000_default",
  "created_at": "2026-03-11T16:30:00+08:00",
  "stage": "train_completed",
  "project_version": "V5.96",
  "git_commit": "optional",
  "training": {
    "entry": "python -m model_core.engine",
    "config_path": "model_core/config_slow_cs_replace.yaml",
    "resolved_config_snapshot_path": "artifacts/runs/<run_id>/resolved_model_config.yaml",
    "data_start_date": "2022-08-01"
  },
  "artifacts": {
    "best_formula_path": "artifacts/runs/<run_id>/best_cb_formula.json",
    "model_weight_path": "artifacts/runs/<run_id>/alphagpt_cb.pt"
  }
}
```

作用：

- 记录本次 run 的唯一身份
- 作为后续 select / bundle / verify / sim 的锚点

### 5.2 Candidate Selection Output

建议新增：

`artifacts/runs/<run_id>/selection/top_candidates.json`

建议字段：

- `run_id`
- `selection_config_path`
- `candidate_rank`
- `formula`
- `selection_score`
- `metrics`
- `ai_review_summary`

### 5.3 Strategy Bundle

建议新增：

`artifacts/runs/<run_id>/bundle/strategy_bundle.json`

建议字段：

```json
{
  "bundle_version": "v1",
  "run_id": "20260311_163000_default",
  "strategy_id": "king_slow_run_20260311",
  "strategy_name": "king_slow_run_20260311",
  "formula_path": "artifacts/runs/<run_id>/bundle/formula_top1.json",
  "model_config_path": "artifacts/runs/<run_id>/resolved_model_config.yaml",
  "source": {
    "training_manifest": "artifacts/runs/<run_id>/manifest.json",
    "selection_output": "artifacts/runs/<run_id>/selection/top_candidates.json",
    "candidate_rank": 1
  },
  "params": {
    "initial_capital": 1000000,
    "top_k": 5,
    "take_profit_ratio": 0.06,
    "fee_rate": 0.001,
    "replay_strict": true,
    "replay_source": "sql_eod",
    "state_backend": "sql"
  }
}
```

作用：

- 成为 verify/sim 的统一输入
- 摆脱对人工修改 `strategies_config.json` 的依赖

## 6. 建议新增的目录结构

建议在项目根目录新增：

```text
artifacts/
  runs/
    <run_id>/
      manifest.json
      resolved_model_config.yaml
      train/
        best_cb_formula.json
        alphagpt_cb.pt
        king_trades/
      selection/
        top_candidates.json
        top_candidates_report.md
        factor_ai_reviews.json
      bundle/
        strategy_bundle.json
        formula_top1.json
        generated_strategy_config.json
      verify/
        verification_report.md
        daily_returns.csv
        daily_trades.json
        daily_holdings.json
      sim/
        strict_replay_summary.json
        live_summary.json
```

### 6.1 为什么建议用 `artifacts/runs/<run_id>/`

优点：

- 单次实验所有产物聚合
- 更容易做归档与比较
- 减少当前 `model_core/`、`tests/artifacts/`、`execution/portfolio/` 分散造成的上下文丢失

## 7. CLI 设计草案

## 7.1 设计目标

新增统一 orchestration 层，而不是侵入式重写现有脚本。

建议新建：

- `workflow/`
- 或 `pipeline/`

推荐入口：

`python -m workflow.pipeline`

### 7.2 推荐子命令

#### 7.2.1 train

```bash
python -m workflow.pipeline train --config model_core/config_slow_cs_replace.yaml --data-start-date 2022-08-01
```

功能：

- 调用 `model_core.engine`
- 生成 `run_id`
- 收集训练产物
- 写 `manifest.json`

#### 7.2.2 select

```bash
python -m workflow.pipeline select --run-id 20260311_163000_default
```

功能：

- 基于 manifest 自动定位 `best_cb_formula.json`
- 调用 `model_core.select_top_factors`
- 输出 selection 产物

#### 7.2.3 bundle

```bash
python -m workflow.pipeline bundle --run-id 20260311_163000_default --candidate top1
```

功能：

- 从 selection 结果生成 `strategy_bundle.json`
- 自动写 `formula_top1.json`

#### 7.2.4 verify

```bash
python -m workflow.pipeline verify --run-id 20260311_163000_default --candidate top1 --start 2025-01-01 --end 2025-12-31
```

功能：

- 从 bundle 自动取：
  - 公式
  - 模型配置
  - 策略参数
- 调用 `tests/verify_strategy.py`

#### 7.2.5 sim

```bash
python -m workflow.pipeline sim --run-id 20260311_163000_default --candidate top1 --mode strict_replay --date 2025-12-01
```

功能：

- 从 bundle 自动取配置
- 调用 `strategy_manager/run_sim.py`

#### 7.2.6 e2e

```bash
python -m workflow.pipeline e2e --config model_core/config_slow_cs_replace.yaml --data-start-date 2022-08-01 --verify-start 2025-01-01 --verify-end 2025-12-31 --sim-date 2025-12-01
```

功能：

- 串联：
  - train
  - select
  - bundle
  - verify
  - sim

## 8. 最小改造方案

目标是低风险、逐步直通，不一次性重构全项目。

### 8.1 Phase 1：加标准产物，不改主流程

#### 目标

把“人工记忆链路”变成“文件驱动链路”。

#### 改动建议

1. 训练阶段新增 `manifest.json`
2. 保存 merged model config snapshot
3. 增加标准产物目录 `artifacts/runs/<run_id>/`
4. 将训练关键产物复制或输出到 run 目录

#### 影响文件

- `model_core/engine.py`
- 新增 `workflow/run_manifest.py`

#### 价值

- 立即提升可追踪性
- 为后续 straight-through 奠定基础

### 8.2 Phase 2：加 bundle，不再手改策略配置

#### 目标

让策略从“手工 JSON”变成“自动生成 JSON”。

#### 改动建议

1. 新增 `workflow/bundle_builder.py`
2. 从 `best_cb_formula.json` 或 `top_candidates.json` 自动生成：
   - `formula_top1.json`
   - `strategy_bundle.json`
   - `generated_strategy_config.json`
3. 扩展 `load_strategies_config` 兼容 generated config

#### 影响文件

- 新增 `workflow/bundle_builder.py`
- 可选新增 `strategy_manager/generated/`

#### 价值

- 去掉最容易出错的手工步骤
- 使 verify/sim 有统一输入

### 8.3 Phase 3：verify/sim 支持 bundle 输入

#### 目标

下游统一消费 bundle。

#### 改动建议

1. `tests/verify_strategy.py` 增加：
   - `--bundle`
   - `--manifest`
2. `strategy_manager/run_sim.py` 增加：
   - `--bundle`
   - `--manifest`
3. 若传 bundle，则自动解析：
   - 模型配置路径
   - 公式路径
   - 策略参数

#### 影响文件

- `tests/verify_strategy.py`
- `strategy_manager/run_sim.py`
- 新增 `workflow/bundle_loader.py`

#### 价值

- 真正实现“下游只认上游标准产物”

### 8.4 Phase 4：统一 pipeline 入口

#### 目标

提供一套统一命令，而不是让用户记住 4~5 个脚本。

#### 改动建议

新增：

- `workflow/pipeline.py`

支持子命令：

- `train`
- `select`
- `bundle`
- `verify`
- `sim`
- `e2e`

#### 价值

- 降低操作门槛
- 降低出错率
- 让项目真正具备研究到模拟的一条龙入口

## 9. 推荐实施顺序

### 第一阶段：本周可完成

- 新增 `run_id`
- 新增 `manifest.json`
- 新增 `resolved_model_config.yaml`
- 新增 `artifacts/runs/<run_id>/` 输出结构

### 第二阶段：下一步优先

- 自动生成 `strategy_bundle.json`
- 自动生成 `generated_strategy_config.json`
- 减少人工改 `strategies_config.json`

### 第三阶段：完成直通

- verify/sim 支持 `--bundle`
- 新增统一 pipeline CLI

## 10. 验收标准

### 10.1 Phase 1 验收

训练完成后，能够在 run 目录看到：

- `manifest.json`
- `resolved_model_config.yaml`
- `train/best_cb_formula.json`

### 10.2 Phase 2 验收

无需人工修改 `strategies_config.json`，即可生成：

- `bundle/strategy_bundle.json`
- `bundle/formula_top1.json`

### 10.3 Phase 3 验收

以下命令能够成立：

```bash
python tests/verify_strategy.py --bundle artifacts/runs/<run_id>/bundle/strategy_bundle.json --start 2025-01-01 --end 2025-12-31
python strategy_manager/run_sim.py --bundle artifacts/runs/<run_id>/bundle/strategy_bundle.json --mode strict_replay --date 2025-12-01
```

### 10.4 Phase 4 验收

以下命令能够贯通：

```bash
python -m workflow.pipeline e2e --config model_core/config_slow_cs_replace.yaml --verify-start 2025-01-01 --verify-end 2025-12-31 --sim-date 2025-12-01
```

## 11. 风险与注意事项

### 11.1 不要过早重构核心逻辑

优先做 orchestration 层，而不是改训练、回测、模拟内核。

### 11.2 不要直接把研究结果覆盖正式策略配置

建议保留：

- 正式策略配置：`strategy_manager/strategies_config.json`
- 实验生成策略：`artifacts/runs/<run_id>/bundle/generated_strategy_config.json`

### 11.3 需要保留兼容路径

以下旧路径应继续可用：

- `python -m model_core.engine`
- `python tests/verify_strategy.py --strategy-id ...`
- `python strategy_manager/run_sim.py --strategy-id ...`

### 11.4 README 历史描述要注意版本漂移

README 中存在一些历史版本说明，与当前最新代码默认行为可能不完全一致。实施时应以当前代码逻辑为准。

## 12. 结论

当前项目不缺训练、验证、模拟能力，缺的是“统一产物和统一入口”。

最优改造方向不是重写三套系统，而是新增一层：

- `manifest`
- `bundle`
- `pipeline`

通过这三层，可以把目前依赖人工搬运和手工改配置的流程，逐步升级为真正 straight-through 的研究到模拟流程。

---

## 13. 推荐下一步实现清单（最小可落地）

若按最小闭环推进，建议下一轮直接实施以下 5 项：

1. 新增 `workflow/run_manifest.py`
2. 训练结束自动产出 `artifacts/runs/<run_id>/manifest.json`
3. 训练结束自动落盘 `resolved_model_config.yaml`
4. 新增 `workflow/bundle_builder.py`
5. 新增 `tests/verify_strategy.py --bundle` 支持

完成这 5 项后，项目就会从“人工串联”进入“半自动直通”阶段。

