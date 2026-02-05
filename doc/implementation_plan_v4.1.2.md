# AlphaGPT V4.1.2 实施方案：逻辑精修 (Refinement Plan)

## 目标 (Goal)
响应专家审查反馈，对 V4.1.1 进行最后三点非功能性的逻辑校准与代码清理。

## 评估与对策 (Evaluation)

### 1. SimPass 统计口径歧义
- **现状**: 当前代码中 `SimPass` 包含了 `SIM_REPLACE` (因为 `!= SIM_REJECT`)。
- **专家意见**: 需明确口径。建议采纳 "Choice A"（即包含 Replace）。
- **我的判断**: 合理。`SIM_REPLACE` 意味着发现了一个比池中现有因子更优的变体，这是有效且高质量的产出，**应当**算作 SimPass。
- **对策**: 保持现有逻辑不变，但在代码中添加明确注释：`# Include SIM_REPLACE as Pass`。

### 2. SimFailShare 分母错误
- **现状**: `sim_fail_share = sim_abs / (bs - counts['HardPass'])`。这意味着把 SimFail 和 StructFail 放在一起比。
- **专家意见**: `SIM_REJECT` 发生在 `PASS` 样本上，分母应该是 `MetricPass`。
- **我的判断**: 正确。当前的计算方式缺乏统计学意义。
- **对策**: 修改分母为 `MetricPass` (即 `step_stats['PASS']`)。`SimFailShare = SimReject / MetricPass`。这一指标将直接反映“生成的好因子中有多少是重复的”。

### 3. 代码清理
- **现状**: `beta_is_locked` 变量已不再被逻辑使用，但声明仍在。
- **对策**: 彻底删除该变量的定义与引用。

## 实施计划 (Execution)
1.  **Engine Update**:
    -   修复 `SimFailShare` 计算公式。
    -   删除 `beta_is_locked`。
    -   添加 `SimPass` 统计口径的注释。
2.  **Verification**:
    -   这属于纯逻辑修正，无需新增测试配置，直接进行原有测试即可。

## 验收标准
- `SimFS` 指标在高熵阶段应能正确反映重复率（如 20%~50%），而不再是基于 Fail 的奇怪比例。
- 代码中无未使用的变量告警。
