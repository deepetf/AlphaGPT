# AlphaGPT V4.1.1 实施方案：逻辑对齐与口径修正

## 目标 (Goal)
修复 V4.1 实现中存在的 3 处逻辑偏差：针对 `SimPass` 统计缺失、一级熵触发力度不足、以及奖励饱和监控口径不一的问题进行精准修正。

## 核心修正 (Core Refinements)

### 1. 通用成功口径对齐 (Success Definitions)
- **修正 SimPass 统计**: 将 `SimPass` 的计算移出相似度处理分支。遍历结束后，统一对 `status == "PASS"` 且 `final_status != "SIM_REJECT"` 的样本进行计数。
- **确立 MetricFailReward Std**: 统计 `status` 包含 `METRIC_` 的样本奖励标准差，用于监控奖励地形的饱和度。

### 2. 控制器逻辑强化 (Control Logic)
- **强化一级采样触发 (PR < 1%)**: 将 `current_beta` 直接设为 `0.06` (或 `max(0.06, START)`)，确保在该“救火”状态下拥有足够的探索压强，而非仅仅重置到 0.04。
- **清理逻辑**: 移除未使用的 `beta_is_locked` 标志位。
- **初始化调整**: 将 `hard_pass_rate_history` 的初始化均值下调 (e.g., 0.5) 或保持 1.0 但引入预热期，防止前期“由于历史太好”而反应过迟。

### 3. 多维度监控补完 (Observability)
- **新增 SimFailShare%**: 显示 `SIM_REJECT` 在总失败中的占比，用于评估去重机制是否过于激进。
- **奖励饱和告警**: 若 `reward_std < 0.1` 持续 50 步，在日志中打印 `[LOG] Reward Saturation Detected` 预警。

### 4. 动态惩罚回路闭环 (Penalty Loop)
- **LowVar 倍率回收**: 若 `LowVarRate < 10%` 持续 20 步，将 `lowvar_penalty_multiplier` 恢复至 1.0 (可选，增加系统回弹能力)。

## 评估与计划
- **合理性**: 极高。修正了因口径问题导致的统计失真。
- **实施计划**: 
    1. 修改 `engine.py` 内部统计逻辑。
    2. 更新 `train` 循环中的监控触发器。
    3. 调整 Tqdm 日志输出格式。
