# AlphaGPT V4.0 实施方案：动力学闭合与效率革命

## 目标 (Goal)
根本性治理 `LOW_VARIANCE` 引起的模式坍缩。通过**惩罚阶梯 (Hierarchy of Failure)**、**规范化 LRU 回测缓存**及**自适应熵控制回路**，确保模型在 800 步及以上的训练中保持高质量探索能力。

## 核心挑战
1.  **坍缩迁移与奖励饱和**: 防止智能体在 `LowVar` 与 `StructInvalid` 坑位间横跳，并利用分级梯度引导指标优化。
2.  **冗余计算与确定性缓存**: 在提速的同时，必须保证缓存不会因为评估噪声而锁死错误的策略。
3.  **恢复逻辑的稳健性**: 熵控制器需在低 Pass 频率下具备高敏感度与合理的冷却复位机制。

## 提议变更

### 1. 训练引擎 (model_core/engine.py)

#### [FIX] 惩罚阶梯与分层 (The Hierarchy of Failure)
-   **精密梯队 (由坏到好)**:
    -   `PENALTY_EXEC`: -10.0 (系统性崩溃)
    -   `PENALTY_STRUCT`: -8.0 (语法错误/禁用算子)
    -   `PENALTY_LOWVAR`: -6.5 (低方差/智能体避风港)
    -   **`PENALTY_METRIC` (Clamped Gradient)**: `clamp(base - scale * gap, min=-6.0, max=-4.0)`。
        -   **Gap 定义**: 采用归一化合成 Gap（Sharpe/MDD 等偏离阈值的加权标量）。
        -   **监控**: 记录 Gap 的 `mean / p50 / p90` 分布。
    -   **`PENALTY_SIM`**: -1.5 (轻微惩罚，减少冗余，避免过度驱动随机探索)。

#### [NEW] 确定性 LRU 缓存 (Safe LRU Cache)
-   **实现**: `collections.OrderedDict`，上限 100k 条，淘汰 20%。
-   **规范化 (Canonical Form)**: Token 序列标准化后再生成 Key（如去除尾部无效算子）。
-   **确定性保护**: 仅针对逻辑/方差过滤等高确定性结果进行缓存；若回测包含非确定性（如随机采样窗口），Key 必须包含 `seed/partition_id`。

#### [NEW] 动力学观测 2.0++ (Observability 2.0+)
-   **指标集**:
    -   **`FailShare%`**: `Struct / LowVar / Metric / Sim` 各类失败占总样本的比例（监测迁移）。
    -   `PassAbs`: 每步通过硬过滤的绝对公式数。
    -   `TopFail`: 统计前三名报错原因。
    -   `CHit%` & `CanonCollide%`。

#### [NEW] Adaptive Entropy 控制器 2.0 (分级版)
-   **策略**:
    -   **一级触发**: `PassRate < 1%` -> $\beta$ 重置至 0.04，上限 0.06。
    -   **二级触发**: `1% < PassRate < 5%` -> $\beta$ 线性上调。
    -   **安全恢复**: `PassRate > 2%` 且 `Uniq%(生成口径) > 80%` 且 `StructInvalid` 未恶化。
    -   **冷却/滞回**: 触发后强制锁定 10 步；恢复需满足 `PassAbs` 持续稳定。

### 2. 配置同步与同步调整 (test_config.yaml)
-   同步建立上述惩罚阶梯参数，定义归一化 Gap 的基准值。

## 验收标准 (Acceptance Criteria)
1.  **Pass 活性**: 训练后 200 步平均 `PassRate > 0.3%`。
2.  **迁移判定**: `StructFailShare%` 的 p95 需长期低于 95%（防止彻底向语法错误坍缩）。
3.  **指标优化**: `MetricFail` 的 Gap 分位数呈现收敛趋势。
4.  **提速**: `Steps/Sec` 提升需与 `CHit%` 强相关。
