# AlphaGPT: Agentic Quantitative Mining System

**An industrial-grade symbolic regression framework for Alpha factor mining, powered by Reinforcement Learning.**

Current Version: **V4.2: Alpha Efficiency (Current)**

---

## 📅 Version History

### **V4.2: Alpha Efficiency (Current)**
*止盈逻辑与评价体系优化。*
- **Vectorized Take-Profit**: 全面向量化止盈逻辑，大幅提升训练速度；支持开盘跳空与盘中止盈。
- **Simplified Buy-Back**: 锁定止盈当日收益，仅对仍在 Top-K 的标的计算额外买回成本，平衡收益与换手。
- **Strict Price Filter**: 引入价格有效性检查 (0 < Price < 10000)，彻底消除脏数据引起的收益率污染。
- **IC/IR Alignment**: 修正因子评价对齐口径，对无效样本返回 None 以防止 RL 指标偏置。

### **V4.1: Engineering Hardening**
*工程加固与口径真实性：确保每一分 Alpha 都经得起推敲。*
- **Hierarchy of Failure**: 建立 (`EXEC` < `STRUCT` < `LOWVAR` < `METRIC` < `SIM`) 惩罚阶梯，提供清晰的强化学习梯度。
- **Grammar-Guided Decoding**: 利用 Action Masking 技术将无效语法生成率从 99% 降至 **0%**。
- **Rolling Window Controller**: 引入 10步滑动窗口熵控制，并结合 **3-Level Success (Hard/Metric/Sim)** 口径，彻底消除训练震荡。
- **SimPass 2.0**: 重新定义多样性成功标准，认可“优胜劣汰”的替换行为 (`SIM_REPLACE`)。

### **V3.5: Efficiency**
*效率革命：解决相似性冗余。*
- **Safe LRU Cache**: 100k上限确定性缓存，消除冗余回测。
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
    - 将 `INPUT_FEATURES` 和 `RobustConfig` 移至外部 YAML 配置文件 (`default_config.yaml`)。
    - 支持通过命令行参数 `--config` 加载自定义配置文件，无需修改代码。
- **CLI Support**: `engine.py` 支持 `--config` 参数。
- **Backward Compatibility**: `config.py` 使用 Metaclass 保持 API 兼容性，确保旧代码无缝运行。

### **V3.1: Diversity & Anti-Stacking**
*Implemented advanced mechanisms to prevent "Formula Stacking" and boost diversity.*
- **Local Density Check**:
    - 在 `Formula Validator` 中引入滑动窗口密度检测 (`Window=6, MaxTS=3`)。
    - 有效打击利用 `TS_MEAN` 堆叠刷高 Sharpe 的行为，同时保护了合法的多因子逻辑。
- **Enhanced Exploration**:
    - **Entropy Regularization**: 引入线性衰减的熵正则项 (`Beta: 0.04 -> 0.005`)，在训练初期强制模型探索未知领域。
    - **Diversity Pool**: 维护一个 Top-50 多样性池，基于 **Jaccard Similarity** (阈值 0.8) 过滤同质化公式。
- **Outcome**: 成功挖掘出 `DBLOW` (双低), `VOL_STK` (正股波动), `PREM` (溢价率) 等多种不同逻辑的高性能因子。

### **V3.0: CB Migration & Perfect Simulation**
*Completed migration from Crypto to Convertible Bonds and achieved pixel-perfect simulation alignment.*
- **System Migration**:
    - 全面适配可转债 (CB) 市场特性（T+0, 涨跌幅, 整手交易, 债券属性）。
    - 因子工厂与特征工程针对 CB 结构进行了重构。
- **Perfect Verification**:
    - **Event-Driven Simulation** 与 **Vector Backtest** 达到 >99% 相关性 (Correlation)。
    - 修复了持仓系统"双轨制" Bug，彻底消除了模拟误差。
    - 验证了策略在 2023-2026 四年间的稳健表现 (年化 30%~60%)。
- **Simulation Suite**:
    - 升级 `verify_strategy.py` 支持命令行参数配置（初始资金、时间区间）。
    - 增强交易日志，支持名称显示与详细一致性检查 (Return MAE, Jaccard)。
- **Configuration**:
    - 费率下调至实盘水平 (万五, 0.0005)，释放了高频策略潜力。

### **V2.3: Formula Structure Validator**
*防止进化搜索产生"分数坍缩"的公式结构，并增强收益-风险平衡。*
- **Hard Filters**: 直接拒绝已知有害序列 (如 `SIGN → LOG`)，避免所有资产得分相同。
- **Soft Penalties**: 对可疑结构（如连续 LOG、TS_* 过多）进行扣分而非直接拒绝。
- **Documentation**: 新增 `docs/dangerous_structures.md` 记录所有规则。
- **Integration**: 在 `_worker_eval` 回测前进行验证，减少无效计算。
- **Fee Rate Management**: 交易费率统一由 `RobustConfig.FEE_RATE` 管理 (0.001，千分之一)。
- **Return Reward**: 评分体系新增**年化收益率奖励** (`RET_W = 5.0`)，鼓励挖掘高回报策略。

### **V2.2: Performance & Operators**
*修复随着训练进行导致的性能衰减，并扩充算子库。*
- **CPU Thread Fix**: 强制单进程模式下 Worker 使用单线程 (`torch.set_num_threads(1)`)，解决 CPU Oversubscription 问题。
- **I/O Optimization**: 引入 `MIN_SCORE_IMPROVEMENT` 阈值，减少无效 I/O。
- **New Operator**: 新增 `TS_BIAS5` (5日乖离率) 算子。

### **V2.1: Robustness Enhanced**
*专注于挖掘"稳健、可实盘"的因子，而非单纯的高夏普。*
- **Split Validation**: 强制进行 **Train/Test 分段验证**。若 Valid 段表现差，直接淘汰。
- **Rolling Stability**: 引入 **Mean - k*Std** 评分机制，惩罚收益曲线剧烈波动的因子。
- **Drawdown Control**: 显式惩罚最大回撤 (MDD)。
- **Tradability Constraints**:
    - **Active Ratio**: 剔除因停牌或数据缺失导致无法满仓的因子。
    - **Valid Days**: 剔除样本量不足的偶然高分因子。
- **Composite Score**: 综合 `(Train+Val) * Stability - MDD` 的多维评分体系。

### **V1.0: Foundation (No Lookahead)**
*专注于消除"未来函数"与"数据泄露"。*
- **Strict Causality**: 所有算子 (`TS_DELAY`, `TS_MEAN`) 经严格审计，杜绝 `torch.roll` 造成的循环泄露。
- **Robust Normalization**: 使用 `Rolling(60)` 进行特征标准化，而非全局 Z-Score，彻底消除全样本偏差。
- **Aligned Data Pipeline**: 统一 `CBDataLoader`，确保训练、验证、回测使用完全一致的数据流 (`[Time, Assets]`)。

---

## 📊 Key Metrics Explained

| Metric | Definition | Good | Bad |
|--------|------------|------|-----|
| **Split Sharpe** | 训练集与验证集的夏普比率 | Train与Val接近且>1.5 | Val < 0 或 Train >> Val (Overfitting) |
| **Stability** | 滚动夏普均值 - 1.5 * 标准差 | > 0.5 | < 0 (不稳定) |
| **Max Drawdown** | 历史最大回撤 | < 20% | > 40% |
| **Active Ratio** | 实际持仓数 / 目标TopK | > 90% | < 50% (不可交易) |

---

## 🛠️ Core Modules

### 1. AlphaGPT Model
- 一个轻量级 Transformer Decoder。
- 学习算子语法 (`ADD`, `SUB`, `TS_DELAY`...) 和特征组合。
- 通过 RL (Policy Gradient) 优化，根据回测 Reward 调整生成概率。

### 2. StackVM (Vectorized)
- 高性能 PyTorch 向量化栈虚拟机。
- 支持时序算子 (`TS_*`)、横截面算子 (`CS_*`) 和逻辑算子 (`IF_POS`...)。
- **零未来函数设计**。

### 3. CBBacktest (Robust)
- Top-K 轮动策略回测器。
- 支持 `Transaction Fee` 和 `Turnover` 惩罚。
- 内置 V2.1 稳健性评估指标 (`Split Sharpe`, `Stability`, `Active Ratio`)。

---

## ⚡ Usage

### 1. Training (Mining)
启动多进程挖掘：
```bash
python -m model_core.engine
```
*自动保存表现最好的 "Kings" 到 `model_core/verified_trades/` 和 `best_cb_formula.json`。*

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

## 📝 Configuration

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
