# AlphaGPT: Agentic Quantitative Mining System

**An industrial-grade symbolic regression framework for Alpha factor mining, powered by Reinforcement Learning.**

Current Version: **V2.1 (Robustness Enhanced)**

---

## 🚀 Project Overview

AlphaGPT 是一个基于 Transformer 和强化学习 (Legacy Policy Gradient) 的自动化因子挖掘系统。它不像遗传算法那样随机变异，而是像 GPT 一样"学习"如何写出优秀的量化公式。

本项目针对 **可转债 (Convertible Bond)** 市场进行了深度适配，解决了传统 AI 挖因子容易过拟合、难以实盘的痛点。

---

## 📅 Version History

### **V2.3: Formula Structure Validator (Latest)**
*防止进化搜索产生"分数坍缩"的公式结构。*
- **Hard Filters**: 直接拒绝已知有害序列 (如 `SIGN → LOG`)，避免所有资产得分相同。
- **Soft Penalties**: 对可疑结构（如连续 LOG、TS_* 过多）进行扣分而非直接拒绝。
- **Documentation**: 新增 `docs/dangerous_structures.md` 记录所有规则。
- **Integration**: 在 `_worker_eval` 回测前进行验证，减少无效计算。

### **V2.2: Performance & Operators**
*修复随着训练进行导致的性能衰减，并扩充算子库。*
- **CPU Thread Fix**: 强制单进程模式下 Worker 使用单线程 (`torch.set_num_threads(1)`)，解决 CPU Oversubscription 问题。
- **I/O Optimization**: 引入 `MIN_SCORE_IMPROVEMENT` 阈值，减少无效 I/O。
- **New Operator**: 新增 `TS_BIAS5` (5日乖离率) 算子。

### **V2.1: Robustness Enhanced (Current)**
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
-内置 V2.1 稳健性评估指标 (`Split Sharpe`, `Stability`, `Active Ratio`)。

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

## 📊 Key Metrics Explained

| Metric | Definition | Good | Bad |
|--------|------------|------|-----|
| **Split Sharpe** | 训练集与验证集的夏普比率 | Train与Val接近且>1.5 | Val < 0 或 Train >> Val (Overfitting) |
| **Stability** | 滚动夏普均值 - 1.5 * 标准差 | > 0.5 | < 0 (不稳定) |
| **Max Drawdown** | 历史最大回撤 | < 20% | > 40% |
| **Active Ratio** | 实际持仓数 / 目标TopK | > 90% | < 50% (不可交易) |

---

## 📝 Configuration

修改 `model_core/config.py` 中的 `RobustConfig` 类以调整稳健性参数：

```python
class RobustConfig:
    TRAIN_TEST_SPLIT_DATE = '2024-06-01'  # 验证集切分点
    ROLLING_WINDOW = 60                   # 稳定性窗口
    MIN_SHARPE_VAL = 0.2                  # 验证集最低门槛
    MDD_W = 20.0                          # 回撤惩罚权重
```

---

**Disclaimer**: Quantitative trading involves significant risks. This code is for research purposes only.
