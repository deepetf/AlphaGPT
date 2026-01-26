# Model Core Analysis & CB Migration Plan

## 1. 现有架构分析

`model_core` 是一个基于 Transformer 强化学习生成 Alpha 因子的挖掘引擎。

### 核心模块
- **AlphaGPT (`alphagpt.py`)**: 因子生成器。使用 Transformer Decoder 架构，自回归地生成因子表达式（Opcodes + Features）。
- **StackVM (`vm.py`)**: 因子执行器。将 AlphaGPT 生成的 token 序列作为 RPN (Reverse Polish Notation) 公式执行，在特征 Tensor 上计算出最终的信号 Tensor。
- **CryptoDataLoader (`data_loader.py`)**: 数据加载器。从数据库读取 OHLCV 数据，转为 PyTorch Tensor，并计算基础特征（Input Features）。
- **AlphaEngine (`engine.py`)**: 训练循环。协调生成 -> 执行 -> 回测 -> 优化 (PPO/Policy Gradient) 的全过程。
- **MemeBacktest (`backtest.py`)**: 在线回测器。基于向量化计算对生成的信号进行快速回测，计算 Sharpe/Fitness 作为 RL 的 Reward。

### 数据流
```mermaid
graph LR
    DB[(MySQL/Local)] --> A[CBDataLoader]
    A --> |Tensor (T, N, F)| B[FeatureEngineer]
    B --> |Features (Bond & Stock)| C[StackVM]
    D[AlphaGPT] --> |Formula Tokens| C
    C --> |Signal Tensor| E[CBBacktest]
    E --> |Portfolio Sharpe| D
```
- **输入维度**: (Batch, Time, Assets, Features)
- **Time**: 时间步长（Daily）
- **Assets**: 资产数量（Top 1000 可转债 + 对应正股）
- **Features**: 债行情 + 正股行情 + 衍生指标

---

## 2. 可转债因子挖掘适配任务列表

为了挖掘“未曾想到”的因子，核心在于提供丰富的信息维度（正股+债）和强大的比较能力（截面算子），并约束其在安全边界内（强赎过滤）。

| 模块 | 改造点 | 详细说明 |
|------|--------|------|
| **配置** | `config.py` | 1. **Data Source**: 指向 `CB_HISTORY` (MySQL)<br>2. **Fees**: 设置为万分之一 (0.01%)<br>3. **Filters**: 定义 `MIN_BALANCE` (剩余规模), `MAX_PRICE` (价格上限过滤) |
| **数据加载** | `data_loader.py` | 1. **Dual-Data Loading**: 同时加载转债行情和正股行情。<br>2. **Tensor构造**: 特征维度应包含 `[O, H, L, C, V, Prem_Rt, Underlying_C, Underlying_V]`。<br>3. **Masking**: 生成 `Valid_Mask` 用于标记停牌、未上市或已退市的日子。 |
| **操作符** | `ops.py` | 1. **Cross-Sectional Ops (关键)**: 必须增加 `CS_RANK`, `CS_ZSCORE`。这是挖掘“相对性价比”因子的核心。<br>2. **TS Ops**: 保留 `TS_MEAN`, `TS_STD`, `TS_DELAY`, `TS_DELTA`。<br>3. **Logic Ops**: 增加 `IF_ELSE` (模拟条件选债)。 |
| **回测模型** | `backtest.py` | 1. **Portfolio Logic**: 摒弃单币择时，改为 **Top-K 轮动** (如每日持有 Score 最高的 20 只)。<br>2. **Reward Function**: 使用组合净值的 **Sharpe Ratio** 替代单资产收益中位数。<br>3. **Constraints**: 强制过滤即将强赎或剩余年限过短的标的。 |

---

## 3. 具体实施步骤 (Task List)

### Phase 1: 数据接入 (Data Layer - Parquet Optimized)
- [x] **Config**: 在 `config.py` 添加 `CB_PARQUET_PATH` 配置项。
- [x] **CBDataLoader**: 直接读取 `CB_DATA.parquet` (Long Format)。
    - [x] **Schema Integration**: 利用该单一宽表包含债与正股数据的特性，一次性提取核心特征：
        - **Bond**: `open`, `high`, `low`, `close`, `vol`
        - **Stock**: `close_stk`, `vol_stk`
        - **Metrics**: `conv_prem`, `dblow`, `ytm`, `remain_size`, `left_years`
    - [x] **Tensor Construction**:
        - 执行 `Pivot` 操作将长表转为 `[Time, Assets, Features]` 三维张量。
        - 自动处理停牌填充 (Forward Fill) 和未上市填充 (NaN/Zero)。

### Phase 2: 算子库增强 (Ops Layer)
- [x] **注册机制 (Registry)**: 实现 `OpsRegistry` 和 `FeatureEngineer` 配置驱动，支持灵活扩展。
- [x] **多维截面算子开发** (让 AI 自由选择稳健性 vs 爆发力):
    - [x] `CS_RANK(x)`: 纯排名 (0~1)。
    - [x] `CS_ROBUST_Z(x)`: 稳健标准化 (Median/MAD)。
    - [x] `CS_DEMEAN(x)`: 仅去均值。
- [x] **逻辑门算子开发** (让 AI 学习“软过滤”规则):
    - [x] `IF_ELSE`, `IF_POS`: 条件分支。
    - [x] `CUT_NEG`, `CUT_HIGH`: 排除算子。
- [x] **AlphaGPT 动态词表**: 模型自动根据 Registry 构建词表。

### Phase 3: 回测引擎重构 (Execution Layer)
- [ ] **CBBacktest 类**:
    - [x] `select_top_k(signal, k=20)`: 每天根据因子值选出前 20 名。
    - [x] `apply_physical_filters()`: **仅剔除物理上不可交易的标的** (停牌、退市、跌停)。**不要**硬编码人类偏见 (如双低/价格过滤)，留给 AI 的 `NEG_CUT` 去发现。
    - [x] `calculate_portfolio_returns()`: 计算每日持仓收益率。
    - [x] `calculate_reward()`: 计算 `Sharpe Ratio * 10` 作为奖励信号 (放大奖励以利于梯度传播)。

### Phase 4: 验证与训练
- [x] **基础设施验证**: `test_cb_backtest.py` 通过，确认了 Data->VM->Backtest 链路畅通。
- [ ] **开始挖掘**: 运行 `python -m model_core.engine`。
    - [ ] 观察 Log 输出的 "New King" 公式。
    - [ ] 收集一段时间后的 `best_cb_formula.json`。
- [ ] **解读**: 使用 `decode_formula` 解析出的公式（如 `ADD(CS_RANK(CLOSE), ...)`）。
