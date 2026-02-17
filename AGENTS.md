# AlphaGPT - Agent Development Guide (V5.0)

## 语言偏好
永远使用中文交互和写文档。

## 项目定位
AlphaGPT 是一个面向可转债（CB）策略研究与模拟盘执行的量化框架，核心目标是：
- 因子挖掘（`model_core`）
- 向量化回测（`model_core/backtest.py`）
- SQL EOD 严格回放模拟（`strategy_manager/run_sim.py`）
- 事件驱动验证与对齐（`tests/verify_strategy.py`）

## 当前核心架构
```text
model_core/        因子引擎、VM、向量化回测
data_pipeline/     SQL/实时数据接入与严格回放加载器
strategy_manager/  模拟盘运行器、调仓器、净值管理、多策略调度
execution/         模拟成交执行与交易历史
tests/             对齐验证脚本与测试
```

## 常用命令
```bash
# 训练/挖掘
python -m model_core.engine

# 单日模拟
python strategy_manager/run_sim.py --date 2025-12-01 --replay-strict --replay-source sql_eod --top-k 10 --take-profit 0.06

# 区间模拟（连续持仓）
python strategy_manager/run_sim.py --start-date 2025-01-01 --end-date 2025-12-31 --replay-strict --replay-source sql_eod --top-k 10 --take-profit 0.06

# 验证（事件驱动 vs 向量回测）
python tests/verify_strategy.py --start 2025-01-01 --end 2025-12-31 --take-profit 0.06 --initial-cash 1000000
```

## 对齐口径（必须遵守）
1. 收益时序：`t` 日持仓收益使用 `t+1` 价格口径。
2. TP 规则：触发价为 `prev_close * (1 + take_profit)`。
3. TP 成交规则：
- 跳空触发：按 `open` 成交；
- 盘中触发：按 `tp_trigger` 成交。
4. Verify 口径：
- 当日收益按“交易后净值”计算；
- TP 与手续费计入当日收益。
5. Sim 严格回放：
- 必须优先使用 `--replay-strict --replay-source sql_eod`。

## 代码风格
- Imports 顺序：stdlib -> third-party -> local。
- 命名：类 PascalCase，函数 snake_case，常量 UPPER_SNAKE_CASE。
- IO 操作使用 `async/await`（适用模块内保持一致）。
- 对外部依赖调用加 `try/except` 并记录日志。
- 复杂逻辑可加简短注释，避免冗余注释。

## V5.0 新模块（关键）
- `strategy_manager/run_sim.py`
- `strategy_manager/sim_runner.py`
- `strategy_manager/nav_tracker.py`
- `strategy_manager/strategy_config.py`
- `strategy_manager/multi_sim_runner.py`
- `execution/sim_trader.py`
- `data_pipeline/realtime_provider.py`
- `data_pipeline/sql_strict_loader.py`

## Git 提交边界
可以提交：
- 代码文件（`.py`、必要配置）
- 文档（`README.md`、`AGENTS.md`）

不要提交：
- 回放/验证产物：`execution/plans/*.json`、`execution/portfolio/*.json`、`tests/artifacts/*`
- 临时输出：`*.log`、`ConsoleLog.txt`、临时测试文件

## 测试建议
提交前至少执行：
```bash
python -m py_compile tests/verify_strategy.py strategy_manager/sim_runner.py strategy_manager/run_sim.py execution/sim_trader.py data_pipeline/sql_strict_loader.py
```

## README 版本更新规范（新增）
- 从 V5.4 起，每次更新 `README.md` 的版本历史时，必须同步补充“本次主要功能更新对应的示例命令行”。
- 示例命令需可直接复制执行，优先覆盖本次变更涉及的核心路径（如 `live`、`strict_replay`、`verify`）。
- 若本次包含多项关键改动，至少提供 2-3 条代表性命令，并在命令说明中明确适用场景（单日/区间/多策略）。
