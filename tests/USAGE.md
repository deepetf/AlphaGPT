# 策略验证脚本使用说明

## 📋 功能说明

`verify_strategy.py` 用于验证Event-Driven策略引擎与Vector Backtest的一致性，并生成详细的交易和绩效报告。

---

## 🚀 使用方法

### 基本用法

```bash
# 使用最佳因子(best)运行2024全年
python tests\verify_strategy.py --start 2024-01-01 --end 2024-12-31

# 使用最佳因子运行2024 Q1
python tests\verify_strategy.py --start 2024-01-01 --end 2024-03-31
```

### 指定King因子

```bash
# 使用King #25 (Step 25)
python tests\verify_strategy.py --start 2024-01-01 --end 2024-12-31 --king 25

# 使用King #33 (Step 33)
python tests\verify_strategy.py --start 2024-01-01 --end 2024-12-31 --king 33

# 使用King #49 (当前最佳，等同于不指定--king)
python tests\verify_strategy.py --start 2024-01-01 --end 2024-12-31 --king 49
```

---

## 📊 可用的King因子

根据 `model_core/best_cb_formula.json`，当前可用的King因子：

| Step | Sharpe | 年化收益 | 最大回撤 | 特点 |
|:---|:---|:---|:---|:---|
| **49** | **1.78** | **31.08%** | **11.24%** | **当前最佳** (默认) |
| 33 | 1.32 | 20.52% | 12.92% | 首次正分 |
| 30 | 1.30 | 21.34% | 14.42% | 回撤优化 |
| 25 | 1.48 | 33.25% | 21.57% | 高收益 |
| 25 | 0.60 | 6.15% | 18.53% | 早期版本 |
| 20 | 0.80 | 15.50% | 26.72% | 引入VOL_STK_60 |
| 18 | 0.37 | 6.09% | 38.84% | 初期探索 |

---

## 📁 输出文件

所有输出文件保存在 `tests/artifacts/` 目录：

### 1. **verification_report.md**
- 绩效指标（累计收益、年化收益、夏普、最大回撤）
- 一致性指标（MAE, Correlation）
- 验证结论

### 2. **daily_trades.json**
- 每日交易明细（买入/卖出）
- 包含：code, name, quantity, price, amount

### 3. **daily_holdings.json**
- 每日持仓快照
- 包含：holdings, cash, equity

### 4. **daily_returns.csv**
- 每日收益率数据（CSV格式，可用Excel打开）
- 列：Date, Sim_Return, Backtest_Return, Diff, Sim_Equity

### 5. **selection_comparison.json**
- 选股一致性验证（Jaccard Index）

---

## 💡 使用示例

### 示例1：对比不同King因子的表现

```bash
# 运行King #25
python tests\verify_strategy.py --start 2024-01-01 --end 2024-12-31 --king 25

# 运行King #33
python tests\verify_strategy.py --start 2024-01-01 --end 2024-12-31 --king 33

# 运行最佳King #49
python tests\verify_strategy.py --start 2024-01-01 --end 2024-12-31 --king 49
```

然后对比 `tests/artifacts/verification_report.md` 中的绩效指标。

### 示例2：验证特定季度

```bash
# Q1
python tests\verify_strategy.py --start 2024-01-01 --end 2024-03-31

# Q2
python tests\verify_strategy.py --start 2024-04-01 --end 2024-06-30

# Q3
python tests\verify_strategy.py --start 2024-07-01 --end 2024-09-30

# Q4
python tests\verify_strategy.py --start 2024-10-01 --end 2024-12-31
```

---

## ⚠️ 注意事项

1. **数据范围**：确保指定的日期在数据集范围内（2022-08-01 至 2026-01-26）
2. **King编号**：使用 `--king` 时，必须是history中存在的step编号
3. **执行时间**：全年验证约需5-10分钟
4. **文件覆盖**：每次运行会覆盖 `tests/artifacts/` 中的文件

---

## 🔧 参数说明

```
--start     开始日期 (YYYY-MM-DD)，默认: 2024-01-01
--end       结束日期 (YYYY-MM-DD)，默认: 2024-12-31
--king      King因子step编号 (18, 20, 25, 30, 33, 49)，默认: None (使用best)
```

---

## 📞 问题排查

### 错误：King step XX not found
- 检查 `model_core/best_cb_formula.json` 中的 `history` 字段
- 确保使用的step编号存在

### 输出文件为空
- 检查日期范围是否有效
- 查看 `tests/verification.log` 日志文件

---

**最后更新**: 2026-01-30
