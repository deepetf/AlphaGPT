# AlphaGPT: Agentic Quantitative Mining System

**An industrial-grade symbolic regression framework for Alpha factor mining, powered by Reinforcement Learning.**

Current Version: **V5.1: SQL Live State & Replay QA (Current)**

---

## 馃搮 Version History

### **V5.1: SQL Live State & Replay QA (Current)**
*完成 SQL 状态持久化、回放可观测性与一致性验收工具。*
- **SQL State Backend**: `run_sim` 支持 `--state-backend sql`，将 `nav/holdings/trades` 写入 `sim_nav_history`、`sim_daily_holdings`、`sim_trade_history`。
- **Strategy Isolation**: 单策略固定 `strategy_id=default`，多策略使用 `strategies_config.json` 的 `id` 对应数据库分区。
- **Replay Robustness**: 修复区间回放状态重置与运行时异常；完善日志编码与关键日志可读性。
- **Execution Reliability**: 调仓新增“现金预算裁剪 + 排名优先买入”，避免候选 10 只但落仓 9 只的静默失败。
- **Diagnostics**: 新增 `tests/verify_sim_sql_replay.py`，自动检查 NAV 恒等式、持仓计数一致性、异常日和候选缺失。
- **Docs & Migration**: 补充 `simrun_live.md` 与建表脚本 `infra/migrations/20260214_create_simrun_live_tables.sql`。

### **V5.0: Sim-Verify Alignment**
*SQL EOD 回放、模拟盘执行链路与验证口径对齐。*
- **SQL EOD Strict Replay**: 新增 `SQLStrictLoader`，`run_sim` 支持 `--replay-strict --replay-source sql_eod`，并支持 `--start-date/--end-date` 连续区间回放。
- **Simulation Infrastructure**: 新增 `SimulationRunner`、`SimTrader`、`NavTracker`、`StrategyConfig`、`MultiSimRunner` 等模块，完善单策略/多策略模拟盘框架。
- **Take-Profit Alignment**: TP 规则统一为 `prev_close*(1+tp)`，跳空按 `open`、盘中按 `tp_trigger`；交易历史支持 `SELL-TP` 标记。
- **Verify Alignment Fix**: `verify_strategy` 修复为“交易后净值记当日收益”，并将 TP 与手续费纳入当日收益口径。
- **Rebalance Alignment**: 模拟盘复用 `CBRebalancer` 与 `Top-K + valid_mask` 约束，提升与回测/验证的一致性。
- **Latest Alignment Metrics**:
  - 2024 区间：`Corr≈0.999`，`MAE≈5e-4`
  - 2025 区间：`Corr≈0.999`，`MAE≈4e-4`

### **V4.2: Alpha Efficiency (Current)**
*姝㈢泩閫昏緫涓庤瘎浠蜂綋绯讳紭鍖栥€?
- **Vectorized Take-Profit**: 鍏ㄩ潰鍚戦噺鍖栨鐩堥€昏緫锛屽ぇ骞呮彁鍗囪缁冮€熷害锛涙敮鎸佸紑鐩樿烦绌轰笌鐩樹腑姝㈢泩銆?
- **Simplified Buy-Back**: 閿佸畾姝㈢泩褰撴棩鏀剁泭锛屼粎瀵逛粛鍦?Top-K 鐨勬爣鐨勮绠楅澶栦拱鍥炴垚鏈紝骞宠　鏀剁泭涓庢崲鎵嬨€?
- **Strict Price Filter**: 寮曞叆浠锋牸鏈夋晥鎬ф鏌?(0 < Price < 10000)锛屽交搴曟秷闄よ剰鏁版嵁寮曡捣鐨勬敹鐩婄巼姹℃煋銆?
- **IC/IR Alignment**: 淇鍥犲瓙璇勪环瀵归綈鍙ｅ緞锛屽鏃犳晥鏍锋湰杩斿洖 None 浠ラ槻姝?RL 鎸囨爣鍋忕疆銆?

### **V4.1: Engineering Hardening**
*宸ョ▼鍔犲浐涓庡彛寰勭湡瀹炴€э細纭繚姣忎竴鍒?Alpha 閮界粡寰楄捣鎺ㄦ暡銆?
- **Hierarchy of Failure**: 寤虹珛 (`EXEC` < `STRUCT` < `LOWVAR` < `METRIC` < `SIM`) 鎯╃綒闃舵锛屾彁渚涙竻鏅扮殑寮哄寲瀛︿範姊害銆?
- **Grammar-Guided Decoding**: 鍒╃敤 Action Masking 鎶€鏈皢鏃犳晥璇硶鐢熸垚鐜囦粠 99% 闄嶈嚦 **0%**銆?
- **Rolling Window Controller**: 寮曞叆 10姝ユ粦鍔ㄧ獥鍙ｇ喌鎺у埗锛屽苟缁撳悎 **3-Level Success (Hard/Metric/Sim)** 鍙ｅ緞锛屽交搴曟秷闄よ缁冮渿鑽°€?
- **SimPass 2.0**: 閲嶆柊瀹氫箟澶氭牱鎬ф垚鍔熸爣鍑嗭紝璁ゅ彲鈥滀紭鑳滃姡姹扳€濈殑鏇挎崲琛屼负 (`SIM_REPLACE`)銆?

### **V3.5: Efficiency**
*鏁堢巼闈╁懡锛氳В鍐崇浉浼兼€у啑浣欍€?
- **Safe LRU Cache**: 100k涓婇檺纭畾鎬х紦瀛橈紝娑堥櫎鍐椾綑鍥炴祴銆?
- **Adaptive Entropy 1.0**: 鍒濇寮曞叆鑷€傚簲鐔垫帶鍒躲€?

### **V3.4: Grammar-Guided Decoding**
*寮曞叆绠楀瓙璇硶绾︽潫锛屽交搴曡В鍐虫棤鏁堝叕寮忕敓鎴愰棶棰樸€?

### **V3.3: Long-Run Optimization**
*Optimized for long-duration training stability and diversity.*
- **Training Stability**: Adjusted default `Entropy Beta` and `Train Steps` (2000) to prevent premature convergence.
- **Documentation**: Added "Playbook-style" tuning comments in `default_config.yaml`.
- **Dynamic Config**: `TRAIN_STEPS` is now dynamically configurable via YAML.

### **V3.2: Dynamic Configuration**
*Refactored configuration architecture for flexibility and ease of use.*
- **Dynamic Loading**:
    - 灏?`INPUT_FEATURES` 鍜?`RobustConfig` 绉昏嚦澶栭儴 YAML 閰嶇疆鏂囦欢 (`default_config.yaml`)銆?
    - 鏀寔閫氳繃鍛戒护琛屽弬鏁?`--config` 鍔犺浇鑷畾涔夐厤缃枃浠讹紝鏃犻渶淇敼浠ｇ爜銆?
- **CLI Support**: `engine.py` 鏀寔 `--config` 鍙傛暟銆?
- **Backward Compatibility**: `config.py` 浣跨敤 Metaclass 淇濇寔 API 鍏煎鎬э紝纭繚鏃т唬鐮佹棤缂濊繍琛屻€?

### **V3.1: Diversity & Anti-Stacking**
*Implemented advanced mechanisms to prevent "Formula Stacking" and boost diversity.*
- **Local Density Check**:
    - 鍦?`Formula Validator` 涓紩鍏ユ粦鍔ㄧ獥鍙ｅ瘑搴︽娴?(`Window=6, MaxTS=3`)銆?
    - 鏈夋晥鎵撳嚮鍒╃敤 `TS_MEAN` 鍫嗗彔鍒烽珮 Sharpe 鐨勮涓猴紝鍚屾椂淇濇姢浜嗗悎娉曠殑澶氬洜瀛愰€昏緫銆?
- **Enhanced Exploration**:
    - **Entropy Regularization**: 寮曞叆绾挎€ц“鍑忕殑鐔垫鍒欓」 (`Beta: 0.04 -> 0.005`)锛屽湪璁粌鍒濇湡寮哄埗妯″瀷鎺㈢储鏈煡棰嗗煙銆?
    - **Diversity Pool**: 缁存姢涓€涓?Top-50 澶氭牱鎬ф睜锛屽熀浜?**Jaccard Similarity** (闃堝€?0.8) 杩囨护鍚岃川鍖栧叕寮忋€?
- **Outcome**: 鎴愬姛鎸栨帢鍑?`DBLOW` (鍙屼綆), `VOL_STK` (姝ｈ偂娉㈠姩), `PREM` (婧环鐜? 绛夊绉嶄笉鍚岄€昏緫鐨勯珮鎬ц兘鍥犲瓙銆?

### **V3.0: CB Migration & Perfect Simulation**
*Completed migration from Crypto to Convertible Bonds and achieved pixel-perfect simulation alignment.*
- **System Migration**:
    - 鍏ㄩ潰閫傞厤鍙浆鍊?(CB) 甯傚満鐗规€э紙T+0, 娑ㄨ穼骞? 鏁存墜浜ゆ槗, 鍊哄埜灞炴€э級銆?
    - 鍥犲瓙宸ュ巶涓庣壒寰佸伐绋嬮拡瀵?CB 缁撴瀯杩涜浜嗛噸鏋勩€?
- **Perfect Verification**:
    - **Event-Driven Simulation** 涓?**Vector Backtest** 杈惧埌 >99% 鐩稿叧鎬?(Correlation)銆?
    - 淇浜嗘寔浠撶郴缁?鍙岃建鍒? Bug锛屽交搴曟秷闄や簡妯℃嫙璇樊銆?
    - 楠岃瘉浜嗙瓥鐣ュ湪 2023-2026 鍥涘勾闂寸殑绋冲仴琛ㄧ幇 (骞村寲 30%~60%)銆?
- **Simulation Suite**:
    - 鍗囩骇 `verify_strategy.py` 鏀寔鍛戒护琛屽弬鏁伴厤缃紙鍒濆璧勯噾銆佹椂闂村尯闂达級銆?
    - 澧炲己浜ゆ槗鏃ュ織锛屾敮鎸佸悕绉版樉绀轰笌璇︾粏涓€鑷存€ф鏌?(Return MAE, Jaccard)銆?
- **Configuration**:
    - 璐圭巼涓嬭皟鑷冲疄鐩樻按骞?(涓囦簲, 0.0005)锛岄噴鏀句簡楂橀绛栫暐娼滃姏銆?

### **V2.3: Formula Structure Validator**
*闃叉杩涘寲鎼滅储浜х敓"鍒嗘暟鍧嶇缉"鐨勫叕寮忕粨鏋勶紝骞跺寮烘敹鐩?椋庨櫓骞宠　銆?
- **Hard Filters**: 鐩存帴鎷掔粷宸茬煡鏈夊搴忓垪 (濡?`SIGN 鈫?LOG`)锛岄伩鍏嶆墍鏈夎祫浜у緱鍒嗙浉鍚屻€?
- **Soft Penalties**: 瀵瑰彲鐤戠粨鏋勶紙濡傝繛缁?LOG銆乀S_* 杩囧锛夎繘琛屾墸鍒嗚€岄潪鐩存帴鎷掔粷銆?
- **Documentation**: 鏂板 `docs/dangerous_structures.md` 璁板綍鎵€鏈夎鍒欍€?
- **Integration**: 鍦?`_worker_eval` 鍥炴祴鍓嶈繘琛岄獙璇侊紝鍑忓皯鏃犳晥璁＄畻銆?
- **Fee Rate Management**: 浜ゆ槗璐圭巼缁熶竴鐢?`RobustConfig.FEE_RATE` 绠＄悊 (0.001锛屽崈鍒嗕箣涓€)銆?
- **Return Reward**: 璇勫垎浣撶郴鏂板**骞村寲鏀剁泭鐜囧鍔?* (`RET_W = 5.0`)锛岄紦鍔辨寲鎺橀珮鍥炴姤绛栫暐銆?

### **V2.2: Performance & Operators**
*淇闅忕潃璁粌杩涜瀵艰嚧鐨勬€ц兘琛板噺锛屽苟鎵╁厖绠楀瓙搴撱€?
- **CPU Thread Fix**: 寮哄埗鍗曡繘绋嬫ā寮忎笅 Worker 浣跨敤鍗曠嚎绋?(`torch.set_num_threads(1)`)锛岃В鍐?CPU Oversubscription 闂銆?
- **I/O Optimization**: 寮曞叆 `MIN_SCORE_IMPROVEMENT` 闃堝€硷紝鍑忓皯鏃犳晥 I/O銆?
- **New Operator**: 鏂板 `TS_BIAS5` (5鏃ヤ箹绂荤巼) 绠楀瓙銆?

### **V2.1: Robustness Enhanced**
*涓撴敞浜庢寲鎺?绋冲仴銆佸彲瀹炵洏"鐨勫洜瀛愶紝鑰岄潪鍗曠函鐨勯珮澶忔櫘銆?
- **Split Validation**: 寮哄埗杩涜 **Train/Test 鍒嗘楠岃瘉**銆傝嫢 Valid 娈佃〃鐜板樊锛岀洿鎺ユ窐姹般€?
- **Rolling Stability**: 寮曞叆 **Mean - k*Std** 璇勫垎鏈哄埗锛屾儵缃氭敹鐩婃洸绾垮墽鐑堟尝鍔ㄧ殑鍥犲瓙銆?
- **Drawdown Control**: 鏄惧紡鎯╃綒鏈€澶у洖鎾?(MDD)銆?
- **Tradability Constraints**:
    - **Active Ratio**: 鍓旈櫎鍥犲仠鐗屾垨鏁版嵁缂哄け瀵艰嚧鏃犳硶婊′粨鐨勫洜瀛愩€?
    - **Valid Days**: 鍓旈櫎鏍锋湰閲忎笉瓒崇殑鍋剁劧楂樺垎鍥犲瓙銆?
- **Composite Score**: 缁煎悎 `(Train+Val) * Stability - MDD` 鐨勫缁磋瘎鍒嗕綋绯汇€?

### **V1.0: Foundation (No Lookahead)**
*涓撴敞浜庢秷闄?鏈潵鍑芥暟"涓?鏁版嵁娉勯湶"銆?
- **Strict Causality**: 鎵€鏈夌畻瀛?(`TS_DELAY`, `TS_MEAN`) 缁忎弗鏍煎璁★紝鏉滅粷 `torch.roll` 閫犳垚鐨勫惊鐜硠闇层€?
- **Robust Normalization**: 浣跨敤 `Rolling(60)` 杩涜鐗瑰緛鏍囧噯鍖栵紝鑰岄潪鍏ㄥ眬 Z-Score锛屽交搴曟秷闄ゅ叏鏍锋湰鍋忓樊銆?
- **Aligned Data Pipeline**: 缁熶竴 `CBDataLoader`锛岀‘淇濊缁冦€侀獙璇併€佸洖娴嬩娇鐢ㄥ畬鍏ㄤ竴鑷寸殑鏁版嵁娴?(`[Time, Assets]`)銆?

---

## 馃搳 Key Metrics Explained

| Metric | Definition | Good | Bad |
|--------|------------|------|-----|
| **Split Sharpe** | 璁粌闆嗕笌楠岃瘉闆嗙殑澶忔櫘姣旂巼 | Train涓嶸al鎺ヨ繎涓?1.5 | Val < 0 鎴?Train >> Val (Overfitting) |
| **Stability** | 婊氬姩澶忔櫘鍧囧€?- 1.5 * 鏍囧噯宸?| > 0.5 | < 0 (涓嶇ǔ瀹? |
| **Max Drawdown** | 鍘嗗彶鏈€澶у洖鎾?| < 20% | > 40% |
| **Active Ratio** | 瀹為檯鎸佷粨鏁?/ 鐩爣TopK | > 90% | < 50% (涓嶅彲浜ゆ槗) |

---

## 馃洜锔?Core Modules

### 1. AlphaGPT Model
- 涓€涓交閲忕骇 Transformer Decoder銆?
- 瀛︿範绠楀瓙璇硶 (`ADD`, `SUB`, `TS_DELAY`...) 鍜岀壒寰佺粍鍚堛€?
- 閫氳繃 RL (Policy Gradient) 浼樺寲锛屾牴鎹洖娴?Reward 璋冩暣鐢熸垚姒傜巼銆?

### 2. StackVM (Vectorized)
- 楂樻€ц兘 PyTorch 鍚戦噺鍖栨爤铏氭嫙鏈恒€?
- 鏀寔鏃跺簭绠楀瓙 (`TS_*`)銆佹í鎴潰绠楀瓙 (`CS_*`) 鍜岄€昏緫绠楀瓙 (`IF_POS`...)銆?
- **闆舵湭鏉ュ嚱鏁拌璁?*銆?

### 3. CBBacktest (Robust)
- Top-K 杞姩绛栫暐鍥炴祴鍣ㄣ€?
- 鏀寔 `Transaction Fee` 鍜?`Turnover` 鎯╃綒銆?
- 鍐呯疆 V2.1 绋冲仴鎬ц瘎浼版寚鏍?(`Split Sharpe`, `Stability`, `Active Ratio`)銆?

---

## 鈿?Usage

### 1. Training (Mining)
鍚姩澶氳繘绋嬫寲鎺橈細
```bash
python -m model_core.engine
```
*鑷姩淇濆瓨琛ㄧ幇鏈€濂界殑 "Kings" 鍒?`model_core/verified_trades/` 鍜?`best_cb_formula.json`銆?

### 2. Verification
楠岃瘉鐗瑰畾鍥犲瓙锛堝 King #8锛夌殑绋冲仴鎬э細
```bash
python verify_kings.py --king 8
```
杈撳嚭绀轰緥锛?
```text
鉁?Backtest Result:
   Composite Score: 10.76
   Sharpe (Train/Val): 1.27 / 1.98  <-- 鍏抽敭鎸囨爣
   Max Drawdown: 14.9%
   Stability: -0.86
   Active Ratio: 100.0%
```

### 3. Real-time Simulation
妯℃嫙瀹炵洏閫夎偂锛堜娇鐢ㄦ渶杩?70 澶╂暟鎹級锛?
```bash
python verify_king8_realtime.py
```

---

## 馃摑 Configuration

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



