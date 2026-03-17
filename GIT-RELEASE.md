# AlphaGPT Git 说明（V7.0）

## 1) 版本定位
- 发布版本：`V7.0`
- 发布日期：`2026-03-17`
- 主题：未来数据泄露修复 + run 数据快照锁定 + CI 门禁

## 2) 提交规范
- 分支命名：`feature/xxx` 或 `hotfix/xxx`
- 提交信息：`type(scope): 简述`
- 推荐本次提交信息：`feat(release): 发布 V7.0`

## 3) 建议发布步骤
```bash
git checkout -b feature/v7-leakage-guardrails
git add README.md GIT-RELEASE.md .github/workflows/ci.yml .gitignore
git add model_core/data_loader.py model_core/engine.py model_core/factors.py model_core/features_registry.py model_core/ops_registry.py model_core/select_top_factors.py model_core/vm.py
git add data_pipeline/sql_strict_loader.py strategy_manager/cb_runner.py strategy_manager/sim_runner.py workflow/pipeline.py workflow/run_manifest.py
git add tests/test_stage_a_masks.py tests/test_stage_b_feature_validity.py tests/test_stage_b_ts_ops.py tests/test_vm_masked_cs.py tests/test_vm_ts_cs_validity.py tests/test_run_manifest_snapshot.py tests/test_pipeline_snapshot_lock.py tests/test_stage_e_guardrails.py
git commit -m "feat(release): 发布 V7.0"
git tag -a v7.0 -m "AlphaGPT V7.0"
git push origin feature/v7-leakage-guardrails --tags
```

## 4) 本次发布核心文件
- 数据加载与掩码语义：`model_core/data_loader.py`
- 因果特征与 TS 算子：`model_core/factors.py`、`model_core/ops_registry.py`
- VM 横截面执行：`model_core/vm.py`
- 主训练/筛选接线：`model_core/engine.py`、`model_core/select_top_factors.py`
- strict replay 加载与模拟口径：`data_pipeline/sql_strict_loader.py`、`strategy_manager/sim_runner.py`、`strategy_manager/cb_runner.py`
- run 快照锁定：`workflow/run_manifest.py`、`workflow/pipeline.py`
- 回归与 CI：`tests/test_stage_a_masks.py`、`tests/test_stage_b_feature_validity.py`、`tests/test_stage_b_ts_ops.py`、`tests/test_vm_masked_cs.py`、`tests/test_vm_ts_cs_validity.py`、`tests/test_run_manifest_snapshot.py`、`tests/test_pipeline_snapshot_lock.py`、`tests/test_stage_e_guardrails.py`、`.github/workflows/ci.yml`
- 发布说明：`README.md`
