# AlphaGPT Git 说明（V6.0）

## 1) 版本定位
- 发布版本：`V6.0`
- 发布日期：`2026-03-12`
- 主题：Straight-Through 工作流（train → select → bundle → verify → sim）

## 2) 提交规范
- 分支命名：`feature/xxx` 或 `hotfix/xxx`
- 提交信息：`type(scope): 简述`
- 推荐本次提交信息：`feat(workflow): release V6.0 straight-through pipeline`

## 3) 建议发布步骤
```bash
git checkout -b feature/v6-straight-through
git add README.md CHANGELOG.md Commands.MD STP-PLAN.MD GIT-RELEASE.md
git add model_core/engine.py tests/verify_strategy.py strategy_manager/run_sim.py
git add workflow/__init__.py workflow/run_manifest.py workflow/bundle_builder.py workflow/bundle_loader.py workflow/pipeline.py workflow/pipeline_state.py
git commit -m "feat(workflow): release V6.0 straight-through pipeline"
git tag -a v6.0 -m "AlphaGPT V6.0"
git push origin feature/v6-straight-through --tags
```

## 4) 本次发布核心文件
- 工作流：`workflow/`
- 训练产物标准化：`model_core/engine.py`
- Verify bundle 接入：`tests/verify_strategy.py`
- Sim bundle 接入：`strategy_manager/run_sim.py`
- 命令文档：`Commands.MD`
- 方案文档：`STP-PLAN.MD`
