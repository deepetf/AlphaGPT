# AlphaGPT Git 说明（V6.01）

## 1) 版本定位
- 发布版本：`V6.01`
- 发布日期：`2026-03-13`
- 主题：Hybrid 二次筛选相似度去重 + 默认策略刷新

## 2) 提交规范
- 分支命名：`feature/xxx` 或 `hotfix/xxx`
- 提交信息：`type(scope): 简述`
- 推荐本次提交信息：`feat(release): 发布 V6.01`

## 3) 建议发布步骤
```bash
git checkout -b feature/v6-straight-through
git add README.md Commands.MD GIT-RELEASE.md
git add model_core/default_config.yaml model_core/factor_ai_review.py model_core/select_top_factors.py model_core/top_factor_config.yaml
git add strategy_manager/strategies_config.json
git commit -m "feat(release): 发布 V6.01"
git tag -a v6.01 -m "AlphaGPT V6.01"
git push origin feature/v6-straight-through --tags
```

## 4) 本次发布核心文件
- 二次筛选主逻辑：`model_core/select_top_factors.py`
- AI review 报告：`model_core/factor_ai_review.py`
- 筛选配置：`model_core/top_factor_config.yaml`
- 默认训练基线：`model_core/default_config.yaml`
- 当前策略组合：`strategy_manager/strategies_config.json`
- 命令文档：`Commands.MD`
- 发布说明：`README.md`
