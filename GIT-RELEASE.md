# AlphaGPT Git 说明（V6.02）

## 1) 版本定位
- 发布版本：`V6.02`
- 发布日期：`2026-03-16`
- 主题：结构化 AI review + Select 兼容接口接入

## 2) 提交规范
- 分支命名：`feature/xxx` 或 `hotfix/xxx`
- 提交信息：`type(scope): 简述`
- 推荐本次提交信息：`feat(release): 发布 V6.02`

## 3) 建议发布步骤
```bash
git checkout -b feature/v6-structured-ai-review
git add README.md Commands.MD GIT-RELEASE.md
git add model_core/factor_ai_review.py model_core/formula_simplifier.py model_core/select_top_factors.py model_core/top_factor_config.yaml
git add tests/test_factor_ai_review.py tests/test_factor_ai_review_integration.py
git commit -m "feat(release): 发布 V6.02"
git tag -a v6.02 -m "AlphaGPT V6.02"
git push origin feature/v6-structured-ai-review --tags
```

## 4) 本次发布核心文件
- AI review 主逻辑：`model_core/factor_ai_review.py`
- 公式展开与结构提示：`model_core/formula_simplifier.py`
- 二次筛选接线：`model_core/select_top_factors.py`
- AI review/筛选模板：`model_core/top_factor_config.yaml`
- 单元测试：`tests/test_factor_ai_review.py`
- 真实接口集成脚本：`tests/test_factor_ai_review_integration.py`
- 命令文档：`Commands.MD`
- 发布说明：`README.md`
