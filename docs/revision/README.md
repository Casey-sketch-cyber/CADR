# CADR 修订工作区

被拒后的修订计划。分工：**Gemini 实现与跑实验 → Claude 审核 PR 并合并 → 作者把关关键决策**。

| 文件 | 内容 |
|---|---|
| [REVIEW_2026-11-19.md](REVIEW_2026-11-19.md) | 被拒原因分析：设定可信度、年龄后验推导错误、缺失消融与基线、文献树、两条修改路线 |
| [TASKS.md](TASKS.md) | **执行 AI 从这里开始**：工作规则、分阶段计划（P0–P4）、任务 T1–T9 的规格与验收标准 |
| [DATA_RECOVERY.md](DATA_RECOVERY.md) | 服务器数据删除后的数据集、演示、上游代码、trace、checkpoint 重建步骤 |
| [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md) | Claude 审核 PR 时使用的检查清单 |
| [`scripts/revision/age_posterior_check.py`](../../scripts/revision/age_posterior_check.py) | 年龄后验验证仿真（CPU，几秒钟） |

## 当前状态（2026-11-19）

- [x] 问题分析与文献树（第一版）
- [ ] P1 代码：T1 风险率后验、T2 无删失消融、T3 oracle-age、T4 简单基线、T5 匹配先验、T8b 计时
- [ ] P0 数据：Hopper + PickCube 重建（服务器数据已删）
- [ ] P2 快速验证 → **决策关口**（路线 A 继续，还是转向路线 B）
- [ ] P3 完整矩阵与 trace
- [ ] P4 论文重写

## 待作者确认

1. 两条本地实测的 WiFi trace 原始数据能否找回（见 DATA_RECOVERY.md §5）。
2. 下一个目标期刊。
