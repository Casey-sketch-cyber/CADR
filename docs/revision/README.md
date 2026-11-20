# CADR 修订工作区

被拒后的修订计划。分工：**Gemini 实现与跑实验 → Claude 审核 PR 并合并 → 作者把关关键决策**。

| 文件 | 内容 |
|---|---|
| [REVIEW_2026-11-19.md](REVIEW_2026-11-19.md) | 被拒原因分析：设定可信度、年龄后验推导错误、缺失消融与基线、文献树、两条修改路线 |
| [TASKS.md](TASKS.md) | **执行 AI 从这里开始**：工作规则、分阶段计划（P0–P4）、任务 T1–T9 的规格与验收标准 |
| [DATA_RECOVERY.md](DATA_RECOVERY.md) | 恢复方案：权重和 trace 全在本地，只需重新下载 Minari 数据和 ManiSkill 演示 |
| [handoff/](handoff/) | 作者本地的恢复交接说明，以及权重和数据的逐文件索引（路径、字节数、SHA-256） |
| [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md) | Claude 审核 PR 时使用的检查清单 |
| [`scripts/revision/age_posterior_check.py`](../../scripts/revision/age_posterior_check.py) | 年龄后验验证仿真（CPU，几秒钟） |

## 当前状态（2026-11-19）

- [x] 问题分析与文献树（第一版）
- [ ] P1 代码：T1 风险率后验、T2 无删失消融、T3 oracle-age、T4 简单基线、T5 匹配先验、T8b 计时
- [ ] P0 恢复：复制 Hopper + PickCube 的权重，重新下载数据并校验（**不重训**，155 个正式权重和 5 条 trace 都在作者本地）
- [ ] T0 复现关口：v1 配置重跑，与原 episodes.csv 逐回合一致
- [ ] P2 快速验证 → **决策关口**（路线 A 继续，还是转向路线 B）
- [ ] P3 完整矩阵与 trace
- [ ] P4 论文重写

## 待作者确认

1. 下一个目标期刊（上一次投的是 Applied Soft Computing）。
2. 执行 AI 在哪台机器上运行：本地 Windows 项目，还是新服务器？在新服务器上运行的话，需要作者把 P0 所需的权重复制过去。
