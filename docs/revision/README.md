# CADR 修订工作区

被拒后的修订计划。分工：**执行 AI（当前为 Codex）实现与跑实验 → Claude 审核 PR 并合并 → 作者把关关键决策**。

| 文件 | 内容 |
|---|---|
| [REVIEW_2026-11-19.md](REVIEW_2026-11-19.md) | 被拒原因分析：设定可信度、年龄后验推导错误、缺失消融与基线、文献树、两条修改路线 |
| [TASKS.md](TASKS.md) | **执行 AI 从这里开始**：工作规则、分阶段计划（P0–P4）、任务 T1–T9 的规格与验收标准 |
| [DATA_RECOVERY.md](DATA_RECOVERY.md) | 恢复方案：权重和 trace 全在本地，只需重新下载 Minari 数据和 ManiSkill 演示 |
| [handoff/](handoff/) | 作者本地的恢复交接说明，以及权重和数据的逐文件索引（路径、字节数、SHA-256） |
| [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md) | Claude 审核 PR 时使用的检查清单 |
| [`scripts/revision/age_posterior_check.py`](../../scripts/revision/age_posterior_check.py) | 年龄后验验证仿真（CPU，几秒钟） |

## 当前状态（2026-11-24）

- [x] 问题分析与文献树（第一版）
- [x] P0 恢复（seed 101）：Hopper 和 PickCube 的权重与数据 SHA-256 全部一致
- [x] **P1 代码全部合并**：T2（#4）、T8b（#7）、T1（#3）、T5（#6）、T3（#5）、T4（#8）。合并后的 main 与执行 AI 测试过的堆叠末端逐文件一致（101 passed）
- [x] T0（#9）+ **T0b 结论通过**：同一张 T4 卡上，只把动力学从 float32 换成 float64，CADR 20/20 回合都变，最大差 547 / 88（跨 GPU 时是 606 / 321），均值差区间覆盖 0，直接 IQL 逐位不变。跨硬件差异确认来自数值敏感性。T0b 的代码（#11）待 rebase 后合并
- [ ] **T0a**：同一硬件上，P1 合入前（`33d104e`）与合入后（当前 main）的 legacy 配置逐回合逐位一致
- [ ] T6 设计 → P2 快速验证 → **决策关口**（路线 A 继续，还是转向路线 B）
- [ ] P3 完整矩阵与 trace
- [ ] P4 论文重写

## 执行 AI 当前待办（Claude 2026-11-24 审核结论）

1. **#11（T0b）rebase 到最新 main**：与 T3/T4 在 `evaluate_checkpoints.py` 有冲突。保持 `diagnostic_float64_dynamics` 默认关闭，跑完整 pytest，并在 legacy 默认路径上做逐位回归。
2. **T0a**：在**同一类 GPU、同一会话**内，分别用 `33d104e` 和当前 main，以 v1 配置跑 Hopper seed101 与 PickCube seed101（与 T0 相同的场景和回合），用 `compare_episodes.py` 比对，要求逐回合逐位一致。结果写到 `results/revision/T0a/`。
3. **硬件规则（因 Colab 分配的卡型会变）**：P2/P3 中**同一个比较批次**（同一任务、seed 下的全部方法，含新跑的 legacy 对照）必须在同一类 GPU 上跑完，结果里记录 GPU 型号。不同批次可以用不同卡型，但不同卡型的结果不混在同一个配对比较里。
4. **T6 先交设计、再跑实验**：提交 `docs/revision/notes/T6_design.md`，Claude 审核后再开跑。内容包括：
   - P2 的配置清单（Hopper IQL + PickCube DP，seed 101/102，TASKS.md T6 表中的全部变体）；
   - 方法名与输出目录约定；
   - 每个配置的回合数与预计 GPU 时长；
   - 结果表模板：绝对分数、无时延参考、恢复率、配对差值及 95% 层次 bootstrap 区间；
   - 准备怎么回答 TASKS.md 中 P2 的四个问题。
5. **关于 `fixed_horizon_model` 的 H**：名义分布 IID {0,1,2}:[0.65,0.25,0.10] 的均值是 0.45，四舍五入得 H=0，这就是直接策略本身。上次审核写的“H_mean=1”是 Claude 的算术错误。现确认采用 T4 的实现：均值规则取 H=max(1, round(均值))=1，即“最小非平凡步长”；max_support 规则取 H=2。论文中按“H=0 等价于直接策略，因此取最小非平凡步长 1”来写。

## 待作者确认

1. 下一个目标期刊（上一次投的是 Applied Soft Computing）。
2. 执行 AI 在哪台机器上运行。当前 P0/T0 用的是 Colab L4。**之后 P2/P3 的所有对比要固定在同一类硬件上**，换机器就要重做 T0a。
