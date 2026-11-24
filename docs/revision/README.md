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

## 当前状态（2026-11-24 下午）

- [x] 问题分析与文献树（第一版）
- [x] P0 恢复（seed 101）
- [x] P1 代码全部合并：T1、T2、T3、T4、T5、T8b
- [x] **T0 关口通过**：
  - T0b（#11）：同一张卡上只改动力学精度，差异量级与跨 GPU 时相当，确认是数值敏感性；
  - T0a（#14）：同一张 T4、同一会话，P1 合入前后 legacy 配置 1020/1020 回合逐位一致（Hopper 480，PickCube 540）
- [x] T6 设计（#13）已合并，**有条件批准**，修改见下方
- [x] T6-runner（#16）已合并：单方法入口、state_mean、ensemble_members、批量运行器、汇总脚本。已在 T4 上验证与旧入口逐位等价（24/24），无时延自检通过
- [ ] **P2a 快速验证（可以开跑）** → 决策关口
- [ ] P3 完整矩阵、trace 与其余消融
- [ ] P4 论文重写

## 执行 AI 当前待办（Claude 2026-11-24 傍晚审核结论，优先于下一节）

1. **P2a 可以开跑**，按 `results/revision/T6/p2a_plan.json` 的 7 个批次执行。决策只认 Hopper 5 个 seed 的结果，**这 5 个批次必须用同一类 GPU**，否则汇总脚本会判为数据不足。
2. **开跑的同时交一个小 PR `codex/T6-followups`**（不影响运行，但要在生成决策报告前合并）：
   - `assess_decision` 中“新后验不比旧后验显著差”这一条，`route_a_criteria_met` 要按**全部 12 个场景**判断，与 §D 的预注册一致；当前只检查了 4 个重点场景。
   - 把 `docs/revision/handoff/data_index.csv` 中 Hopper `metadata.json` 那一行恢复成交接原样。补算的 SHA-256 放到单独的补充索引（如 `docs/revision/handoff/supplemental_hashes.csv`），注明计算日期、来源文件和计算方式，`prepare_t6.py` 同时读取两份。交接文件是作者本地的原始记录，不要改动。
3. 跑完后提交 `results/revision/T6/` 的汇总（absolute、paired、timing、clean checks、report.md 和决策 JSON）。决策结论由 Claude 审核后再定，执行 AI 不要自行判断走哪条路线。

## 执行 AI 当前待办（Claude 2026-11-24 下午审核结论）

T6 设计整体批准：配对、目录、统计方法、对四个问题的回答方式都可以照做。以下几点优先于设计文档。

### A. 缩小 P2 规模：先跑能回答四个问题的部分（P2a）

原设计是 21–37 GPU 小时，P2 作为决策关口用不了这么多。改为：

| 批次 | 任务 | 方法 | 训练 seed |
|---|---|---|---|
| P2a-1 | Hopper | 设计中的全部 14 个变体 | 101, 102 |
| P2a-2 | Hopper | 核心 6 个：`full_hazard`、`full_legacy`、`no_censoring`、`in_order_age`、`cadr_oracle_age`、`direct` | 103, 104, 105（先按 DATA_RECOVERY 恢复并校验这 3 个 seed 的权重） |
| P2a-3 | PickCube | 同样的核心 6 个 | 101, 102 |

- 估计约 11–18 GPU 小时。PickCube 每个 seed 约 3–4.5 小时，可以在一个 Colab 会话内跑完同一卡型的一整批。
- Hopper 加到 5 个 seed 的理由：T0b 显示单回合结果对微小扰动非常敏感，只有 2 个 seed 时层次 bootstrap 几乎退化。Hopper 便宜，补 seed 的性价比最高。
- 推迟到关口之后（P3）：PickCube 的其余 7 个变体、`replan_every_step`（单这一项就要 7–11.5 小时，且与四个问题无关）、分段诊断。

### B. 开跑前先交一个代码 PR：`codex/T6-runner`

内容：`aggregation=state_mean`、Hopper 的 `ensemble_members`、按方法单独运行的入口（避免每个配置都重跑 direct），以及 §3 的批次清单。验收：
1. **单方法入口等价**：同一会话下，在 Hopper 和 PickCube 各取 2 个场景 × 3 个回合，新入口与现有入口逐回合逐位一致。
2. **MAP 复用获批**：`aggregation=map` 定义为“argmax 年龄，再对 M 个成员的 actor 动作取平均”，复用 `ack_map_hazard` 这一行。要用固定输入证明两者逐位相同；不另设“MAP 年龄下先平均状态”的实验。
3. **无时延一致性自检**：在 `clean_delay_0` 中，所有使用内容约束和 ACK 的变体（含 oracle）应与 direct 逐位一致。`fixed_horizon_*` 不读 ACK，预期不一致，属正常；`no_content` 如果不一致，如实记录原因。这项检查要写进结果报告。
4. PickCube 的原生崩溃（`gilstate_tss_set`）：每个（方法，seed）用单独进程跑，保留 `--resume`，记录崩溃和续跑次数。

### C. 统计口径补充

- 每个场景都列出各 seed 各自的差值。只有 2 个 seed 的 PickCube 结果只看方向，不下显著性结论。
- Hopper 主表用原始回报；附表加历史 D4RL 刻度的描述列，标注“按旧环境参考值重标度”。

### D. 决策标准（开跑前定下，避免事后挑场景）

判断依据是 Hopper 5 个 seed 的结果，PickCube 用来看方向是否一致。关注的场景：`drifting_to_long_delay`、`markov_switching`、`ood_support_shift`、`ood_changepoint`。

1. **路线 A 继续的条件**：在上述场景中，至少两个场景的 `full_hazard − in_order_age` 区间不跨零且为正，其余场景没有明显为负；同时 `full_hazard − full_legacy` 不显著为负。
2. **“删失”能否作为卖点**：`full_hazard − no_censoring` 在至少一个长时延场景区间为正，才在标题和贡献里保留“censored”；全部接近 0 的话，改写贡献表述。
3. **触发路线 B 讨论**：`full_hazard ≈ in_order_age` 在全部场景都成立，也就是说时延统计没有带来可辨的收益。
4. **oracle 差距**：`cadr_oracle_age − full_hazard` 仍很大，说明还有改进空间；接近 0 的话，主要瓶颈在动力学模型而不在年龄推断。这两种情况都如实写进论文。

## 待作者确认

1. 下一个目标期刊（上一次投的是 Applied Soft Computing）。
2. 执行 AI 在哪台机器上运行。当前 P0/T0 用的是 Colab L4。**之后 P2/P3 的所有对比要固定在同一类硬件上**，换机器就要重做 T0a。
