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

## 当前状态（2026-11-23）

- [x] 问题分析与文献树（第一版）
- [x] P0 恢复（seed 101，Colab L4）：Hopper 和 PickCube 的权重与数据 SHA-256 全部一致
- [x] 已合并：T2 无删失消融（#4）、T8b 计时（#7）、T0 首次复现报告与 `compare_episodes.py`（#9）
- [ ] **待 rebase 后合并**：T1（#3）→ T5（#6）→ T3（#5）→ T4（#8），代码审核均已通过，见下方“执行 AI 当前待办”
- [ ] T0 关口：首次跨硬件比对未逐位一致，判断为数值敏感性所致，关口已按 TASKS.md 修订；需完成 T0a + T0b
- [ ] P2 快速验证 → **决策关口**（路线 A 继续，还是转向路线 B）
- [ ] P3 完整矩阵与 trace
- [ ] P4 论文重写

## 执行 AI 当前待办（Claude 2026-11-23 审核结论）

T1/T5/T3/T4 都从同一个旧 `main` 分出，彼此在 `controllers.py`、`delay_belief.py`、评测入口上冲突。Claude 的环境没有 PyTorch，无法验证冲突解决，所以由执行 AI 负责 rebase。

1. **按顺序做成堆叠 PR**：
   - T1 rebase 到最新 `main`（已含 T2、T8b）；
   - T5 rebase 到 T1 分支上，PR 的 base 设为 T1 分支；
   - T3 叠在 T5 上，T4 叠在 T3 上。
   每个分支都要跑完整的 `python -m pytest tests -q`，并重新确认 legacy 默认路径逐位不变（T1 已有回归测试，T2 的 `use_censoring` 也要一起覆盖）。PR 保持 draft 也可以，Claude 审核后会改为 ready 再合并。
2. **T1 附加修改（必须）**：
   - `hazard_age_probabilities` 遇到 [0, K] 上 q 的质量为 0 时，不要抛异常。改为退回 [0, K] 上的均匀分布（与 legacy 分支的回退一致），并记录诊断 `hazard_zero_mass_fallback`。
   - 原因：T5 的匹配先验允许 q 有零项，两者联用时理论上会在评测中途崩溃。
   - 补一个测试：零质量回退；再补一个 `fixed_prior_matched` + `hazard` 在 changepoint 场景下完整跑一个短回合不报错的测试（放在 T5 分支上）。
3. **T1 可选**：在 docstring 里注明，风险率乘积会逐项相消，h(k)·Π_{j<k}(1−h(j)) = q(k)/S(0)，所以后验等于“q 截断到 [0, K] 再归一化”（Claude 已用 2000 个随机分布验证，最大误差 2e-16）。实现不用改，论文要用这个更简洁的表述。
4. **T3、T4**：rebase 时，`cadr_oracle_age`、`in_order_age`、`fixed_horizon_model` 的 `evaluate_controller` 调用也要传 `controller_timing_device`（T8b 已合并）。
5. **T4 的 T6 配置约定**：`fixed_horizon_model` 的主基线，所有场景统一用同一个名义参考分布 IID {0,1,2}:[0.65,0.25,0.10]（H_mean=1，H_max=2），因为部署时不知道是哪个场景。如果另做“按场景真实分布取 H”的变体，方法名要加 `_scenario_tuned`，结果表标注 privileged。
6. **T0a + T0b**：见 TASKS.md 的“2026-11-23 关口修订”。T0b 可以和 rebase 并行做；T0a 在 T1–T4 全部合并后做。

## 待作者确认

1. 下一个目标期刊（上一次投的是 Applied Soft Computing）。
2. 执行 AI 在哪台机器上运行。当前 P0/T0 用的是 Colab L4。**之后 P2/P3 的所有对比要固定在同一类硬件上**，换机器就要重做 T0a。
