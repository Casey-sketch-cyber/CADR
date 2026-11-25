# CADR 修订任务说明（给执行 AI：当前为 Codex）

> 背景与理由见 [REVIEW_2026-11-19.md](REVIEW_2026-11-19.md)。审核标准见 [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md)。数据重建见 [DATA_RECOVERY.md](DATA_RECOVERY.md)。
> 执行 AI 实现与跑实验，Claude 负责审核 PR、决定合并。

## 0. 工作规则（必须遵守）

1. **不要直接推 `main`**。每个任务一个分支、一个 PR，命名 `<执行AI>/T<编号>-<短名>`，例如 `codex/T1-hazard-posterior`。
2. **保留原论文可复现性**：
   - 不修改、不删除 `configs/generated_formal_v1_*`、`configs/generated_ablation_v1_*`；
   - 新实验一律用新系列名 `formal_v2_*` / `ablation_v2_*`；
   - 原算法行为必须能通过配置项还原，例如 `age_posterior: "legacy_cdf"`。
3. **不提交数据、checkpoint、原始 trace**，`.gitignore` 已覆盖。汇总结果（小 CSV/JSON，单文件 < 1 MB）放在 `results/revision/<任务>/`。
4. 每个 PR 的描述必须包含：
   - 改了什么；
   - `pytest` 输出；
   - 实验类任务的结果表（均值、95% 区间、seed 数、episode 数）和对应结果文件路径；
   - 已知问题。
5. 新增代码必须有单元测试；已有测试必须继续通过：`python -m pytest tests -q`（缺依赖时可跳过的测试，要在 PR 里写明跳过原因）。
6. 遇到本说明没覆盖、需要改动算法设计的地方：**先在 PR 里提问或写设计说明，不要自行发挥**。

## 1. 分阶段计划

| 阶段 | 内容 | 需要数据/GPU | 通过条件 |
|---|---|---|---|
| P1 代码 | T1–T5、T8b 的实现 + 单元测试 | 否（CPU） | Claude 审核每个 PR |
| P0 恢复 | 按 DATA_RECOVERY.md 复制 Hopper + PickCube 的正式权重，重新下载对应数据并校验 SHA-256 | 是 | 全部哈希一致 |
| T0 复现关口 | 用恢复的权重和 **v1 原配置** 重跑 Hopper seed101、PickCube seed101，与原 `episodes.csv` 逐回合比对 | 是（只评测） | 回报或成功率逐回合一致；硬件不同时允许浮点级差异，但必须量化并报告 |
| P2 快速验证 | 在 Hopper（IQL）与 PickCube（DP）上跑 T6 的核心对照：先 2 个 seed，结果清楚后补到 5 个 | 是（只评测） | **决策关口**：Claude 根据结果决定继续路线 A，还是转向路线 B |
| P3 完整矩阵 | 10 个任务 × 5 个 seed + trace 回放（5 条 trace 都在）+ T7 | 是（只评测） | Claude 审核 |
| P4 论文 | 重写（Claude 主导，按 paperwrite skill） | — | — |

**本轮修订不重训任何模型**：作者本地保留了全部 155 个正式权重（10 个任务 × 5 个 seed，见 `handoff/checkpoint_index.csv`）。T1–T6 都只改部署期逻辑，直接加载冻结权重评测即可。这本身也符合论文“冻结策略、即插即用”的设定。不要重训；遇到确实需要训练的情况，先在 PR 里说明。

P1 不依赖数据，可以与 P0 并行。T0 不通过（旧配置跑不出旧结果），不要往下做。**P3 只在 P2 关口通过后才开始**。

## 2. 任务明细

### T0 旧结果复现关口【P0 之后，任何新实验之前】

- 用 `configs/generated_formal_v1_hopper_cadr_seed101.json` 和 `configs/generated_formal_v1_pickcube_synthetic_seed101.json`（交接文档 §5 有完整命令），加载恢复的权重，输出到新目录 `runs/recovery_*`。
- 与原 `artifacts/remote/formal_v1_{minari,maniskill}_synthetic/episodes.csv` 中对应任务、seed、方法、场景的逐回合结果比对，写成 `scripts/revision/compare_episodes.py`（输入两个 episodes.csv 和筛选条件，输出逐回合差异、最大绝对差、均值差）。
- 必须在 P1 代码合入**之前**跑一次，并在合入之后用 legacy 配置再跑一次，证明新代码不改变旧行为。

**2026-11-23 关口修订（Claude 审核 PR #9 后）**

第一次 T0（Colab L4 vs 归档 RTX 5090）的结果：
- 直接策略 240 回合逐位一致；
- CADR 只在需要多步动力学 rollout 的场景出现差异，固定时延 1/2、IID、无时延场景都逐位一致；
- 同一台 L4 上重复运行逐位一致；场景均值差的区间基本覆盖 0。

这符合“跨 GPU 架构的浮点舍入，经多步 rollout 和 Hopper 混沌动力学放大”的特征，不像代码或数据问题。跨硬件逐回合一致因此不再作为通过条件。关口改为以下三条：

1. **T0a 同硬件回归（硬条件）**：P1 合入前后，在同一台机器、同一环境下用 legacy 配置各跑一次，逐回合**逐位一致**。之后所有新旧方法的对比，都必须在同一硬件上新跑 legacy 作为对照，不和归档数字比较。
2. **T0b 数值敏感性对照（一次性）**：在同一台 L4 上，对 Hopper seed101 的 `drifting_to_long_delay` 和 `markov_switching`（各 20 回合），用 legacy 配置跑两次：
   - (a) 默认 float32；
   - (b) 只把 CADR 动力学集成的前向计算改成 float64（仅用于本对照，加配置开关，默认关闭）。
   用 `compare_episodes.py` 报告 (a) 与 (b) 的逐回合差异。如果差异量级与“归档 vs L4”相当（几十到几百的回报差，且均值差的区间覆盖 0），就确认原因是数值敏感性，T0 通过。如果 (a) 与 (b) 逐位一致或差异极小，说明另有原因，停下来报告。
3. **跨硬件分布一致（软条件）**：每个场景的均值差在 seed 内的 bootstrap 区间应覆盖 0，或者差值相对场景均值 < 5%；不满足的要单独列出。

PickCube 的固定时延 1 只有 2 个回合成功结果互换、总成功率不变，按同一原则处理。T0b 对 DP 不强制。

### T1 用风险率后验替换 Eq.(9)/(10)【P1，最高优先级】

**问题**：`controllers.py::_content_conditioned_age_probabilities` 在观测变化时，用 `estimator.effective_age_probabilities()`（无条件 CDF 乘积，对应论文 Eq.9）再截断。它没有用到一条已知信息：生成步 > G_t 的包在 t−1 之前都没有到达。

**正确后验**（ack_lag=1、观测在 t 时刻变化，K = t − G_t − 1）：

```
S(k) = Σ_{u≥k} q_t(u)                    # 生存函数 P(d ≥ k)
h(k) = q_t(k) / S(k)    (若 S(k)=0，令 h(k)=1)
w(k) = h(k) · Π_{j<k} (1 − h(j)),   k = 0..K
p_t(k) = w(k) / Σ_{u≤K} w(u)
```

含义：年龄为 k，等价于包 t−k 恰好在 t 时刻到达（已知它在 t−1 前没到），而且所有更新的包都没在 t 时刻到达。

**实现要求**：
1. 在 `delay_belief.py` 新增纯函数 `hazard_age_probabilities(delay_probabilities: np.ndarray, maximum_age: int) -> np.ndarray`。
2. 在 `controllers.py::CadrController._content_conditioned_age_probabilities` 中加配置项 `age_posterior`。CADR-DP 的 `maniskill_diffusion.py::CadrDiffusionPolicyController` 继承了这个方法，要确认配置项能从 `evaluate_maniskill_cadr_diffusion.py` 传进去。`official_dtcorl.py::OfficialDTCorlAckMapController`（第 173 行附近）也要支持同一配置项，保证 DT-ACK 基线用的是同一个后验，比较才公平。取值：
   - `"legacy_cdf"`：原论文行为；
   - `"hazard"`：新行为。
   - **代码默认值保持 `legacy_cdf`**，旧配置的结果不变；新的 v2 配置显式写 `"hazard"`。
3. 只替换“观测变化”分支；`t=0` 分支和“观测未变”分支保持不变。
4. `use_content_conditioning=False` 时仍走旧的无条件公式（无 content 时 G_t 约束不成立），在代码注释里说明。

**测试** `tests/test_age_posterior_hazard.py`：
- 基本性质：和为 1；支撑在 0..K；固定时延 d 时为点质量 d；K=0 时为 [1]。
- 统计检验：复用 `scripts/revision/age_posterior_check.py` 的仿真逻辑，给定真实 q，在 {0,1,2}、{3,5,7}、{8,12,16} 三种分布下，hazard 后验与经验分布的 TV < 0.02（样本量可降到 5 万步以控制测试时长）。
- 回归测试：`legacy_cdf` 的输出与修改前逐位一致。

### T2 “不用删失”消融【P1】

- `CensoredAckDelayEstimator` 加参数 `use_censoring: bool = True`。为 False 时跳过所有未收到 ACK 的记录，只用精确时延 + 先验。
- `prepare_cadr_ablation_config.py` 的 `SUPPORTED_VARIANTS` 加 `"no_censoring"`。
- 测试：时延突然变长后的前若干步，`use_censoring=False` 的均值估计应明显偏短（构造一个 changepoint 序列来断言）。

### T3 oracle-age CADR 基线【P1】

- 新方法名 `cadr_oracle_age`：用信道真实的 `selected_packet.generated_step` 算出真实年龄，构成点质量；动力学 rollout、策略接口与 CADR 完全相同。
- **特权信息仅供评估**：在方法名、结果表和代码注释里标注 privileged。
- 用途：拆分误差。
  - CADR 与 oracle-age 的差距 = 年龄推断误差；
  - oracle-age 与无时延表现的差距 = 动力学模型误差。

### T4 简单基线【P1】

1. `in_order_age`：观测未变时用精确年龄；观测变化时假设新到的是 G_t+1，即 age = t − G_t − 1。用到了 ACK 和 content，**完全不用时延统计**。这是检验“时延分布估计到底有没有用”的关键对照。
2. `fixed_horizon_model`：不做年龄推断，固定预测步长 H，用同一动力学模型。H 取参考分布均值（四舍五入）；另做一组取 H = 最大支撑。不用 ACK。
3. 两者都实现成 `CadrController` 的 `age_decision_rule` 新取值或独立 controller，复用 rollout 代码。

### T5 公平的固定先验对照【P1】

- 新消融 `fixed_prior_matched`：先验 q0 设为该场景的“变点前/参考”分布（IID 场景就用其本身），并且关闭在线更新。
- 原 `fixed_prior` 保留。两者一起报告，用来说明“在线自适应”的收益不是来自先验失配。

### T6 消融矩阵【P2 快速版 → P3 完整版】

任务：Hopper（IQL）、Walker2d（IQL，仅 P3）、PickCube（DP）。场景沿用 v1 的全部合成场景。

| 变体 | 说明 |
|---|---|
| full_hazard | 新的默认方法（T1） |
| full_legacy | 原论文方法 |
| no_censoring | T2 |
| no_content | 原有 |
| ack_map (hazard) | 原有 ack_map，但后验用 hazard |
| fixed_prior / fixed_prior_matched | 原有 / T5 |
| single_model | 原有 |
| in_order_age | T4 |
| fixed_horizon_model | T4 |
| cadr_oracle_age | T3（privileged） |
| direct | 冻结策略直接吃时延观测 |
| replan_every_step | 仅 PickCube（原有） |

IQL 额外比较三种聚合方式，新增配置项 `aggregation: action_mean | state_mean | map`。

single_model 的做法：
- PickCube 有专门训练的单模型权重（`ablation_v1_dp_dynamics_single_model_pickcube_*`），沿用原做法。
- Hopper/Walker2d 没有单独训练的单模型。用部署期配置 `ensemble_members: [0]`，只用集成里的第 0 个成员，并在结果表中注明这和 PickCube 的做法不同。
- P2：2 个 seed × 20 episode。
- P3：5 个 seed × 20 episode，沿用论文的配对设计与层次 bootstrap。

**P2 需要回答的问题**（写进 PR 描述）：
1. full_hazard 相对 full_legacy 是否显著更好，在哪些场景？
2. full_hazard 相对 in_order_age 是否显著更好？如果不显著，说明时延统计的价值有限，会触发路线 B 讨论。
3. full_hazard 相对 no_censoring 的差距主要出现在哪些场景（预期：changepoint / drift 刚变长的阶段）？
4. 与 oracle_age 的差距有多大？

### T7 随机 ACK 回传通道【P3，先写设计再写代码】

- 目标：ACK 滞后 ℓ_g 随机（可设成与前向时延同分布，或用 trace 的反向半 RTT），并允许 ACK 以概率 p 丢失。
- 此时“观测未变 ⇒ 年龄精确”和 G_t 的完整性都不再成立，后验需要推广成对“当前所选包生成步”的递推贝叶斯滤波。
- **先提交 `docs/revision/notes/T7_design.md`**（推导 + 近似方案 + 复杂度），Claude 审核通过后再写代码。
- trace 回放：ACK 走反向路径（同一 trace 的半 RTT），不再固定 1 步。

### T8 报告与开销【T8b 在 P1，其余在 P3】

- T8a：所有结果表同时给出绝对分数、无时延参考、D4RL normalized score（locomotion/Adroit），以及恢复率 = (方法 − direct) / (clean − direct)。
- T8b：在 `evaluate_checkpoints.py` 和 DP 评估中记录每步 controller 的 wall-clock 时间（GPU 同步后计时）；报告在 Dmax=32、M=5 下的中位数和 p95。
- T8c：把“无时延一致性”从实验亮点改为命题，代码里旁路分支的存在要在论文中说明（P4 写作时处理）。

### T9 路线 B 探索（不写代码）【P2 关口之后视情况】

只在 P2 关口判定需要时，写一页 `docs/revision/notes/routeB_proposal.md`：
- 设定 1：动作通道时延 + 执行端 ACK；
- 设定 2：有时间戳但 action chunk 期间未来年龄未知。

### T10 偏向陈旧的年龄决策（作者 2026-11-25 确认的新主线）

**动机**（见 `notes/P2a_decision.md`）：后验推断的年龄比“按顺序到达”准，但出错时偏向低估年龄；低估年龄的代价远大于高估。年龄决策应偏向“往老了估”。

**方法**：
- 新增 `age_decision_rule="upper_quantile"`，配置项 `age_quantile = α ∈ (0, 1]`。
- 在内容约束后的风险率后验 p_t（观测变化步；观测未变时仍用精确年龄）上，取最小的 k 使累积概率 Σ_{j≤k} p_t(j) ≥ α，记为 k_α。之后用点质量 k_α 做单一年龄的 rollout。成员的合并方式与各路径现有的 MAP 相同：IQL 对 M 个成员的 actor 动作取平均；DP 先对成员状态加权平均，再调用一次扩散模型。
- 理论依据：两种误差代价成线性不对称时，期望损失最小的点估计是分位数，α = c_低估 / (c_低估 + c_高估)。
- **α=1 时退化为“按顺序到达”**：先验让 q 处处为正，所以 k_1 = K。这必须用单元测试证明逐位相同，论文中也要把“按顺序到达”写成这一族规则在 α=1 时的特例。

**实现（T10-code，可立即开始，与 P2b-1 并行）**：
- 在 IQL 和 DP 两条路径上都实现；`aggregation` 只允许 `action_mean`。
- 单元测试至少包括：
  - 分位数的边界情况（α 恰好等于某个累积值、K=0）；
  - α=1 时与 `in_order_age` 的动作逐位相同；
  - legacy 默认路径不变。
- 方法名 `upper_quantile_a{α×100}`，例如 `upper_quantile_a90`。

**实验（T10-run，P2b-1 预测成立后才开始）**：
1. **选参（seed 101、102，Hopper，全部 12 个场景，各 20 回合）**：α ∈ {0.5, 0.75, 0.9, 0.95}。选择规则预先固定为：取在 4 个重点场景上“规则 − in_order_age”配对差值均值最大的 α，若持平取较大的 α。选定后写进 PR，**之后不再改动**。
2. **留出检验（seed 103、104、105，Hopper）**：用选定的 α，与 `in_order_age`、`full_hazard`、`full_legacy`、`cadr_oracle_age`、`direct` 配对比较，同一类 GPU，各 20 回合。
   预注册通过标准：在 4 个重点场景中，至少两个场景的“规则 − in_order_age”区间为正；在全部 12 个场景中，没有明显为负的场景。
   3 个 seed 时区间会偏宽，如实报告。另外列一张 5 个 seed 的合并表，**标明 101/102 用于选参**。
3. **跨任务泛化（P3）**：不重新选 α，直接在 Walker2d、HalfCheetah 上用同一个 α 评测。PickCube 先提出有分辨力的指标或场景（例如 `success_at_end`、成功占比，或更难的任务），交 Claude 审核。

**失败时的处理**：P2b-1 的预测不成立，或留出检验未通过，都停下来如实报告，由作者和 Claude 重新决定方向，**不要临时调整 α 网格或通过标准**。

## 3. 完成后的交付清单

- [ ] T1–T5、T8b 的 PR 已合并（P1）
- [ ] Hopper + PickCube 权重复制、数据下载并通过 SHA-256 校验，记录在 `results/revision/P0/env_record.md`（P0）
- [ ] T0 复现比对报告 `results/revision/T0/reproduction.md`，含逐回合差异统计（T0）
- [ ] P2 结果表 + 四个问题的回答（P2）
- [ ] 完整矩阵 + trace + T7（P3）
