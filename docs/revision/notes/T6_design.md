# T6 · P2 快速验证设计

根据 `41e9ca6` 的 [Claude 审核结论](../README.md) 和 [TASKS.md 的 T6/T8](../TASKS.md) 制定。本文提交 Claude 审核。T0a 须先在同一 Colab 会话、同一硬件上通过，随后按审核批准的设计开展 P2。P3 的任务扩展、额外训练种子和时延轨迹回放由 P2 决策关口决定。

## 1. 任务、输入和配对

| 项目 | Hopper · IQL | PickCube · Diffusion Policy |
| --- | --- | --- |
| 原配置模板 | `configs/generated_formal_v1_hopper_cadr_seed{101,102}.json` | `configs/generated_formal_v1_pickcube_synthetic_seed{101,102}.json` |
| 数据与环境 | Minari `mujoco/hopper/medium-v0`，Hopper-v5，最多 1000 步 | `maniskill/PickCube-v1/motionplanning_state_official_dp_v3_0_1`，状态、`pd_ee_delta_pos`、`physx_cpu`、稀疏奖励、最多 100 步 |
| 策略文件 | `runs/formal_v1_iql_hopper_seed{seed}/checkpoint.pt` | `runs/formal_v1_dp_pickcube_seed{seed}/checkpoints/best_eval_success_once.pt` |
| 动力学文件 | `runs/formal_v1_cadr_hopper_seed{seed}/dynamics_checkpoint.pt` | `runs/formal_v1_dp_dynamics_pickcube_seed{seed}/dynamics_checkpoint.pt` |
| `single_model` | 同一正式动力学的第 0 个成员 | `runs/ablation_v1_dp_dynamics_single_model_pickcube_seed{seed}/dynamics_checkpoint.pt` |
| 评测种子基数 | `31000001` | `51000007` |
| 评测支持 | `max_delay=32`，`ack_lag=1`，5 个动力学成员 | `max_delay=8`，`ack_lag=1`，5 个动力学成员，动作分块长度 8 |
| 每个方法、每个训练 seed | 12 场景 × 20 回合 = 240 回合 | 9 场景 × 20 回合 = 180 回合 |
| 主要指标 | 每回合未折扣总回报 | 每回合至少成功一次的比例（`success_once`） |

权重从 [交接索引](../handoff/checkpoint_index.csv) 的 `role=preferred` 恢复，`{seed}` 逐一替换为 101、102，并验证字节数与 SHA-256。按 [数据恢复说明](../DATA_RECOVERY.md) 校验 Hopper 数据文件和 PickCube 回放数据。PickCube 读取 `data/maniskill/processed/PickCube-v1/motionplanning_state_official_dp_v3_0_1/trajectory.state.pd_ee_delta_pos.physx_cpu.h5`，并加载 `external/ManiSkill-v3.0.1` 的固定 commit。所有评测均使用已冻结模型，训练参数更新数为 0。

- Hopper 的 12 个场景：`clean_delay_0`、`fixed_delay_1`、`fixed_delay_2`、`fixed_delay_4`、`fixed_delay_8`、`fixed_delay_16`、`iid_in_support`、`markov_switching`、`changepoint_in_support`、`drifting_to_long_delay`、`ood_support_shift`、`ood_changepoint`。
- PickCube 的 9 个场景：`clean_delay_0`、`fixed_delay_1`、`fixed_delay_2`、`fixed_delay_4`、`fixed_delay_8`、`iid_in_distribution`、`markov_switching`、`ood_changepoint`、`ood_drifting`。
- 保留完整 v1 场景、顺序、环境、评测 seed 和预先设定的回合数。在每个训练 seed 中，方法共用重置种子 `eval_seed + 100003 * episode_index`、信道 RNG 种子 `episode_seed + 1`。DP 使用同一回合 seed 和原有按重规划序号派生的噪声种子。`replan_every_step` 的重规划时点按定义改变，配对仍以相同初始状态和信道为准。
- 同一个任务和训练 seed 的所有方法在同一类 GPU 上完成，在该批次内重跑 `full_legacy`、`direct` 和无时延参考。每批记录实际 GPU、驱动、CUDA/PyTorch、仿真器、代码 commit、运行时环境与权重数据摘要。不同任务可使用不同卡型。同一任务的两 seed 优先统一卡型，若卡型不同，分层报告各自的配对差值。用于决策的两 seed 合并区间须等同卡型的完整批次齐备。

## 2. 配置与方法

每个配置从对应的 v1 模板复制。所有产物都进入独立 v2 系列。`full_hazard` 以冻结配置为基础，明确设置 `age_posterior=hazard`、`use_content_conditioning=true`、`posterior_updates=true`、`use_censoring=true`、`age_decision_rule=posterior_mean`，保留原风险混合和滚动窗口。PickCube 各配置默认动作分块 8，且 `include_oracle_upper_bound=false`。新特权对照使用单独的 `include_cadr_oracle_age=true` 路径。诊断精度开关在纳入代码的情况下设为 `diagnostic_float64_dynamics=false`。

下表列全 T6 表中的方法；两种固定步长分别计入。同一行对应每个适用任务的 seed 101、102 各一配置，回合数和时长见 §4。

| T6 方法 | 汇总方法 ID | Hopper | PickCube | 相对 `full_hazard` 的设置 |
| --- | --- | :---: | :---: | --- |
| full_hazard | `full_hazard` | ✓ | ✓ | Hopper 使用 `aggregation=action_mean`，PickCube 保留原状态聚合和动作分块 |
| full_legacy | `full_legacy` | ✓ | ✓ | `age_posterior=legacy_cdf`，其他控制参数沿用 v1 |
| no_censoring | `no_censoring` | ✓ | ✓ | `ablation_variant=no_censoring`，`use_censoring=false`，保留 hazard 与在线更新 |
| no_content | `no_content` | ✓ | ✓ | `ablation_variant=no_content`，`use_content_conditioning=false`，使用原无条件 CDF |
| ack_map (hazard) | `ack_map_hazard` | ✓ | ✓ | `ablation_variant=ack_map`，`age_decision_rule=map`，显式保留 `age_posterior=hazard` 和 `use_content_conditioning=true` |
| fixed_prior | `fixed_prior` | ✓ | ✓ | `ablation_variant=fixed_prior`，保留原有弱先验，`posterior_updates=false` |
| fixed_prior_matched | `fixed_prior_matched` | ✓ | ✓ | `ablation_variant=fixed_prior_matched`，`posterior_updates=false`，见下文匹配规则 |
| single_model | `single_model` | ✓ | ✓ | `ablation_variant=single_model`。Hopper 使用 `ensemble_members=[0]`，PickCube 加载专门单模型权重 |
| in_order_age | `in_order_age` | ✓ | ✓ | `age_decision_rule=in_order_age`，按已知生成步与观测变化推断，不用时延分布 |
| fixed_horizon_model（名义均值） | `fixed_horizon_model_mean` | ✓ | ✓ | `age_decision_rule=fixed_horizon_model`，`fixed_horizon_rule=mean` → 1 步 |
| fixed_horizon_model（名义最大支撑） | `fixed_horizon_model_max_support` | ✓ | ✓ | 同上，`fixed_horizon_rule=max_support` → 2 步 |
| cadr_oracle_age | `cadr_oracle_age` | ✓ | ✓ | `include_cadr_oracle_age=true`，特权 **privileged**，仅评价器提供真实年龄 |
| direct | `direct` | ✓ | ✓ | 同一冻结策略直接接收时延观测，PickCube 保持分块 8 |
| replan_every_step | `replan_every_step` | — | ✓ | `ablation_variant=replan_every_step`，`cadr_act_horizon=1`，直接参考仍为 8 |

- 原 `ack_map` 生成器关闭内容约束。生成本行后必须显式打开，确认发生观测变化时使用 hazard 后验并取 MAP，才符合 T6 的 `ack_map (hazard)`。原 v1 的无内容固定配置保留。
- `fixed_prior_matched` 对 fixed 采用点质量，对 IID 采用其参考分布，对变点采用 `before`，对 drift 采用起始分布，对 Markov 采用预先指定的初始 regime 0。先验在整回合固定，运行期间的真实抽样、场景阶段与当前年龄不会传给部署控制器。表中注明该方法预先知道参考分布，并报告 `hazard_zero_mass_fallback` 的使用率。
- 固定步长的名义分布对所有场景固定为 `{0,1,2}:[0.65,0.25,0.10]`。其均值为 0.45，四舍五入为 0。零步等价于直接策略，因而均值规则按已审核的最小非平凡步长取 `max(1, round(0.45))=1`。最大支撑取 2 步。两个主基线都不使用场景真实分布。

### IQL 聚合补充

在同一个 hazard 后验和同一动力学集成上预先指定以下聚合。设 `k` 为候选年龄，`m` 为动力学成员，`p(k)` 为年龄权重，`x(k,m)` 为重建状态。

| 聚合 | 控制计算 | 汇总 ID 与预算 |
| --- | --- | --- |
| `aggregation=action_mean` | 对每个 `x(k,m)` 运行同一冻结 actor，再按 `p(k)/M` 平均动作 | 复用 `full_hazard` |
| `aggregation=state_mean` | 先按 `p(k)/M` 平均状态，再运行同一冻结 actor | 额外 `full_hazard_state_mean`，两个 seed 各一配置 |
| `aggregation=map` | 对同一 `p(k)` 按原 `argmax` 规则选年龄，再按原规则平均该年龄的 `M` 个 actor 动作 | 待 Claude 确认。若与 `ack_map_hazard` 的先验、内容条件、并列处理、聚合与旁路均一致，验证逐位相同后复用该行 |

若 Claude 选择先平均 MAP 年龄下的状态再运行 actor，该聚合与表中的 `ack_map_hazard` 会是两个实验，另加 Hopper 2 个配置、480 回合和约 0.2–0.4 GPU 小时。评测启动前先实现并测试 `aggregation` 和 Hopper 的 `ensemble_members`。在只选第 0 个成员时检查实际前向与聚合的成员数；在 `state_mean` 时验证权重、归一化、零步旁路及冻结权重；在 `map` 时用固定输入检查与 `ack_map_hazard` 等价。这两个配置项截至 `41e9ca6` 还未接入评测，需在开跑前由 Claude 审核实现与兼容性。

## 3. 方法名、目录和运行安排

- 工作单元为 `(task, train_seed, method_id)`，配置文件为 `configs/revision/T6/<task>/seed<seed>/<method_id>.json`，跟踪的系列为 `formal_v2_t6_<task>`（direct、full 两项、特权参照）或 `ablation_v2_t6_<task>`（消融及简单基线）。任务名固定为 `hopper`、`pickcube`。将 v2 `required_methods` 更新为该配置实际执行的方法，以整批清单核验齐备性。
- 原始运行目录为 `runs/revision_v2_t6_<task>_seed<seed>_<method_id>__<batch_id>/`，`batch_id` 由任务、seed、实际卡型和尝试号组成。每个任务和 seed 保存一份场景与方法清单。运行中保存配置、输入哈希、进度和完整输出，续跑须核对这些信息与 GPU 类型。中断或失败沿用原配对种子，在新尝试目录重跑并保留记录。
- 汇总使用表中的 `method_id`，另保留入口原始 `method` 文本。Hopper 的原始直接法为 `direct_observation_iql`，两个 full 变体在当前入口均输出 `cadr_iql`。PickCube 的原始直接法为 `official_maniskill_dp_direct_delayed_observation`，默认 CADR-DP 名为 `cadr_posterior_state_reconstruction_plus_official_maniskill_dp`。变体 ID 从清单和配置映射，避免覆盖同名的 hazard 与 legacy。两个任务的评估专用 oracle 原始名均为 `cadr_oracle_age`。
- 当前 Hopper 与 PickCube 入口在评测 CADR 变体时同时跑直接基线。为了按 §4 的预算执行，P2 前须提供按方法选择的运行入口，或由同一批次执行器复用已经在本批次跑完的规范 `direct`。方法路径继续调用同一 `evaluate_controller` 及同样的冻结控制器。验证新的单方法运行与现有入口同环境相同参数的逐回合动作及主指标一致。新的 oracle 可与 `full_hazard` 共用生成的配置，仅登记两条各自的结果。
- Git 中仅提交设计、配置和经脱敏的小汇总。拟将批次清单、绝对分数、配对差值、计时和解释放在 `results/revision/T6/{manifest.json,absolute_hopper.csv,absolute_pickcube.csv,paired_hopper.csv,paired_pickcube.csv,timing.csv,report.md}`，每个文件小于 1 MB。逐回合原始记录留在同批次的运行归档。

## 4. 回合数和 GPU 占用时间预算

下表以每种方法每个训练 seed 为一配置，按以上 MAP 复用方案计。Clean 场景已包含在各配置内，直接基线每个任务、seed、场景只计一次。时间为单卡被评测作业占用的墙钟规划值，包括该作业内的 CPU 仿真，按顺序执行的工作量相加。

| 配置组 | 变体数 × seed 数 | 单配置回合数 | 总回合数 | 单配置估计 | 小计 GPU 小时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hopper 全部（含额外 `state_mean`） | 14 × 2 | 240 | 6,720 | 6–12 min | 2.8–5.6 |
| PickCube 分块 8 | 13 × 2 | 180 | 4,680 | 27–45 min | 11.7–19.5 |
| PickCube 逐步重规划 | 1 × 2 | 180 | 360 | 208–346 min | 6.9–11.5 |
| 合计 | 56 | — | 11,760 | — | **21.4–36.6** |

预算依据：

1. **Hopper**。[T0](../../../results/revision/T0/reproduction.md) 的 Colab L4 seed 101 快照中，12 个场景分别评估 20 回合 direct 和 20 回合 CADR，总墙钟 897.83 s，单场景两方法 46.04–106.02 s。将总时长粗分成单个方法，240 回合约 7.48 min。T0b 在 Tesla T4 上的运行监看记录为两场景每种精度约 3 min（80 方法回合），折算 240 回合约 9 min。方法间动作及终止长度可能不同，所以用 6–12 min 作计划区间。
2. **PickCube**。T0 的 Colab L4 部分快照只覆盖 clean 和固定 1 步，100 步回合中分块 8 需约 13 次重规划。历史完成块估计每 20 回合约 3–5 min，控制器步耗时 p95 约 0.78 s 可作数量级核对。按 9 场景外推 27–45 min。逐步重规划需 100 次，暂用 `100/13` 倍估算 208–346 min。该比例是基于扩散推理占主要成本的规划近似，真实增量受 CPU 仿真及排队比例影响。
3. 以上依据为已有 T0/T0b 观测及外推。T0a 完成后，在对应 Colab 卡型上记录实际的直接、legacy CADR、DP 正常分块每方法墙钟。逐步重规划仍属外推，待批准后优先以同卡块计时更新。建议另留 15% 计时、恢复或重跑余量，规划总量约 24.6–42.1 GPU 小时，数据下载、运行时配额和等待时间另计。单个 PickCube seed 约需 9.3–15.5 小时，可能跨 Colab 会话。跨会话时须核对卡型与软件版本，保持原配对与输入摘要。
4. 若继续使用每次都跑 direct 的原入口，一个配置同时跑 `full_hazard` 和 oracle，其余分别执行。每个任务、每个 seed 将多跑 11 份直接基线，合计额外 5,280 个 Hopper 和 3,960 个 PickCube 方法回合。调度前须据实际入口更新总时长，分析只计每批经过校验的一份 direct。

T8b 使用 `measure_controller_timing=true`。随每个方法报告 `controller.act` GPU 同步墙钟的 p50、p95 和样本数，注明原生支持、动力学成员数与动作分块。Hopper 全模型口径为 `Dmax=32, M=5`，PickCube 为 `Dmax=8, M=5`，单模型为 `M=1`。结果表中的步耗时与以上完整作业预算分别记录。

## 5. 统计与结果表

配对键为 `task, scenario, train_seed, episode_seed`，另用 `gpu_model, batch_id` 核对同批来源。每种方法在每个场景须有 2 个训练 seed、每 seed 20 回合，共 40 回合。方法差用同一场景的同 seed、同回合先作差，再计算 seed 等权均值。无时延参考取同批次直接策略在 `clean_delay_0` 的 20 回合，使用相同 seed 和初态键关联。绝对分数与对比列报告原始单位；PickCube 差值以百分点呈现，同时报告最终成功率、成功占用和变点后成功占用。

- 95% 层次 percentile bootstrap 预设 10,000 次，固定重采样种子 `20261124`。每次先有放回地抽 2 个训练 seed，再在每个抽中的 seed 内对同配对键有放回地抽 20 个回合。配对比较在同一次抽样中共享索引。均值点估计和区间分别给出，同列附两个 seed 的单独差值。
- 恢复率按 `(方法均值 − 直接均值) / (无时延均值 − 直接均值)`，先计算配对各列的均值，再取比值，每次 bootstrap 同时重采样三列。分母 ≤ 0 或无时延比较自身时写 `NA`。若 bootstrap 出现跨零分母，报告分母不稳定及有效比例，恢复率区间写 `NA`。保留负值和高于 100% 的结果。
- Hopper-v5 数据的元数据缺少同版本 `ref_min_score/ref_max_score`，主表的同版本 D4RL/Minari 归一化记为 `NA`，绝对分数主指标为原始回报。若 Claude 要求在附表给历史 D4RL 刻度，可补 `100 × (return + 20.272305)/(3234.3 + 20.272305)` 并明确按旧环境参考值重标度，作为描述性列。版本同源性见 [Minari Hopper-v5](https://minari.farama.org/v0.5.3/datasets/mujoco/hopper/medium-v0/)、[Minari 归一化说明](https://minari.farama.org/v0.5.3/api/minari_functions/#normalize-score) 与 [D4RL 原始参考值](https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/infos.py)。
- P2 只有两个独立训练 seed，精确符号翻转检验最小双侧 p 值为 0.5。区间和 seed 一致性用于快速判断方向和效应大小，正式显著性结论留给更完整的实验。保留所有场景及负结果；对未完成或硬件不符的配对标明缺口。

| 任务 | GPU | 场景 | 方法及 privileged 标记 | seed 数 × 每 seed 回合 | 绝对分数（95% 区间） | 同批 direct | 无时延直接分数 | 恢复率（区间） | 同版本归一化 | 控制器 p50/p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hopper / PickCube | 待实测 | 原场景名 | 表中 ID | 2 × 20 | 待实测 | 待实测 | 待实测 | 待实测或 NA | Hopper 若缺参考则 NA | 待实测 |

| 任务 | GPU | 场景 | 方法 A − 方法 B | seed 数 × 配对回合 | A/B 绝对分数 | seed 101 / 102 各自差值 | 配对均值差 | 95% 层次 bootstrap 区间 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hopper / PickCube | 待实测 | 原场景名 | 见 §6 | 2 × 20 | 待实测 | 待实测 | 待实测 | 待实测 | 区间是否跨零及数据完整性 |

## 6. 提交给 Claude 的四个判断

1. **风险率后验相对原来的后验**。分别在 12 个 Hopper 场景和 9 个 PickCube 场景报告 `full_hazard − full_legacy`。突出效果方向在两个 seed 一致且探索区间远离零的场景，同时列出负差、跨零和近零结果。
2. **估计时延分布的实际作用**。报告 `full_hazard − in_order_age`，补充 `full_hazard` 与两种固定时域，以及两种固定先验的对照。若未见一致、可辨的增益，将分歧场景、区间和 seed 差值交给 Claude 讨论路线 B。跨零区间表示证据仍不确定。
3. **删失信息在哪些情况下有用**。报告 `full_hazard − no_censoring` 的全部场景结果，预先单列长时延 drift、变点和 support shift，并保留固定与 IID 参照。PickCube 以变点后成功占用补充全回合主指标。若要解释变化后最初 20 个控制步的机制，先审核评价器的诊断采集方案，再按预设的变点窗口和 drift 阶段分段报告，列出各段实际可配对数。
4. **距离已知真实年龄还有多大差距**。报告 `cadr_oracle_age − full_hazard`，同时列 `cadr_oracle_age − direct` 与 `clean_direct − cadr_oracle_age`。第一项描述当前实现可从真实年龄得到的额外收益，后者给出状态重建和闭环控制的剩余差距。真实年龄只给特权评估列，保留 oracle 低于可部署方法的场景。

设计审核关注三个待定口径：IQL MAP 和 ACK-MAP 是否共用一行，新增配置和单方法调度如何保证 v1 逐位回归，以及是否需要额外的分段诊断或历史 D4RL 刻度。T0a 与设计审核通过后，再按批准的口径落实配置和执行器并核对预算。
