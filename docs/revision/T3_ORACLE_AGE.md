# T3：CADR 真实年龄对照

在新的 `formal_v2_*` 或 `ablation_v2_*` 评测配置里设置
`"include_cadr_oracle_age": true`，并使用独立运行目录。IQL 使用
`dt_corl_delay.evaluate_checkpoints`，ManiSkill 使用
`dt_corl_delay.evaluate_maniskill_cadr_diffusion`。若从 v1 复制 DP 配置，
请把旧的 `include_oracle_upper_bound` 设为 `false`，避免重复比较。
若配置用 `required_methods` 约束比较完整性，把 `cadr_oracle_age` 加入 v2 比较集。

只在该次 privileged 评测中，评测器把选中观测的
`decision_step - selected_packet.generated_step` 送给 oracle。此差值对应当前
持有的观测，可以不同于该包最初的传输时延。对照将年龄分布替换为点质量，
随后复用同一动力学、执行动作历史和冻结策略。IQL 保留动作聚合及风险混合，
DP 保留状态聚合、风险混合、执行块长度和采样种子。零龄沿用同一直接策略计算路径。
每一步都需要新提供的年龄，使用一次后清除。普通 CADR 不从信道读取真实年龄。

结果方法名为 `cadr_oracle_age`。方法级 JSON 和 v2 的回合、汇总表通过
`privileged_selected_age` 列标注特权。v1 的配置、方法名及表格结构沿用原样。
结果解释采用同一场景和种子下的配对对照。CADR 与 oracle-age 的差异衡量给定动力学
和策略时年龄推断带来的表现差异。oracle-age 和无时延参考的差异可用于诊断重建误差，
闭环轨迹的变化也会影响两者的分数差异。
