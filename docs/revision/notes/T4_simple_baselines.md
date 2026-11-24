# T4 简单基线

在新的 v2 配置中通过 `age_decision_rule` 启用。IQL 与 CADR-DP 共用年龄选择逻辑，
沿用各自已冻结的动力学及策略输出接口。简单基线与特权对照使用独立配置。
CADR-DP 的简单基线配置应关闭 `include_oracle_upper_bound` 及 `include_cadr_oracle_age`。

- `in_order_age` 适用于固定一步 ACK 滞后。观测未变时取 `t - G_t`，变化时用
  假设 `G_t + 1` 得出 `t - G_t - 1`。只从已揭示 ACK 更新 `G_t`，结合观测内容
  选择年龄。年龄在 `[0, max_delay]` 内，推理路径不实例化时延估计器。
- `fixed_horizon_model` 在所有场景使用同一名义 IID 参考分布，
  时延 `[0, 1, 2]` 对应概率 `[0.65, 0.25, 0.10]`。
  `fixed_horizon_rule` 取 `mean` 时，期望 0.45 按四舍五入为 0，
  再应用预先声明的一步下限得到 T6 规定的 `H=1`。
  `max_support` 取 `H=2`。控制器只接收 H，用已执行动作中最多 H 步作重建。
  程序在摘要中保留每个场景实际使用的 H。

主基线忽略场景专属参考分布。如果另设采用场景真实分布的对照，应使用
`_scenario_tuned` 后缀并在结果表标注 privileged。实际比较在 T6 执行。
