# T0b · Hopper 动力学计算精度对照

2026-11-23。保留 Hopper seed 101 的 v1 配置和冻结权重。独立诊断配置使用 `diagnostic_float64_dynamics: true`，将动力学集成的权重、输入及多步计算切换到 float64，完成状态重建后转回原始张量类型供冻结 IQL 策略使用。该开关默认关闭，既有 v1 配置文件保持原样。

已准备好评测代码、测试和输入校验。正式结果待 Colab L4 分配成功后生成。

## 正式运行口径

复制 `configs/generated_formal_v1_hopper_cadr_seed101.json` 到两个临时配置，保留 `markov_switching` 和 `drifting_to_long_delay`、原配对 seed 和每场景 20 个回合。两个配置的 `diagnostic_float64_dynamics` 分别设为 `false` 与 `true`，其余字段相同。用 `dt_corl_delay.evaluate_checkpoints` 在同一台 Colab L4 加载 P0 校验通过的两份权重。在不同输出目录保存结果，用仓库的汇总入口生成回合 CSV，然后用 `scripts/revision/compare_episodes.py` 按任务、方法、训练 seed、场景和回合 seed 配对，报告逐回合差异、场景平均差与 seed 内配对 bootstrap 区间。

首次 RTX 5090 与 L4 比对的现有 28 个方法/场景组合均满足修订后的跨硬件软条件，目前包括 Hopper 完整复跑与 PickCube 部分复跑。其中 Hopper 的 `ood_support_shift` CADR 回报差区间不覆盖零，但平均差为归档均值的 0.061%，低于 5%。
