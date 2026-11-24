# T0b · Hopper 动力学计算精度对照

2026-11-23。保留 Hopper seed 101 的 v1 配置和冻结权重。独立诊断配置使用 `diagnostic_float64_dynamics: true`，将动力学集成的权重、输入及多步计算切换到 float64，完成状态重建后转回原始张量类型供冻结 IQL 策略使用。该开关默认关闭，既有 v1 配置文件保持原样。

## Colab T4 结果

2026-11-23，作者同意使用 Colab 当前可分配的 GPU。使用同一台 Tesla T4（15360 MiB，驱动 580.82.07）完成两组评测，组别前后的 GPU 标识一致。软件为 Python 3.12.15、PyTorch 2.8.0+cu128、CUDA 12.8、Gymnasium 1.3.0、MuJoCo 3.2.3、Minari 0.5.3 和 h5py 3.16.0。Hopper HDF5 和两份冻结权重的 SHA-256 与归档逐一相符。评测代码固定在 `7c7d8196f8660d359254932cec5f12117add64a2`。精度边界测试也已在该 Colab 环境通过（1 passed）。

| 场景 | 方法 | 配对回合 | 回报逐位变化 | 最大绝对回报差 | 平均回报差的 95% 区间 |
| --- | --- | ---: | ---: | ---: | ---: |
| `drifting_to_long_delay` | 直接 IQL | 20 | 0 | 0 | [0, 0] |
| `drifting_to_long_delay` | CADR | 20 | 20 | 546.997 | [−81.782, 24.514] |
| `markov_switching` | 直接 IQL | 20 | 0 | 0 | [0, 0] |
| `markov_switching` | CADR | 20 | 20 | 87.537 | [−10.137, 13.632] |

精度差定义为 float64 减 float32。区间对同一训练 seed 内的配对回合做 10,000 次 percentile bootstrap，采样 seed 为 20261123。场景的绝对回报均值及与之前跨硬件差异的对照见 [汇总 JSON](t0b_aggregate.json)。

同卡、同种子且只改变动力学计算精度，CADR 的最大单回合差异达到几十至数百，两个场景的平均差区间均覆盖 0。首次 L4 与 RTX 5090 对照中的两个最大差分别为 606.108 和 320.836。T4 的实验支持对动力学数值敏感性的解释。审核原先列出的卡型是 L4，实际 T4 结果供 Claude 核验。T0a 要等 P1 全部合并后另做同环境 legacy 回归。

## 正式运行口径

复制 `configs/generated_formal_v1_hopper_cadr_seed101.json` 到两个临时配置，保留 `markov_switching` 和 `drifting_to_long_delay`、原配对 seed 和每场景 20 个回合。两个配置的 `diagnostic_float64_dynamics` 分别设为 `false` 与 `true`，其余字段相同。用 `dt_corl_delay.evaluate_checkpoints` 在同一台 Colab GPU 加载 P0 校验通过的两份权重。在不同输出目录保存结果，用仓库的汇总入口生成回合 CSV，然后用 `scripts/revision/compare_episodes.py` 按任务、方法、训练 seed、场景和回合 seed 配对，报告逐回合差异、场景平均差与 seed 内配对 bootstrap 区间。

首次 RTX 5090 与 L4 比对的现有 28 个方法/场景组合均满足修订后的跨硬件软条件，目前包括 Hopper 完整复跑与 PickCube 部分复跑。其中 Hopper 的 `ood_support_shift` CADR 回报差区间不覆盖零，但平均差为归档均值的 0.061%，低于 5%。
