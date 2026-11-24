# T0a · 同卡 legacy 回归

按 Claude 最新审核，在同一 Colab 会话、同一张 Tesla T4（15,360 MiB，驱动 580.82.07）上比较 P1 合入前 `33d104e08fe739790672f3c14dc123ff0f4199f1` 与合入后 `41e9ca605be887b2cfa71dd6a46191d392e838f8`。每项任务的两组使用同一 Python 环境、同一原始 v1 配置文件、同一数据和冻结权重。检查了配置、模型和数据的 SHA-256，并在评测开始和结束时核对 GPU 标识。

| 任务 | 训练 seed | 场景 × 方法 × 每组回合数 | 已配对 | 最大绝对回报差 | 状态 |
| --- | ---: | --- | ---: | ---: | --- |
| Hopper · IQL | 101 | 12 × 2 × 20 | 480/480 | 0 | 逐位一致 |
| PickCube · Diffusion Policy | 101 | 9 × 3 × 20 | 540/540 | 0 | 逐位一致 |

运行使用原始评测入口和完整场景配置。首先用 `compare_episodes.py` 按任务、训练 seed、场景、方法、回合编号及回合 seed 配对。另核对回报的浮点位模式、回合长度、已记录的成功指标和变点恢复步数。两个任务的全部 1,020 个配对回合均吻合，最大绝对回报差为 0。计时与运行时元数据单独记录。逐场景汇总见 [Hopper](hopper_aggregate.json) 和 [PickCube](pickcube_aggregate.json)。

PickCube 两组各完成 400 个回合后，在 `markov_switching` 的 oracle 仿真环境初始化时收到 CPython `gilstate_tss_set` 原生错误并中止。保存已有进度后，在同一 Colab 会话和同一张 T4 上以原入口的 `--resume` 完成剩余回合。续跑沿用原配置、输入和评测种子；按同样的完整覆盖与逐位标准检查。

T0a 达到审核规定的逐回合一致性标准。P2 等待 [T6 设计](https://github.com/Casey-sketch-cyber/CADR/pull/13)审核。
