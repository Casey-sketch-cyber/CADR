# T0a · 同卡 legacy 回归

按 Claude 最新审核，在同一 Colab 会话、同一张 Tesla T4（15,360 MiB，驱动 580.82.07）上比较 P1 合入前 `33d104e08fe739790672f3c14dc123ff0f4199f1` 与合入后 `41e9ca605be887b2cfa71dd6a46191d392e838f8`。每项任务的两组使用同一 Python 环境、同一原始 v1 配置文件、同一数据和冻结权重。检查了配置、模型和数据的 SHA-256，并在评测开始和结束时核对 GPU 标识。

| 任务 | 训练 seed | 场景 × 方法 × 每组回合数 | 已配对 | 最大绝对回报差 | 状态 |
| --- | ---: | --- | ---: | ---: | --- |
| Hopper · IQL | 101 | 12 × 2 × 20 | 480/480 | 0 | 逐位一致 |
| PickCube · Diffusion Policy | 101 | 9 × 3 × 20 | 80/540 | 0 | 已比较部分逐位一致，评测继续 |

运行使用原始评测入口和完整场景配置。首先用 `compare_episodes.py` 按任务、训练 seed、场景、方法、回合编号及回合 seed 配对。另核对回报的浮点位模式、回合长度、已记录的成功指标和变点恢复步数。计时与采样环境元数据单独记录。Hopper 的两个方法均完成全部 12 个场景，见 [场景汇总](hopper_aggregate.json)。

T0a 总关口待 PickCube 完整结果。P2 同时等待 [T6 设计](https://github.com/Casey-sketch-cyber/CADR/pull/13)审核。
