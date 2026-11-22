# P0 · Hopper 与 PickCube 的 seed 101 恢复记录

2026-11-22。目标是为 T0 恢复旧版评测输入。Colab 工作副本检出 `33d104e08fe739790672f3c14dc123ff0f4199f1`。选用了 NVIDIA L4（23,034 MiB，驱动 580.82.07），因为 Colab 当时未能分配标准 A100。重新计算的输入摘要均与 2026-11-19 的作者本地恢复索引一致。

| 数据或模型文件 | 字节数 | SHA-256 结果 |
|---|---:|---|
| Hopper Minari `mujoco/hopper/medium-v0` HDF5 | 139,555,224 | 与原归档一致 |
| PickCube 转换后演示 HDF5 | 39,414,170 | 与原归档一致 |
| PickCube 转换后演示元数据 JSON | 231,054 | 与原归档一致 |
| Hopper seed 101 IQL 策略 | 3,964,423 | 与权重索引一致 |
| Hopper seed 101 CADR 动力学 | 4,401,637 | 与权重索引一致 |
| PickCube seed 101 官方 DP 策略 `best_eval_success_once.pt` | 35,257,889 | 与权重索引一致 |
| PickCube seed 101 CADR-DP 动力学 | 5,371,557 | 与权重索引一致 |

本次只恢复 seed 101 所需的模型。PickCube 用原版 `motionplanning` 演示，固定官方 ManiSkill 源码 `a4a4f9272ad64b1564035874b605ceb687b63ed8`。回放使用 `--use-first-env-state --target-control-mode pd_ee_delta_pos --obs-mode state --save-traj --num-envs 10 --sim-backend physx_cpu --record-rewards --reward-mode normalized_dense --render-mode none`，1000 条演示全部保留。

Colab 默认 Python 3.13 缺 `ensurepip`，且原 MuJoCo 3.2.3 和原 `mplib==0.1.1` 无法在该版本使用。使用 `uv` 建立分开的 Python 3.12.15 环境。两套评测环境均为 PyTorch 2.8.0+cu128、CUDA 12.8、NumPy 2.3.2、h5py 3.16.0。Hopper 环境为 Gymnasium 1.3.0、MuJoCo 3.2.3、Minari 0.5.3。Hopper 不用 Adroit 的 Gymnasium-Robotics 1.2.0，因此不在此次隔离环境安装这个与 SciPy 的 NumPy 约束冲突的包。PickCube DP 环境为 Gymnasium 0.29.1、ManiSkill 3.0.1、SAPIEN 3.0.3、diffusers 0.35.2。

原 v1 配置、权重和评测代码均保留。恢复数据在 Colab 使用后与归档 SHA-256 对照。P0 的输入完整性通过，T0 的逐回合一致性由独立比较报告判定。
