# 数据与 checkpoint 重建（服务器数据已删除）

原服务器上的数据集、checkpoint、处理后的 trace 都已不存在。**所有基础策略、动力学模型都必须重训**（seed 101–105）。先只重建 P2 需要的部分：Hopper + PickCube，seed 101、102。

> 下面的命令以仓库现有入口为准；标“需核实”的步骤请先对照上游 v3.0.1 文档，把实际使用的命令写进 PR。

## 1. 环境

- Minari 线：`requirements-research.in`（记录环境为 PyTorch 2.8.0 + CUDA 12.8、Minari 0.5.3、MuJoCo 3.2.3）。
- ManiSkill 线：`requirements-maniskill.in` + `requirements-maniskill-dp.in`（Gymnasium 0.29.1）。**必须与 Minari 线分开建环境。**
- 把实际安装的版本记录到 `results/revision/P0/env_record.md`（`pip freeze` 摘要 + GPU 型号 + CUDA 版本）。

## 2. 离线数据集（公开，可重新下载）

| 任务组 | dataset_id |
|---|---|
| Locomotion | `mujoco/halfcheetah/medium-v0`, `mujoco/hopper/medium-v0`, `mujoco/walker2d/medium-v0` |
| Adroit | `D4RL/door/expert-v2`, `D4RL/hammer/expert-v2`, `D4RL/pen/expert-v2`, `D4RL/relocate/expert-v2` |

```bash
export MINARI_DATASETS_PATH=$PWD/datasets/minari
python -m dt_corl_delay.download_minari_exact --dataset mujoco/hopper/medium-v0 --datasets-root datasets/minari
# P3 时再下载其余 ID
```

## 3. ManiSkill 演示（公开，需要下载 + 回放转换）

配置期望的路径：

```
data/maniskill/processed/PickCube-v1/motionplanning_state_official_dp_v3_0_1/trajectory.state.pd_ee_delta_pos.physx_cpu.h5
data/maniskill/processed/StackCube-v1/motionplanning_state_official_dp_v3_0_1/trajectory.state.pd_ee_delta_pos.physx_cpu.h5
data/maniskill/processed/PegInsertionSide-v1/motionplanning_state_official_dp_v3_0_1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5
```

步骤（需核实，ManiSkill 版本固定为 3.0.1，commit `a4a4f92`）：

```bash
python -m mani_skill.utils.download_demo PickCube-v1
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path <下载目录>/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state --save-traj -b physx_cpu
# PegInsertionSide 用 -c pd_ee_delta_pose
```

- 每个任务 100 条轨迹（与论文一致）。确认后用 `python -m dt_corl_delay.maniskill_dataset_audit` 做审计，并把审计 JSON 放进 `results/revision/P0/`。
- Diffusion Policy 用 ManiSkill 官方 state baseline，按原配置的超参数重训：obs horizon 2、prediction horizon 16、action horizon 8。

## 4. 上游代码

- DT-CORL：<https://github.com/simonjzh/DT-CORL>，commit `7e44adf934b8aef91ca6fb9e057a1935c6a167c9`，放在 `external/DT-CORL`（加载器会校验 `utils/network.py` 的 SHA-256）。
- ManiSkill：3.0.1，commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`。

## 5. 网络 trace

| 文件（相对 trace 根目录） | 来源 | 能否重建 |
|---|---|---|
| `5g/pam2025_50ms/atnt/d3/atnt_day_2_d3_rtt.csv` | PAM 2025 公开数据 <https://github.com/NUWiNS/pam2025-multi-carrier-dataset> | 能：下载后用 `python -m dt_corl_delay.network_traces --source-root ... --output-root ...`（默认 50 ms、one-way 0.5、×1/2/4、max delay 32） |
| `5g/pam2025_50ms/tmobile/d2/tmobile_day_1_d2_rtt.csv` | 同上 | 能 |
| `5g/pam2025_50ms/verizon/d3/verizon_day_3_d3_rtt.csv` | 同上 | 能 |
| `wifi/wlan_bound_google_v4_50ms.csv` | **作者本地实测，未公开** | **不确定**：需要作者从本地或备份找回原始数据，再用 `python -m dt_corl_delay.wifi_traces --raw-directory ... --output-file ...` 处理 |
| `wifi/wlan_gateway_50ms.csv` | 同上 | 同上 |

找不回 WiFi 原始数据的方案（由作者决定）：
1. 重新实测；
2. 改用公开 WiFi RTT 数据；
3. trace 实验只保留 3 个蜂窝来源，并在论文中说明。

trace 实验在 P3，不阻塞 P1/P2。

## 6. 需要重训的 checkpoint（P2 范围）

| 任务 | 模型 | seed |
|---|---|---|
| Hopper | IQL（`generated_formal_v1_hopper_iql_*` 的超参数）+ CADR 动力学集成 | 101, 102 |
| PickCube | 官方 DP + CADR 动力学集成 | 101, 102 |

验收：无时延下直接策略的分数与原论文同量级（Hopper medium IQL 的 normalized score 应在常见报告范围内；PickCube DP 在无时延下接近 100% 成功）。偏差很大时先停下来，在 PR 里报告。
