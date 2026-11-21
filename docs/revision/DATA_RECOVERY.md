# 数据与 checkpoint 恢复（2026-11-20 更新）

**结论：不需要重训任何模型。** 服务器数据删除了，但作者本地（Windows，`D:/PROJECT/lzt/research/dt-corl-delay`）保留了全部正式权重、5 条网络 trace 和原始结果。需要重新下载的只有 Minari 数据集和 ManiSkill 演示。

完整交接说明见 [handoff/RECOVERY_HANDOFF_2026-11-19.md](handoff/RECOVERY_HANDOFF_2026-11-19.md)，逐文件索引（路径、字节数、SHA-256）见 [handoff/checkpoint_index.csv](handoff/checkpoint_index.csv) 和 [handoff/data_index.csv](handoff/data_index.csv)。

## 1. 现状一览

| 资源 | 状态 | 做法 |
|---|---|---|
| 正式权重：IQL、CQL、CADR 动力学、DT-CORL 适配器、官方 DP、CADR-DP 动力学、PickCube 单模型消融 | **本地都在**：155 个 `role=preferred`，10 个任务 × 5 个 seed 全覆盖，SHA-256 已核对 | 从本地 `artifacts/remote/<运行名>/` 复制到服务器 `runs/<运行名>/`，复制后按 `recorded_sha256` 校验 |
| 网络 trace：3 条 5G + 2 条 WiFi | **本地都在**，已与历史校验一致 | 按相同相对路径复制到 `data/processed/` |
| 原论文结果：`episodes.csv`、`main_aggregate.csv` 等 | **本地都在** | 复制到 `artifacts/remote/formal_v1_*`，供 T0 逐回合比对 |
| Minari 7 个数据集 `main_data.hdf5` | 缺失（metadata 和下载清单在） | 重新下载，按 `data_index.csv` 的字节数和 SHA-256 校验，**必须完全一致** |
| ManiSkill 3 个任务的原始和转换后演示 | 缺失 | 重新下载并回放转换。CADR-DP 评测要读取完整回放文件 |

## 2. 只恢复本轮需要的（按阶段）

| 阶段 | 任务 | 需要的权重 | 需要的数据 |
|---|---|---|---|
| T0 / P2 | Hopper | `formal_v1_iql_hopper_seed{101..105}`、`formal_v1_cadr_hopper_seed{…}`（P2 加 `formal_v1_dtcorl_hopper_*`） | `mujoco/hopper/medium-v0` |
| T0 / P2 | PickCube | `formal_v1_dp_pickcube_seed{…}/checkpoints/best_eval_success_once.pt`、`formal_v1_dp_dynamics_pickcube_seed{…}`、`ablation_v1_dp_dynamics_single_model_pickcube_seed{…}` | PickCube-v1 演示（下载 + 转换） |
| P3 | 其余 8 个任务 | 按 `checkpoint_index.csv` 筛选 `role=preferred` | 其余 Minari 数据集与 ManiSkill 演示；5 条 trace |

`role=archive_only` 的 498 个文件是留档快照，**不要使用**。

## 3. 下载与校验

### Minari

```bash
export PYTHONPATH="$PWD/src"
export MINARI_DATASETS_PATH="$PWD/datasets/minari"
export HF_HOME="$PWD/cache/huggingface" XDG_CACHE_HOME="$PWD/cache/xdg" TMPDIR="$PWD/cache/tmp"
mkdir -p "$MINARI_DATASETS_PATH" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
envs/research/bin/python -m dt_corl_delay.download_minari_exact \
  --dataset mujoco/hopper/medium-v0 --datasets-root "$MINARI_DATASETS_PATH"
sha256sum datasets/minari/mujoco/hopper/medium-v0/data/main_data.hdf5
# 应为 17c1b07d01ce461bb9b2866ad0356c542678b66b886b5ed6c066733a75c52c84（139555224 字节）
```

同时把本地保留的 `metadata.json` 和 `download_manifest.json` 一起复制过去，保持 Minari 的目录结构。

### ManiSkill（`scripts/remote/*.sh` 不在公开仓库里，直接用 Python 入口）

```bash
export MS_ASSET_DIR="$PWD/data/maniskill/assets"
envs/maniskill/bin/python -m mani_skill.utils.download_demo PickCube-v1 --output_dir "$PWD/data/maniskill/demos"
envs/maniskill/bin/python -m mani_skill.trajectory.replay_trajectory \
  --traj-path data/maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state --save-traj -b physx_cpu
# 把生成的 trajectory.state.pd_ee_delta_pos.physx_cpu.{h5,json} 移到
# data/maniskill/processed/PickCube-v1/motionplanning_state_official_dp_v3_0_1/
# PegInsertionSide 用 -c pd_ee_delta_pose
```

- 回放参数如果与 ManiSkill 3.0.1（commit `a4a4f92`）的实际 CLI 不一致，以上游为准，并把实际命令写进 PR。
- 转换结果的摘要要与 `data_index.csv` 的记录比对。不一致时，先查转换设置、轨迹和数据来源；查不清就把这次运行标为“新复现”，并在 T0 中额外比对评测结果。

### 上游代码

- `external/ManiSkill-v3.0.1`：完整 Git checkout，commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`。本地备份只有部分文件，**必须重新 checkout**。
- `external/DT-CORL`：commit `7e44adf934b8aef91ca6fb9e057a1935c6a167c9`。

### 环境

沿用交接文档 §5.2 的三套环境：`envs/research`、`envs/maniskill`、`envs/maniskill-dp`。如果服务器 GPU 架构变了，记录实际的 PyTorch/CUDA 版本，写到 `results/revision/P0/env_record.md`。

## 4. 路径注意

- 历史脚本里写死了 `/root/autodl-tmp/dt-corl-delay`。新服务器上要么沿用这个结构，要么只用 Python 入口，并通过 `DT_CORL_PROJECT_ROOT`、`DT_CORL_DATA_ROOT`、`MINARI_DATASETS_PATH` 指定路径。
- 新的评测输出一律放在新目录（`runs/recovery_*` 或 `runs/revision_v2_*`），**不要覆盖原归档**。
