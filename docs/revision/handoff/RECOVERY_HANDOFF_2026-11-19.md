# CADR 实验恢复交接

更新于 2026-11-19。供 Claude、Gemini 或接手复现实验的人使用。用户已清理原远程服务器上的数据。先确认本次需要核对旧结果、补跑评测、从头训练，还是恢复中断的训练，再按任务和种子选择输入。

- Windows 本地项目根目录 `D:/PROJECT/lzt/research/dt-corl-delay`。以下本地相对路径均从这里开始。
- 原远程项目目录 `/root/autodl-tmp/dt-corl-delay`。历史脚本内写死了此路径。换服务器时优先沿用数据盘结构，或使用可配置的 Python 入口并核对配置中的路径。
- [模型路径索引](checkpoint_index.csv)列出 `artifacts/` 和 `runs/` 下 653 个模型文件。筛选 `role=preferred` 得到 155 个优先使用的权重，均已核对本地文件大小和 SHA-256 与原运行清单一致。其余是留档快照和开发权重，仅登记路径及大小。
- [数据及结果路径索引](data_index.csv)列出 39 条恢复记录，涵盖训练数据、演示和配套元数据、5 个时延文件、结果和归档核验报告。`server_relative_path` 是相对远程项目根目录的目标路径。对缺失文件，`local_path` 是本地预期位置，`expected_sha256` 来自历史记录；空值表示索引没有该项的历史校验值。

## 1. 先按目标选择最少的恢复工作

| 目标 | 需要准备 |
|---|---|
| 查论文旧数字、重画图、核对置信区间 | 直接读取本地 `Source_Data.zip`、冻结汇总或逐回合结果，无需启动模拟器。 |
| 补跑某个任务的合成延迟评测 | 恢复对应仿真环境、该任务的数据及同种子的策略和动力学权重，用对应 `generated_formal_v1_*.json`。开新输出目录。 |
| 补跑同一任务的实测网络回放 | 在同样输入上再恢复五个时延 CSV，换用 `real_network` 配置。 |
| 重新训练 | 恢复所选任务的原版离线数据和冻结配置，再训练需要更新的模型。若只训练 CADR 动力学，可配用已有的匹配策略。 |
| 继续中断的训练 | 原运行目录、原配置、输入快照以及含优化器和随机数状态的恢复权重都要齐全。官方 ManiSkill Diffusion Policy 历史训练器没有保存优化器状态，需从头训练该策略，或直接选用已完成策略补评测。 |

**评测脚本的数据依赖。** MuJoCo 和 Adroit 的现有评测入口从 Minari 数据恢复环境。CADR-DP 的现有入口读取已转换的演示 HDF5，涉及输入校验及策略初始化。补跑评测可以复用权重，但要恢复对应的数据文件。

## 2. 原始训练数据和演示

| 任务名 | 配置及模型路径用的任务代号 | 原训练数据 ID 或演示 |
|---|---|---|
| HalfCheetah | `halfcheetah` | `mujoco/halfcheetah/medium-v0` |
| Hopper | `hopper` | `mujoco/hopper/medium-v0` |
| Walker2d | `walker2d` | `mujoco/walker2d/medium-v0` |
| Adroit Door | `adroit_door` | `D4RL/door/expert-v2` |
| Adroit Pen | `adroit_pen` | `D4RL/pen/expert-v2` |
| Adroit Hammer | `adroit_hammer` | `D4RL/hammer/expert-v2` |
| Adroit Relocate | `adroit_relocate` | `D4RL/relocate/expert-v2` |
| PickCube | `pickcube` | `PickCube-v1` 官方 `motionplanning` 演示，`pd_ee_delta_pos` |
| StackCube | `stackcube` | `StackCube-v1` 官方 `motionplanning` 演示，`pd_ee_delta_pos` |
| PegInsertionSide | `peginsertion` | `PegInsertionSide-v1` 官方 `motionplanning` 演示，`pd_ee_delta_pose` |

**Minari。** D 盘 `datasets/minari/` 保留了 7 个数据集的下载清单和元数据。原始 HDF5 缺失。每个任务的服务器目标是 `datasets/minari/<数据集ID>/data/main_data.hdf5` 及同目录 `metadata.json`，并保留 Minari 的目录和命名空间元数据。用原 `download_manifest.json` 或数据索引中的字节数、SHA-256 对照下载结果。原 AutoDL 环境可用 `bash scripts/remote/download_minari.sh <数据集ID>`；如果新服务器没有脚本使用的 `/etc/network_turbo`，可从项目根目录直接调用下载模块：

```bash
export PYTHONPATH="$PWD/src"
export MINARI_DATASETS_PATH="$PWD/datasets/minari"
export HF_HOME="$PWD/cache/huggingface" XDG_CACHE_HOME="$PWD/cache/xdg" TMPDIR="$PWD/cache/tmp"
mkdir -p "$MINARI_DATASETS_PATH" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
envs/research/bin/python -m dt_corl_delay.download_minari_exact \
  --dataset mujoco/hopper/medium-v0 --datasets-root "$MINARI_DATASETS_PATH"
```

**ManiSkill。** 项目内未找到原始或转换后演示 HDF5。在准备好 `envs/maniskill` 后，原 AutoDL 可用 `bash scripts/remote/download_maniskill_demo.sh PickCube-v1` 获取该任务演示。常规网络可运行 `envs/maniskill/bin/python -m mani_skill.utils.download_demo PickCube-v1 --output_dir "$PWD/data/maniskill/demos"`。下载文件应落在 `data/maniskill/demos/<ENV_ID>/motionplanning/trajectory.h5`，以及同目录 `trajectory.json`。随后在项目根目录运行 `bash scripts/remote/replay_maniskill_official_dp.sh PickCube-v1`，把任务名替换成所需任务。保留原设置 `--use-first-env-state`、`state`、`physx_cpu` 及上表控制模式。

转换结果路径是 `data/maniskill/processed/<ENV_ID>/motionplanning_state_official_dp_v3_0_1/trajectory.state.<控制模式>.physx_cpu.h5`，以及同名 `.json`。保留完整回放文件。官方 DP 策略训练的 `num_demos=100` 是训练配置，CADR 动力学和现有评测入口会读取完整回放。数据索引中的 HDF5 和 JSON 摘要来自原正式运行记录。重新生成后需要比对；如果摘要不同，先核验转换设置、轨迹和数据来源，再把运行标为新复现。 [Minari 数据下载文档](https://minari.farama.org/v0.5.3/content/basic_usage/) · [ManiSkill 演示下载和回放说明](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html)

## 3. 模型权重

本地目录 `artifacts/remote/` 是原运行目录的备份。下表路径以该目录为前缀；恢复到原 Linux 目录结构时，使用 `runs/` 作为前缀。CSV 给出每个任务和种子的完整路径、目标位置、字节数和 SHA-256。

| 模型 | 文件路径模板 | 覆盖 |
|---|---|---|
| IQL 策略 | `formal_v1_iql_{task}_seed{seed}/checkpoint.pt` | 7 任务 × 5 |
| CADR 动力学 | `formal_v1_cadr_{task}_seed{seed}/dynamics_checkpoint.pt` | 7 × 5 |
| CQL 策略 | `formal_v1_cql_{task}_seed{seed}/checkpoint.pt` | 3 个 MuJoCo 任务 × 5 |
| DT-CORL 适配器 | `formal_v1_dtcorl_{task}_seed{seed}/dtcorl_checkpoint.pt` | 7 × 5，与同种子 IQL 配用 |
| 官方 Diffusion Policy | `formal_v1_dp_{task}_seed{seed}/checkpoints/best_eval_success_once.pt` | 3 个 ManiSkill 任务 × 5 |
| CADR-DP 动力学 | `formal_v1_dp_dynamics_{task}_seed{seed}/dynamics_checkpoint.pt` | 3 × 5 |
| PickCube 单动力学模型消融 | `ablation_v1_dp_dynamics_single_model_pickcube_seed{seed}/dynamics_checkpoint.pt` | 5 个种子 |

正式训练种子为 `101` 至 `105`。ManiSkill 策略按无延迟验证的 `best_eval_success_once.pt` 选取。PickCube 的 `ack_map`、`no_content`、`fixed_prior`、`replan_every_step` 复用正式多模型动力学；`single_model` 选用上表的单模型权重。五个消融共用同种子的正式 DP 策略，配置在 `configs/generated_ablation_v1_pickcube_<variant>_seed<seed>.json`。

## 4. 实测时延和已有结果

正式回放需要以下 5 个文件。文件已在 D 盘，按相同相对位置拷贝；原生、2 倍、4 倍延迟分别读每个 CSV 的 `delay_steps_x1`、`delay_steps_x2`、`delay_steps_x4`。

```text
data/processed/wifi/wlan_gateway_50ms.csv
data/processed/wifi/wlan_bound_google_v4_50ms.csv
data/processed/5g/pam2025_50ms/tmobile/d2/tmobile_day_1_d2_rtt.csv
data/processed/5g/pam2025_50ms/atnt/d3/atnt_day_2_d3_rtt.csv
data/processed/5g/pam2025_50ms/verizon/d3/verizon_day_3_d3_rtt.csv
```

复核论文数字先用 `artifacts/remote/formal_v1_paper_artifacts/`，其中有 `main_aggregate.csv`、`primary_contrasts.csv`、`ablation_contrasts.csv`、`trace_scale_summary.csv`。逐回合记录在 `artifacts/remote/formal_v1_{minari,maniskill}_{synthetic,real_network}/episodes.csv`，其他归档汇总在 `artifacts/analysis/`。随稿数据是 `paper/applied_soft_computing/output/CADR_ASC_Submission_20261102/05_Supplementary_Material/Source_Data.zip`。

## 5. 新服务器上的执行顺序

1. **确认代码和环境。** 使用本地完整项目的 `src/`、`configs/`、`scripts/remote/`、`requirements*.in` 和历史清单。公开 [CADR 仓库](https://github.com/liangdagongjue-code/CADR) 可用于查代码和复现说明；原运行器和归档核验工具保存在本地完整项目。把大文件和缓存放在数据盘。
2. **分开安装环境。** Minari 路线用 `envs/research`，原环境记录为 PyTorch 2.8.0 / CUDA 12.8、Minari 0.5.3、MuJoCo 3.2.3、Gymnasium 1.3.0，Adroit 用 Gymnasium-Robotics 1.2.0。ManiSkill 转换用 `envs/maniskill`，扩散训练和评测用 `envs/maniskill-dp`，后者用 Gymnasium 0.29.1 和 diffusers 0.35.2。参考 `requirements-research.in`、`requirements-maniskill.in`、`requirements-maniskill-dp.in` 及 `requirements-locks/remote/`。原研究环境安装脚本还检查 GPU 架构；换硬件时为新环境选择兼容的 PyTorch/CUDA 并记录版本。
3. **取得需要的上游源码。** ManiSkill 正式 DP 从 `external/ManiSkill-v3.0.1` 的 Git checkout 读取，固定提交 `a4a4f9272ad64b1564035874b605ceb687b63ed8`。本地同名备份只有部分文件，重建时请检出对应提交，以便评测确认当前版本。DT-CORL 适配器从 `external/DT-CORL` 加载，固定提交 `7e44adf934b8aef91ca6fb9e057a1935c6a167c9`。上游链接和文件摘要见 [第三方依赖说明](../../release/CADR/THIRD_PARTY.md)。
4. **只恢复本轮任务。** 先下该任务的数据并校验，再从模型索引复制必要的配对权重。末端核对评测配置、5 个训练种子中所选种子、权重 SHA-256、评测种子及网络轨迹列。同一轮比较保持相同的环境初值和延迟输入，策略和动力学在部署时冻结，真实年龄只用于明确标记的 oracle 对照。
5. **先跑一项验证，再扩展。** 先选一个任务和一个正式种子，在新的 `runs/recovery_<唯一名称>` 下跑，确认配置和指标，再扩展。如果只是补一项，直接使用下面的评测入口；全量正式脚本会校验整个冻结树，并可能启动训练。保留原归档，把新输出标明为复跑。

### 补跑评测示例

下列示例假设对应数据、权重和环境已恢复，且从 Linux 项目根目录开始。把 `recovery_...` 改成尚不存在的目录名。

```bash
export PYTHONPATH="$PWD/src" DT_CORL_PROJECT_ROOT="$PWD"
export MINARI_DATASETS_PATH="$PWD/datasets/minari"
export DT_CORL_DATA_ROOT="$PWD/data/processed" MUJOCO_GL=egl

# Hopper IQL 与 CADR，种子 101
envs/research/bin/python -m dt_corl_delay.evaluate_checkpoints \
  --config configs/generated_formal_v1_hopper_cadr_seed101.json \
  --base-checkpoint runs/formal_v1_iql_hopper_seed101/checkpoint.pt \
  --dynamics-checkpoint runs/formal_v1_cadr_hopper_seed101/dynamics_checkpoint.pt \
  --output runs/recovery_hopper_101_synthetic

# PickCube Diffusion Policy 与 CADR-DP，种子 101
export MS_ASSET_DIR="$PWD/data/maniskill/assets"
envs/maniskill-dp/bin/python -m dt_corl_delay.evaluate_maniskill_cadr_diffusion \
  --config configs/generated_formal_v1_pickcube_synthetic_seed101.json \
  --policy-checkpoint runs/formal_v1_dp_pickcube_seed101/checkpoints/best_eval_success_once.pt \
  --dynamics-checkpoint runs/formal_v1_dp_dynamics_pickcube_seed101/dynamics_checkpoint.pt \
  --source-root external/ManiSkill-v3.0.1 \
  --output runs/recovery_pickcube_101_synthetic
```

正式实测网络回放分别换成 `generated_formal_v1_hopper_real_network_seed101.json` 或 `generated_formal_v1_pickcube_real_network_seed101.json`。直接 CQL 用对应 CQL 策略和 `--direct-only`，现有入口仍要求同任务的动力学文件参数。DT-CORL 用 `dt_corl_delay.evaluate_dtcorl_checkpoint`，传同种子的 `--base-checkpoint`（IQL 策略）及 `--dtcorl-checkpoint`（适配器权重）。

### 训练入口

| 需要重训的模型 | 入口和关键输入 |
|---|---|
| IQL / CQL | `envs/research/bin/python -m dt_corl_delay.train_offline --config <..._iql_或_cql_seedNNN.json> --output <新目录>` |
| CADR 动力学 | `envs/research/bin/python -m dt_corl_delay.train_cadr --config <..._cadr_seedNNN.json> --base-checkpoint <同种子IQL权重> --output <新目录>` |
| DT-CORL 适配器 | `envs/research/bin/python -m dt_corl_delay.train_dtcorl_adapter --config <..._dtcorl_seedNNN.json> --base-checkpoint <同种子IQL权重> --output <新目录>` |
| 官方 DP 策略 | `bash scripts/remote/run_maniskill_diffusion_policy.sh <ENV_ID> <seed> <新运行ID>`，需原项目目录和匹配的 DP 环境 |
| CADR-DP 动力学 | `envs/maniskill-dp/bin/python -m dt_corl_delay.train_cadr --config <..._synthetic_seedNNN.json> --output <新目录> --dynamics-only` |

续跑 IQL/CQL、CADR 动力学或支持续跑的阶段时，用原输入、配置和未完成的运行目录。原脚本会核验保存的配置、输入来源及环境相关路径；读取已完成目录时可能直接返回原摘要。补跑评测应另开目录。早期审计快照只有在原配置、来源和训练状态都核实后才用于接续。

## 6. 交接核验

- 155 个优先权重已重新读取并计算 SHA-256，与各运行目录的 `INTEGRITY_MANIFEST.json` 一致。5 条网络回放 CSV 与历史校验记录一致。
- 历史核验见 `artifacts/remote/audits/formal_v1_archive_matrix_final.json`、`formal_v1_semantic_audit_final.json`、`formal_v1_submission_readiness_final.json`。冻结源清单是 `research/formal_v1_source_freeze.json`，封存证据在 `artifacts/source-freeze-evidence/`。
- 本次交接基于本地文件、脚本和历史记录。重新下载的数据和新服务器环境需要在实际恢复时验收。
