# single-arm-mount

PiperX 推荐流程已经同时覆盖 Fold_Box 161044 和 Seal_Bag 161504。`python -m scripts.run_piperx_recommended_v31` 读取完整双臂工厂 CSV，以同一个刚体变换注册左右目标，选择任务级 mount 和固定工具坐标旋转，把完整轨迹重采样到 60 Hz，然后在严格 1 mm / 0.5° 门内运行逐臂 warm-start、必要时全局分支恢复、翻腕候选配对和无损局部动力学重定时。默认目标基准是注册并重采样后的 raw 位姿；Savitzky-Golay SE(3) 条件化只有显式传入 `--condition-targets` 才会启用。

项目把“完全 follow”拆成四个可分别审计的结论：每个 raw 源位姿最终是否严格到达、是否仍在原 60 Hz 时刻到达、重定时执行是否满足关节速度/加速度限制、状态与扫掠边是否无碰撞。运行器输出 JSON、NPZ、MuJoCo 场景、官方 PiperX 网格 MP4 与逐帧 provenance；PDF 构建器重新核对源文件哈希、FK 误差、动力学、碰撞计数和视频终点绑定后才生成报告。`factory_bimanual` 承载这条可审计执行链，`design_optimization` 提供更广的 mount、形态、IK、碰撞、Pareto 与实时控制研究工具。

```powershell
python -m scripts.run_piperx_recommended_v31 `
  --family 8-11/Fold_Box `
  --source-take 161044 `
  --output-dir reports/piperx_two_task_complete_follow/fold_box

python -m scripts.run_piperx_recommended_v31 `
  --family 8-11/Seal_Bag `
  --source-take 161504 `
  --output-dir reports/piperx_two_task_complete_follow/seal_bag

python -m scripts.validate_piperx_two_task_bundle `
  --bundle-root reports/piperx_two_task_complete_follow
```

这三条命令分别复现两个任务并重新校验整个发布包。首次运行先安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 双任务发布结果

同一条证据链把“原始位姿能否严格到达”“原时间戳能否执行”“重定时后能否执行”和“是否无碰撞”分别记账。两个任务合计 2818 个 raw 60 Hz 位姿全部满足 1 mm / 0.5°；原时序都只有首帧满足全部动力学条件，因此正式视频使用受限重定时后的执行时间轴。

| 任务 | 严格 raw 位姿 | 固定原时序 | 重定时执行 | 无碰撞严格覆盖 | 视频 |
|---|---:|---:|---:|---:|---:|
| Fold_Box 161044 | 1061/1061，100% | 1/1061 | 74.285113 s | 990/1061，93.31% | 2230 帧，74.3 s |
| Seal_Bag 161504 | 1757/1757，100% | 1/1757 | 100.632572 s | 1740/1757，99.03% | 3020 帧，100.633 s |

Seal_Bag 的 PDF `horizontal_forward` 基线在第 0 帧不可达。最终方案采用历史构型对比中表现更好的直立挂载，左 base `[-0.35, 0.25, 0.81] m`、右 base `[-0.30, -0.45, 0.81] m`、yaw `15°/15°`，并使用一对整轨迹固定的任务级 `R_tool`。这对旋转只定义 source-hand 到 robot-TCP 的坐标约定，不逐帧改目标，也不启用轨迹平滑。

优先从这些最终工件开始检查：

- [双任务严格完全跟随实验报告](reports/piperx_two_task_complete_follow/PiperX双任务严格完全跟随实验报告.pdf)
- [双任务机器可读 manifest](reports/piperx_two_task_complete_follow/two_task_manifest.json)
- [双任务汇总 CSV](reports/piperx_two_task_complete_follow/two_task_summary.csv)
- [检测与优化实验日志](reports/piperx_two_task_complete_follow/two_task_experiment_log.json)
- [Fold_Box 真实 MuJoCo 视频](reports/piperx_two_task_complete_follow/fold_box/8-11_Fold_Box_161044_complete_follow.mp4)
- [Seal_Bag 真实 MuJoCo 视频](reports/piperx_two_task_complete_follow/seal_bag/8-11_Seal_Bag_161504_complete_follow.mp4)
- [Fold_Box summary](reports/piperx_two_task_complete_follow/fold_box/8-11_Fold_Box_161044_complete_follow.summary.json) 与 [Seal_Bag summary](reports/piperx_two_task_complete_follow/seal_bag/8-11_Seal_Bag_161504_complete_follow.summary.json)

> **Caution:** 100% 严格位姿覆盖不是无碰撞或真机许可。Fold_Box 仍有 71 个碰撞审计帧，Seal_Bag 仍有 17 个；上真机前必须重新规划这些状态与扫掠入边，并重新验证完整覆盖。

## 复现 raw-target 完全跟随

需要复核推荐方案时，从默认命令开始。默认输入是仓库内发布的完整 Fold_Box CSV，输出目录是 `reports/piperx_complete_follow`，验收阈值固定为 1 mm / 0.5°。

```powershell
python -m scripts.run_piperx_recommended_v31 `
  --family 8-11/Fold_Box `
  --source-take 161044 `
  --rate-hz 60 `
  --maximum-candidates 8 `
  --output-dir reports/piperx_complete_follow
```

当前发布 bundle 的结论来自 [`8-11_Fold_Box_161044_complete_follow.summary.json`](reports/piperx_complete_follow/8-11_Fold_Box_161044_complete_follow.summary.json)：

| 审计问题 | 发布结果 | 含义 |
|---|---:|---|
| raw 注册位姿最终到达 | 1061/1061，100% | 左右 TCP 均通过 1 mm / 0.5° 严格 FK 门 |
| 原 60 Hz 时刻同步到达 | 1/1061，0.0943% | 不能把几何可达解释成原速同步执行 |
| 无损局部重定时 | 74.285113 s | 原 17.656956 s 轨迹增加 56.628157 s |
| 插入分支过渡 | 1 frame | 过渡帧不计作 raw strict pose hit |
| 重定时动力学 | PASS | 实测峰值 1.000000 rad/s、4.000000 rad/s² |
| 碰撞源帧 | 71 | 碰撞与位姿是否到达分别记账 |
| 无碰撞严格覆盖 | 990/1061，93.31% | 同时满足 strict pose 与规划器有效碰撞门 |
| 视频 | 30 fps，2230 帧 | 1280×720，编码时长 74.3 s |

> **Caution:** 最终 runner 默认不做目标平滑。只有显式 `--condition-targets` 才允许 5 mm / 1° 的有界 SE(3) 条件化，因此 conditioned 结果不能作为 raw-source 结果发布。

raw target 会在整条轨迹上应用同一个任务级固定 `R_tool`。这个旋转只定义 source-hand 到 robot-TCP 的坐标约定，不逐帧修改、不平滑，也不放宽源位姿。

### 运行器与结果对象

需要在 Python 中接入同一执行链时，`scripts.run_piperx_recommended_v31.run` 负责加载、注册、mount、固定 `R_tool`、raw/可选 conditioned 目标、IK、重定时、审计和视频。`CompleteFollowRunner.run` 返回 `CompleteFollowResult`，其中 pose、timing、dynamics、collision 和 rescue 数组保持分离。

```python
from scripts.run_piperx_recommended_v31 import parse_args, run

options = parse_args([
    "--family", "8-11/Fold_Box",
    "--source-take", "161044",
    "--no-video",
])
paths = run(options)
print(paths["summary"])
print(paths["trajectory"])
```

一键入口还公开 `ROOT`、`DEFAULT_OUTPUT`、`resample_task_60hz`、`smooth_follow_targets`、`condition_complete_follow_targets`、`build_summary`、`build_complete_summary` 和 `main`。应用代码调用 `run`；命令行调用 `main`；报告或自动检查读取 `build_complete_summary` 产生的 schema v2 JSON。

无损 `retime_complete_source_path` 把每个源目标保留为显式终态；插入的 `RETIMED_TRANSITION` 只负责连接分支，不算 strict pose hit。因此 100% `complete_source_pose_coverage` 与 100% `fixed_time_synchronous_coverage` 是两个不同结论。

### 证据包

需要直接检查已发布结果时，优先从最终报告和视频进入，再回到机器可读证据：

- [PiperX 双臂完全跟随优化与验证报告 v4.0](reports/piperx_complete_follow/PiperX双臂完全跟随优化与验证报告_v4.0.pdf)
- [真实 MuJoCo 渲染视频，complete-follow](reports/piperx_complete_follow/8-11_Fold_Box_161044_complete_follow.mp4)
- [逐帧轨迹证据 NPZ](reports/piperx_complete_follow/8-11_Fold_Box_161044_complete_follow.trajectory.npz)
- [视频逐帧 provenance](reports/piperx_complete_follow/8-11_Fold_Box_161044_complete_follow.provenance.json)
- [可复现 MuJoCo 场景](reports/piperx_complete_follow/8-11_Fold_Box_161044_complete_follow.scene.xml)
- [报告与可视化索引](reports/README.md)

历史 recommended-v3.1 基线被保留下来，用于比较优化前后的状态机和严格同步覆盖：

- [PiperX 双臂 IK 跟随优化与验证报告 v3.1](reports/piperx_recommended_v31/PiperX双臂IK跟随优化与验证报告_v3.1.pdf)
- [真实 MuJoCo 渲染视频，recommended-v3.1](reports/piperx_recommended_v31/8-11_Fold_Box_161044_recommended_v31.mp4)
- [recommended-v3.1 汇总指标](reports/piperx_recommended_v31/8-11_Fold_Box_161044_recommended_v31.summary.json)

最终报告由 `scripts.build_piperx_complete_follow_report` 的 `build_report` 生成。该模块还公开 `ARTIFACT_DIR`、`STEM`、`SUMMARY_PATH`、`TRAJECTORY_PATH`、`SCENE_PATH`、`VIDEO_PATH`、`DEFAULT_OUTPUT`、`parse_args` 和 `main`。

```powershell
python -m scripts.build_piperx_complete_follow_report
```

> **Caution:** 最终 PDF 不是任意 summary 的排版器。`build_report` 拒绝 conditioned targets，并重新校验 schema、源 SHA-256、JSON/NPZ 帧数与碰撞计数、1 mm / 0.5° FK 门、1 rad/s / 4 rad/s² 动力学以及视频/场景 provenance。

## 准备权威任务目标

要把另一条工厂轨迹接入求解器，先保留任务身份、完整时间线、左右位姿、valid 与 gripper 通道，再执行一次左右共享的 proper rigid registration。运行域始终是同一个 `FactoryBimanualTask`，不会为左、右臂各自创建不同的世界映射。

```python
from pathlib import Path
import numpy as np

from factory_bimanual.registration import RigidTaskRegistration, register_task
from factory_bimanual.source_data import load_factory_task
from factory_bimanual.task_family import TaskFamily
from scripts.run_piperx_recommended_v31 import resample_task_60hz

family = TaskFamily.parse("8-11/Fold_Box")
task = load_factory_task(
    Path("data/factory/8-11/Fold_Box/handheld_20260811_161044.csv"),
    family.key,
    max_translation_jump_m=0.07,
    repair_invalid_pose_rows=True,
)
registration = RigidTaskRegistration(np.eye(3), np.array([0.0, 0.0, 0.0]))
registered = register_task(task, registration)
targets_60hz = resample_task_60hz(registered, rate_hz=60.0)
print(family.key, len(targets_60hz.time_s), targets_60hz.source_row_index[-1])
```

加载完成后，`factory_bimanual.source_data.FactoryBimanualTask` 保留 `source_path`、`source_row_index`、`time_s`、`coordinate_frame`、左右 position/quaternion、左右 valid 和可选 gripper。注册结果是 `RegisteredBimanualTask`；`RigidTaskRegistration.matrix` 和 `inverse` 分别给出齐次矩阵和逆变换。任务身份由 `factory_bimanual.task_family.TaskFamily.key`、`TaskFamily.parse` 与 `family_from_path` 纳入日期，避免同名任务跨日期共享 mount 或缓存。

> **Caution:** `load_factory_task` 保留权威完整时间线，`resample_task_60hz` 创建新的等间隔目标并显式保留终点。原 CSV 行数、60 Hz pose 数和 execution frame 数是三个不同概念。

### 数据发现与姿态恢复

需要从完整数据树选择代表 take 时，`factory_bimanual.factory_task_catalog` 的 `FactoryEpisode`、`TaskRepresentative`、`ExcludedFactoryInput` 和 `FactoryTaskCatalog` 记录入选与排除原因；`build_factory_task_catalog` 发现任务，`write_dataset_manifest` 写确定性清单。

```python
from pathlib import Path
from factory_bimanual.factory_task_catalog import (
    build_factory_task_catalog,
    write_dataset_manifest,
)

catalog = build_factory_task_catalog(Path("data/factory"))
write_dataset_manifest(catalog, Path("reports/factory_bimanual/dataset_manifest.json"))
```

姿态通道含 held 值时，`factory_bimanual.quaternion_trajectory` 的 `QuaternionReconstruction` 与 `reconstruct_held_quaternions` 按时间戳恢复四元数，`quaternion_poses_equal` 把 `q` 和 `-q` 视为同一姿态。局部遥操作数据进入更广设计研究前，由 `design_optimization.local_pose_dataset` 的 `Hand`、`Split`、`CleanedPoseFrame`、`CleaningRecord`、`infer_hands`、`content_split`、`assign_stratified_splits`、`clean_pose_frame`、`prepare_local_pose_benchmark` 和 `trajectory_has_motion` 完成清洗与内容哈希分割；`design_optimization.local_pose_sampling` 的 `trim_episode_edges` 和 `relative_pose_sample` 提供相对起点采样。

### 可选目标条件化

只有做报告中的对照实验时才启用 `factory_bimanual.trajectory_conditioning` 的 `ConditioningAudit`、`ConditionedTrajectory` 与 `bounded_savgol_se3`：

```powershell
python -m scripts.run_piperx_recommended_v31 --condition-targets --no-video
```

该开关允许的位置偏离上限为 5 mm、姿态偏离上限为 1°，因此输出的 `target_basis` 不再是最终报告要求的 raw 基准。`condition_complete_follow_targets` 会同时返回处理后的任务、映射四元数和实测审计值；默认调用传入 `apply_conditioning=False`。

## 建立 mount、工具坐标与真实场景

从 mount 到 MuJoCo 世界场景需要先确认坐标域。PDF 的 source-frame 基座对通过任务 registration 映射到世界系；已经由优化实验锁定的 `registered_world` 基座直接进入场景。随后把 source-hand 四元数通过任务级固定 `R_tool` 映射到 PiperX TCP 约定。

```python
import numpy as np

from factory_bimanual.registration import RigidTaskRegistration
from factory_bimanual.piperx_recommended import (
    load_recommended_config,
    world_mount_for_family,
)
from factory_bimanual.task_family import TaskFamily

config = load_recommended_config()
family = TaskFamily.parse("8-11/Fold_Box")
registration = RigidTaskRegistration(np.eye(3), np.zeros(3))
mount = world_mount_for_family(
    config,
    family,
    registration.rotation_world_from_vr,
    registration.translation_world_m,
)
print(mount.base_distance_m)
print(mount.xy)
print(mount.as_scene_mount())
```

复现实验需要从一份配置同时取得验收门、DLS 参数、执行约束、mount 漏斗和任务级安装记录。发布配置由 `DEFAULT_CONFIG_PATH` 定位，`PiperXRecommendedConfig` 聚合 `StrictAcceptConfig`、`DLSConfig`、`ExecutionConfig`、`MountFunnelConfig` 与 `RecommendedMountSpec`；`WorldMount.base_distance_m`、`xy` 和 `as_scene_mount` 再把注册后的安装结果交给场景与 summary。frame-0 的 anchor restarts 固定为 40，loader 会拒绝其他值，从而锁定复现实验预算。

> **Caution:** `source_frame` mount 必须先经过任务刚体注册，且左右基座注册后必须保持同一高度。Seal_Bag 的最终记录明确标成 `registered_world`，不得再次注册；loader 会按 `coordinate_domain` 区分两者。

### 固定工具坐标

当源手系与机器人 TCP 的轴约定不一致时，应先标定一个整轨迹固定的旋转，再开始 mount 或 IK 调优。标定过程枚举 proper-axis 候选、在代表姿态上排名，并把左右四元数连同源文件指纹持久化；局部精化被限制在已审计角度范围内。

<details>
<summary>工具坐标 API reference</summary>

`factory_bimanual.tool_frame_calibration` 提供 `proper_axis_rotations`、`fixed_offset_quaternion`、`representative_quaternion_indices`、`apply_fixed_tool_rotation`、`rank_calibration_result`、`CalibrationArtifact.write`、`CalibrationArtifact.read`、`source_file_fingerprint`、`CALIBRATION_TASKS`、`LOCAL_REFINEMENT_LIMIT_DEG` 和 `validate_local_refinement_deg`。

</details>

```powershell
python -m scripts.calibrate_piperx_tool_frames
python -m scripts.diagnose_piperx_complete_follow --help
python -m scripts.refine_piperx_complete_follow_mount
```

三个维护入口分别是 `scripts.calibrate_piperx_tool_frames.calibrate`、`scripts.diagnose_piperx_complete_follow.run` 和 `scripts.refine_piperx_complete_follow_mount.run`。其 `main` 提供命令行封装。

### 官方模型与双臂场景

当 mount 和工具坐标已经确定，场景必须从同一份官方单臂模型复制左右实例。这样关节限位、网格、TCP 父链接和命名规则只有一个权威来源，后续 FK、碰撞与视频才能复核同一个机器人契约。

```python
from pathlib import Path
import numpy as np

from factory_bimanual.piperx_recommended import (
    load_recommended_config,
    world_mount_for_family,
)
from factory_bimanual.registration import RigidTaskRegistration
from factory_bimanual.robot_contracts import get_robot_contract
from factory_bimanual.scene_builder import build_same_model_scene
from factory_bimanual.task_family import TaskFamily

contract = get_robot_contract("piperx")
config = load_recommended_config()
mount = world_mount_for_family(
    config,
    TaskFamily.parse("8-11/Fold_Box"),
    np.eye(3),
    np.zeros(3),
)
manifest = build_same_model_scene(
    contract,
    mount.base_distance_m,
    Path("reports/piperx_complete_follow/reproduction.scene.xml"),
    table_height_m=0.75,
    mount_xy_m=mount.xy,
    mount_yaw_deg={"left": 0.0, "right": 0.0},
    mount_adapter_height_m=mount.shared_base_z_m - 0.75,
    mount_support_mode=mount.mode,
)
print(manifest.left_joint_names)
```

比较 upright、horizontal 或 inverted 安装时，先统一模式四元数和最小基座间距，再用廉价拓扑门排除明显交叉，最后对状态和大幅扫掠边运行 MuJoCo。这个顺序保证不同安装方案使用相同物理约束，同时把昂贵检查留给有效候选。

<details>
<summary>模型与场景 API reference</summary>

`factory_bimanual.robot_contracts` 用 `Side`、`BimanualRobotContract`、`BimanualRobotContract.dof_per_arm`、`prefixed_joint_names`、`ROBOT_CONTRACTS` 和 `get_robot_contract` 固化模型契约；`factory_bimanual.scene_builder` 用 `SceneManifest` 与 `build_same_model_scene` 生成双臂场景。

`factory_bimanual.mount_orientation` 提供 `MOUNT_MODES`、`SUPPORTED_MOUNT_MODES`、`FACTORY_BATCH_MOUNT_MODES` 和 `mount_quaternions`；`factory_bimanual.mount_constraints.MINIMUM_BIMANUAL_BASE_SEPARATION_M` 统一最小间距；`factory_bimanual.mount_topology` 提供 `MountTopologyConfig`、`MountTopologyReport`、`evaluate_mount_topology_positions` 与 `MuJoCoMountTopologyChecker.state/transition`。

</details>

## 求解 IK、连续翻腕与分支恢复

每个新源帧先从上一条已选择指令运行一次局部 DLS。局部解失效，或局部双臂配对存在碰撞时，求解器才扩展全局分层种子，然后按状态碰撞、扫掠边碰撞、branch guard、腕部风险、pose cost 和关节余量排序。

```python
from scripts.run_piperx_recommended_v31 import parse_args, run

paths = run(parse_args(["--no-video"]))
print(paths["summary"])
```

稳定跟踪阶段只从上一条指令运行一次 DLS，不混入 global/rolling seeds，也不改变种子库。局部解缺失或当前双臂配对碰撞时，runner 才扩展双方全局分层候选；两侧严格候选全部耗尽时，异常保留精确 source row，便于针对失败窗口诊断。

<details>
<summary>候选生成 API reference</summary>

`factory_bimanual.mujoco_candidate_generator.MuJoCoCandidateGenerator` 通过 `MuJoCoCandidateGenerator.reset`、`generate_reference_candidate`、`generate_warm_start_candidate` 和 `generate_target` 控制参考、局部与全局搜索。`CandidateGeneratorConfig` 配置严格 FK、DLS、分层种子、受约束补救和去重；`normalized_pose_residual`、`stratified_joint_seeds` 与 `piperx_wrist_risk` 提供评分原语。`CompleteFollowInfeasibleError` 记录最终失败帧。

</details>

### 翻腕连续性

周期关节不能用普通减法判断跳变。求解器先把关节差解包到最短周期方向，再用腕部分支签名区分真正换支与 `2π` 等价解；接近腕部临界角的候选在排序中承担更高风险。需要复现旧版 FOLLOW/HOLD/RESCUE 时，再使用旧调度原语和最小加加速度过渡，而不是把旧状态机混入 complete-follow 的主结论。

当两套初始 IK 只相差周期代表时，应先折叠为同一分支；当任务需要通用离线 beam search 时，再选择 strict-bimanual 接口。两者解决的是分支身份和整轨迹搜索两个不同问题。

<details>
<summary>翻腕与严格路径 API reference</summary>

`factory_bimanual.rescue_v31` 提供 `StrictGate.accepts`、`pose_accepts`、`shortest_joint_delta`、`wrist_branch_signature`、`trapezoidal_transition_time`、`minimum_jerk_transition`、`CandidateFrame.wrist_signature`、`RescueScheduleConfig`、`RescueEvent`、`FollowSchedule`、`StateValid`、`TransitionValid` 和 `schedule_rescue_v31`。历史对照由 `factory_bimanual.recommended_follow.RecommendedFollowRunner.run` 与 `RecommendedFollowResult` 保存。

`factory_bimanual.branch_equivalence` 提供 `ArmPair`、`BranchDecision` 与 `compare_initial_branches`；`factory_bimanual.strict_bimanual_ik` 提供 `IKCandidate`、`CandidateGenerator`、`BimanualIKConfig`、`BimanualIKResult` 和 `solve_strict_bimanual_path`。

</details>

### 姿态让步与安全替代

严格 raw 路径不能满足安全目标时，必须显式切换协议。先在软/硬限内生成可审计的 TCP 姿态替代候选，再用 exact-first 容差层协调双臂让步；新的 target basis 和协议名称必须进入 summary，避免让步结果冒充 raw follow。

这类替代路径不属于本页 1061/1061 raw 结果，summary 必须使用不同的 target basis 和协议名称。

<details>
<summary>安全替代 API reference</summary>

`factory_bimanual.bounded_orientation_adaptation` 提供 `OrientationAdaptationConfig`、`OrientationCandidate`、`orientation_offset_angles` 和 `orientation_adaptation_candidates`；`factory_bimanual.collision_safe_follow` 提供 `PoseToleranceTier`、`DEFAULT_TIERS`、`CandidateProvider`、`CollisionSafeFollowConfig`、`CollisionSafeFollowResult` 和 `solve_collision_safe_follow`。

</details>

## 无损局部动力学重定时

当源关节路径在 60 Hz 下违反 0.3 rad branch guard、1 rad/s 速度或 4 rad/s² 加速度限制时，`retime_complete_source_path` 只拉长违规的局部片段，并保留每个 raw source pose 为显式终态。

```python
import numpy as np

from factory_bimanual.complete_follow import retime_complete_source_path

source_q = np.array([[0.0, 0.0], [0.4, 0.1]])
source_time_s = np.array([0.0, 1.0 / 60.0])
retimed = retime_complete_source_path(
    source_q,
    source_time_s,
    periodic=np.array([True, False]),
    branch_guard_rad=0.3,
    maximum_velocity_rad_s=1.0,
    maximum_acceleration_rad_s2=4.0,
)
print(retimed.inserted_transition_frames, retimed.execution_time_s[-1])
```

重定时结果必须同时保留 source→execution 映射、execution state、插帧数、cycle delay、time scale 和实测导数峰值。动力学修复只拉长违规局部片段，不删除配置，也不替换已选择分支；发布路径插入 1 个 transition，最终时长为 74.285113 s。`CompleteRetiming` 用 `execution_q`、`execution_time_s`、`execution_source_index`、`execution_state`、`source_execution_index`、`source_reached` 和 `fixed_time_accepted` 保存这些证据。

### 固定时刻与通用重定时接口

需要先尝试不改时间戳的关节投影时，`factory_bimanual.fixed_time_refinement` 的 `FixedTimeRefinementResult` 与 `refine_fixed_time_joint_path` 在硬导数限制下细化路径。`factory_bimanual.fixed_time_run_contract` 的 `source_time_schedule` 验证严格递增的源时间，`validate_synchronized_mount` 检查共享高度、直立与最小间距。`design_optimization.retiming` 的 `RetimedPath` 和 `retime_joint_path` 提供只拉伸时间、保持所有配置和分支的通用版本。

发布 summary 同时记录原时路径的 406/1060 速度边通过、12/1061 加速度结点通过、22.679529 rad/s 峰值和 2175.658796 rad/s² 峰值。重定时把零速度起步和零速度停止边界纳入 4 rad/s² 门，执行速度与加速度门为 PASS；这不改变原时同步只有 1/1061 的结论。

## 区分位姿成功、原时同步与碰撞安全

读取结果时先选择正确的问题。`source_reached` 回答 raw pose 最终是否到达；`fixed_time_accepted` 回答该 pose 是否仍在原时间戳到达；执行导数回答重定时路径是否满足关节限制；`source_collision` 和 `execution_collision` 回答状态与扫掠边是否通过规划器碰撞门。执行域还把 `execution_state_collision` 与 `execution_incoming_transition_collision` 分开保存，避免把某个 knot 的状态碰撞和进入该 knot 的扫掠边碰撞混为一谈。

```python
import json
from pathlib import Path

summary = json.loads(Path(
    "reports/piperx_complete_follow/"
    "8-11_Fold_Box_161044_complete_follow.summary.json"
).read_text(encoding="utf-8"))
metrics = summary["metrics"]
assert metrics["complete_source_pose_frames"] == 1061
assert metrics["fixed_time_synchronous_frames"] == 1
assert metrics["retimed_execution_dynamic_limits_passed"] is True
assert metrics["collision_frames"] == 71
```

碰撞审计需要同时覆盖单臂状态、双臂状态和相邻配置之间的扫掠边。通用回调协议负责分类与 valid 判定，最终 MuJoCo 适配器负责具名间隙和模型几何；两层分工让搜索器可以更换碰撞实现，而发布证据始终回到官方网格场景。

<details>
<summary>碰撞审计 API reference</summary>

`factory_bimanual.bimanual_collision` 提供 `CollisionClass`、`CollisionReport.valid`、`PairState`、`PairEvaluator`、`TransitionEvaluator`、`CallbackCollisionChecker.state` 和 `CallbackCollisionChecker.transition`。`factory_bimanual.mujoco_collision_adapter` 提供 `ClearanceReport`、`MuJoCoPairedCollisionChecker.clearance`、`transition_clearance`、`state`、`side_state`、`side_transition` 和 `transition`。

</details>

> **Caution:** 候选排序虽优先无碰撞状态和边，并在碰撞时扩展双方全局候选，但 `source_reached` 不会因碰撞自动变为 false；安全结论必须读取 `collision_free_strict_coverage`。规划器还会忽略已审计的相连底座接触和数值接触，因此“零 MuJoCo contact”也不等于“零有效碰撞”。

### 指标与保守排名

跨构型排序时，先从不可变 NPZ 派生同口径指标，再把失解、碰撞、跳变和姿态误差按固定失败顺序归并。排序可以给出 shortlist，却不能把不同失败原因压成一个“成功率”而丢失安全含义。

在进入昂贵搜索前，可以用刚体变换不变的轨迹包络做必要工作空间否证。采样 reach 只负责淘汰明显不可达候选，不证明完整 IK、安全边或动态可执行性。

<details>
<summary>指标与预筛 API reference</summary>

`factory_bimanual.orientation_comparison_metrics.derive_orientation_metrics` 派生安装朝向指标；`factory_bimanual.per_task_mount_report` 提供 `METRICS`、`build_metric_rows` 和 `write_comparison_metrics`。`design_optimization.episode_follow_metrics` 提供 `FAILURE_REASON_ORDER`、`classify_solver_failures`、`planner_failure_diagnostics`、`failure_reason_diagnostics`、`episode_follow_metrics` 与 `follow_rank`。

`factory_bimanual.workspace_feasibility` 提供 `TrajectoryGeometry`、`trajectory_geometry` 与 `audit_workspace_feasibility`。

</details>

## 渲染、报告与 provenance 验证

证据输出始终围绕同一个 `CompleteFollowResult`。`factory_bimanual.artifacts` 的 `FrameDiagnostics` 记录 source row、execution time、状态、碰撞和来源，`RunArtifactPaths` 与 `RunArtifactWriter.write_run` 把 JSON/NPZ/报告限制在指定根目录并原子写入。

```python
from pathlib import Path
from factory_bimanual.video import decode_check_mp4

video = Path(
    "reports/piperx_complete_follow/"
    "8-11_Fold_Box_161044_complete_follow.mp4"
)
check = decode_check_mp4(
    video,
    expected_frames=2230,
    expected_resolution=(1280, 720),
    expected_duration_s=74.3,
)
assert check.fps == 30.0
print(check.frame_count, check.fps, check.duration_s)
```

要让视频成为执行证据，渲染必须使用重定时 execution timeline，并在线性插值后的关节状态上绘制官方 PiperX 网格。最终编码序列重构的最大关节速度为 1.00000000000005 rad/s，与 1 rad/s 门在数值容差内一致；末编码帧强制绑定精确源终点，容器时长的一帧内量化由 provenance 说明。插值中的编码帧以 upper knot 记录的 incoming edge 作为碰撞审计域，精确 knot 则只读自身 state collision；sidecar 的 `audit_execution_index` 与 `audit_scope` 使两者可逐帧区分。

summary 的 `artifacts` manifest 为 trajectory NPZ、scene XML、video MP4 与 provenance JSON 分别保存路径、大小和 SHA-256。报告构建器以这 4 个哈希把输入绑定为同一批证据。

<details>
<summary>渲染与 provenance API reference</summary>

`factory_bimanual.video` 提供 `RealtimeVideoTiming`、`VideoRenderConfig`、`interpolation_sample`、`is_replanned_transition`、`VideoDecodeCheck`、`VideoRenderResult`、`build_realtime_timing`、`write_video_provenance`、`decode_check_mp4` 和 `render_mujoco_mp4`。schema v2 旁车以 execution timeline 为权威时间域，并记录 knot/edge 分域的碰撞证据。

</details>

> **Caution:** `RETIMED_TRANSITION`、`FOLLOW_RETIMED` 与碰撞审计由 execution-state 文本显式标记。这些帧不能在视频 UI 中写成 HOLD 或普通 tracking。

### 运行边界与交互检查

启动批量矩阵前，先用 preflight 检查模型、控制器、任务和间距证据，再用 dry-run 或 short-prefix 检查编排边界。full 只负责明确确认后的完整执行；受保护文件快照用于证明双臂实验没有改写单臂流水线。

```powershell
python -m factory_bimanual.cli preflight
python -m factory_bimanual.cli dry-run
python -m factory_bimanual.cli full --confirm-full
```

> **Caution:** 通用实验入口的 full matrix 必须显式传入 `--confirm-full`。preflight 只检查证据、模型和输出边界，不会顺便启动实验。

需要交互检查生成场景时，可以打开双臂 MuJoCo viewer；需要观察相对位姿映射时，可以运行只读四格参考播放。四格工具不写最终 evidence，也不替代 complete-follow renderer。

通用比较矩阵、生产 executor 和机器人×任务×控制器编排用于扩展实验；最终 complete-follow PDF 仍只接受前述带一致性门控的 `build_report`。两类报告面向不同证据契约，不能互换输入。

<details>
<summary>运行与编排 API reference</summary>

`factory_bimanual.cli.main` 提供 `preflight`、`dry-run`、`short-prefix` 和 `full`。`factory_bimanual.preflight` 提供 `validate_spacing_evidence` 与 `run_preflight`；`factory_bimanual.isolation` 提供 `protected_paths`、`snapshot_protected_files` 和 `assert_protected_files_unchanged`。

`factory_bimanual.launch_viewer.main` 打开交互场景。`factory_bimanual.four_cell_viewer` 提供 `WORKSPACE_ROOT`、`REFERENCE_ROOT`、`REFERENCE_SCENE`、`DEFAULT_TASKS`、`assert_reference_demo_only`、`load_reference_model`、`lightweight_scene_text`、`mapped_relative_pose`、`LightweightFourCellIK.step` 和 `main`。

`factory_bimanual.report` 提供 `ReportPaths` 与 `write_comparison_report`；`factory_bimanual.executors` 提供 `ProductionExecutors.mapping`、`load_task`、`ik` 和 `mpc`；`factory_bimanual.run_experiment` 提供 `ROBOTS`、`TASKS`、`IK_MODES`、`ExperimentConfig`、`ExperimentJob`、`Executor` 和 `FactoryBimanualExperiment.run`。

</details>

## 搜索 mount 与扩展研究

当推荐构型不适用于新任务时，搜索从廉价必要条件开始，再进入严格全轨迹验证。不要用 coarse reach、学习式代理或前缀通过代替最终 MuJoCo FK、碰撞和执行审计。

### 任务级安装筛选

机器人形态不变、只有新任务或新安装位姿时，优先做任务级筛选。所有候选必须共享任务集合、姿态模式、间距约束和缓存指纹；前缀只负责淘汰，最终排名必须回到完整轨迹，失败窗口再用于局部加密。

```python
from pathlib import Path

from factory_bimanual.source_data import load_factory_task
from factory_bimanual.orientation_mount_search import (
    OrientationSearchConfig,
    generate_mounts,
)

task = load_factory_task(
    Path("data/factory/8-11/Fold_Box/handheld_20260811_161044.csv"),
    "8-11/Fold_Box",
    max_translation_jump_m=0.07,
    repair_invalid_pose_rows=True,
)
candidates = generate_mounts("upright_table", task, OrientationSearchConfig())
print(len(candidates), candidates[0]["mode"])
```

<details>
<summary>任务级 mount API reference</summary>

`factory_bimanual.defaults` 提供 `ROOT`、`derive_initial_registration`、`TaskSpec.registration`、`TASK_SPECS`、`SPACING_CANDIDATES_M`、`SELECTED_SPACING_M`、`CONTROLLER_PROFILES_PATH` 和 `OUTPUT_ROOT`。`factory_bimanual.spacing_scan` 提供 `SpacingCandidateRow`、`SpacingSelection`、`generate_spacing_candidates` 和 `rank_spacing_candidates`。

`factory_bimanual.orientation_mount_search` 提供 `OrientationSearchConfig`、`generate_mounts`、`mount_fingerprint`、`evenly_spaced` 和 `MOUNT_FINGERPRINT_SCHEMA`；`factory_bimanual.per_task_mount_search` 提供 `PerTaskSearchConfig`、`load_registered_representative`、`prefix_registered_task`、`candidate_fingerprint`、`rank_full_finalist`、`summarize_quality_arrays` 和 `select_safe_layout`；`factory_bimanual.staged_mount_search` 提供 `rank_full_fixed_time_mount`、`select_full_fixed_time_mount`、`targeted_source_indices` 与 `MINIMUM_COLLISION_SAFE_BASE_SEPARATION_M`。

</details>

### 批量机器人设计核心

只有任务级 mount 已无法覆盖目标空间时，才把问题扩展为形态与安装联合搜索。设计候选必须继承已审计拓扑、关节范围和 TCP 变换；批量 FK/Jacobian 用于筛选，最终路径仍由多分支 IK 与真实碰撞审计确认。

```python
import torch

from design_optimization import TopologyTemplate
from design_optimization.kinematics import build_designs, fk_tcp

template = TopologyTemplate(
    name="one_joint_demo",
    axes=torch.tensor([[0.0, 0.0, 1.0]]),
    deltas=torch.tensor([[0.25, 0.0, 0.0]]),
    home_rotation=torch.eye(3),
    q_min=torch.tensor([-3.14]),
    q_max=torch.tensor([3.14]),
)
designs = build_designs(template, torch.zeros((1, 1)), torch.zeros(1))
tcp = fk_tcp(designs, torch.zeros((1, 1)))
print(tcp.shape)
```

包根只懒加载三个入口，因此导入轻量搜索工具不会顺带初始化 Torch GPU/OpenMP 运行时。多起点 DLS 保留全部分支供连续性选择；胶囊代理和学习式分类器只负责高吞吐预筛，最终结论仍回到 MuJoCo 碰撞体。

<details>
<summary>设计、IK 与碰撞 API reference</summary>

`design_optimization` 根导出 `DesignBatch`、`TopologyTemplate` 和 `load_templates`。`design_optimization.topology` 提供 `TopologyTemplate.dof/to`、`DesignBatch.count` 与 `load_templates`；`design_optimization.kinematics` 提供 `axis_angle_matrix`、`build_designs`、`assemble_from_deltas`、`fk_flange`、`fk_tcp`、`geometric_jacobian`、`joint_world_positions` 和 `deterministic_joint_samples`。

`design_optimization.ik` 提供 `IKResult`、`SelectedIKPath`、`reverse_ik_time`、`concatenate_ik_branches`、`rotation_log`、`pose_error`、`adaptive_damping`、`joint_limit_centering_velocity`、`solve_multistart`、`deterministic_seeds`、`solve_trajectory_multistart` 和 `select_continuous_branches`。

`design_optimization.collision` 提供 `DEFAULT_LINK_RADII_M`、`CollisionMetrics`、`segment_segment_distance`、`self_segment_pair_distances`、`self_capsule_clearance`、`dual_capsule_clearance`、`table_capsule_clearance` 和 `collision_penalty`；`design_optimization.collision_classifier` 提供 `periodic_joint_features`、`ConfigurationCollisionClassifier.forward` 与 `binary_metrics`。

</details>

### 多保真搜索与 Pareto 选择

候选数量超过完整验证预算时，按候选×帧×种子规模分批，并让 coarse、representative-window、strict-full 逐级晋级。全局 Sobol 保证覆盖，局部扩展开发优胜区域，跨保真排名一致性决定 shortlist 是否可信。

多个目标同时存在时，硬约束先于 Pareto 排名；非支配排序和拥挤距离只比较可行候选。鲁棒扰动、上尾 CVaR、代理不确定性和短窗口 SLP 分别处理安装偏差、尾部风险、探索预算与局部精化，不能替代最终 strict validation。

<details>
<summary>多保真与 Pareto API reference</summary>

`design_optimization.batched_search` 提供 `safe_candidate_batch_size` 和 `evaluate_candidate_batches`。`design_optimization.best_first_mount_search` 提供 `Fidelity`、`SearchBudget`、`CandidateState`、`Frontier.push/pop`、`promotion_target`、`should_stop_after_success`、`representative_frame_indices`、`representative_frame_windows`、`select_diverse_regions` 和 `generate_local_population`。

`design_optimization.hierarchical_mount_search` 提供 `MountSearchBudget`、`global_mount_candidates`、`diverse_region_indices` 与 `local_mount_candidates`；`design_optimization.installation_search_space` 提供 `InstallationSearchSpace`、`first_version_bounds`、`mount_rotation_matrix`、`tabletop_mount_feasible`、`mount_transform_from_record` 与 `torch_mount_rotation_matrices`；`design_optimization.mount_sweep` 提供 `sobol_mount_candidates` 与 `mount_quality_score`；`design_optimization.search_policy` 提供 `dimension_aware_candidate_budget` 与 `staged_sobol_candidates`。

`design_optimization.incremental_mount_evaluator` 提供 `FrameResult`、`CandidateEvaluation`、`Solver` 与 `IncrementalMountEvaluator.evaluation/evaluate`；`design_optimization.fidelity_funnel` 提供 `select_diverse_candidates` 和 `cross_fidelity_metrics`。

`design_optimization.objectives` 提供 `PopulationMetrics`、`topology_intersection_error`、`motor_clearance_violation` 与 `evaluate_population`；`design_optimization.pareto` 提供 `ParetoRanking`、`constraint_dominates`、`nondominated_sort`、`crowding_distance`、`rank_population`、`nsga2_select`、`tournament_parents` 和 `make_offspring`；`design_optimization.robustness` 提供 `symmetric_sobol_perturbations` 和 `upper_tail_cvar`；`design_optimization.reach.numerical_max_flange_reach` 独立估计最大法兰 reach。

`design_optimization.surrogate` 提供 `ObjectiveSurrogate.forward`、`gaussian_nll`、`EnsemblePrediction`、`ensemble_predict`、`standardize_targets` 和 `propose_ucb_candidates`；`design_optimization.slp_refinement` 提供 `SLPRefinementResult` 和 `refine_joint_window_slp`。

</details>

### 数据、任务空间与原生链

跨数据集或跨机器人比较时，先把 pose episode 分割与采样固定，再把双臂目标转换到各自基座系。工具长度、锁定关节和 mount 旋转顺序必须来自版本化契约；原生 URDF/MJCF 链用于独立 FK/reach 交叉检查，轨迹层再加入外肘偏好和碰撞感知选择。

<details>
<summary>数据与原生模型 API reference</summary>

`design_optimization.egodex` 提供 `EgoDexSplit` 与 `EgoDexPoseDataset.split/sample_frames/sample_indices/episode/statistics`；`design_optimization.taskspace` 提供 `BimanualTaskFrames`、`mirrored_mount_transforms`、`quaternion_wxyz_to_matrix`、`make_transform`、`canonical_egodex_targets`、`world_to_base` 和 `world_to_base_population`。

`design_optimization.robot_contract` 提供 `COMMON_TOOL_LENGTH_M`、`DEFAULT_FLANGE_TCP_TRANSLATION_M`、`PANDA_LOCKED_J3_RANGE_RAD` 与 `MOUNT_ROTATION_ORDER`；`design_optimization.robot_registry` 提供 `RobotSpec.active_dof` 和 `load_robot_registry`；`design_optimization.urdf_chain` 提供 `NativeSerialChain.dof`、`load_urdf_chain`、`load_mjcf_chain`、`load_native_chain`、`fk_flange` 与 `sampled_maximum_reach`。

`design_optimization.trajectory` 提供 `BimanualTrajectory.metrics`、`outer_elbow_penalty` 和 `select_bimanual_collision_aware`。

</details>

### 控制器与实时预算

要比较离线 IK、MPC 与 QP-servo，三者必须使用同一个场景契约、目标接口和源时序。原生 DOF runner 负责模型兼容边界和碰撞诊断；实时层把求解 deadline、最新 branch proposal 和笛卡尔目标治理拆开，避免后台搜索阻塞高频 loop。

学习式分支策略只生成候选。其输出必须继续经过受约束 IK/MPC 和碰撞门，不能直接成为控制命令。

<details>
<summary>控制器与实时 API reference</summary>

`factory_bimanual.controller_adapter` 提供 `make_easyik`、`make_mpc` 与 `set_bimanual_targets`。`factory_bimanual.easyik_runner` 提供 `EasyIKBudget`、`EasyIKScene`、`EasyIKResult` 和 `run_easyik_task`；`factory_bimanual.mpc_runner` 提供 `MPCScene`、`BimanualMPCResult`、`scheduled_mpc_runs` 与 `run_mpc_task`。

`factory_bimanual.native_dof_mpc` 提供 `UnsupportedModelControllerContract`、`MPCConfig`、`ArmState`、`NativeDofDualArmMPCPVT.initialize_home/collision_diagnostics/step_arm/step` 和 `NativeDofDualArmQPServoPVT.step_arm`；`factory_bimanual.native_dof_easyik.NativeDofDualArmEasyIKPVT.initialize_home/step_arm/step` 复用同一 PVT 框架。

`design_optimization.realtime` 提供 `ServoMode`、`RealtimeBudget.solve_deadline_s`、`ServoObservation`、`SolveSchedule`、`BranchProposal`、`BranchProposalMailbox.publish/latest`、`schedule_bounded_solve` 和 `govern_cartesian_target`；`design_optimization.branch_policy` 提供 `BranchPolicyOutput`、`MultiBranchIKPolicy.forward`、`wrapped_joint_error` 与 `branch_imitation_loss`。

</details>

### 实验复现工具

当搜索规模超过单次命令时，把数据、IK、搜索和约束写入版本化配置，并固定随机种子、checkpoint、JSONL 日志与输入哈希。这样中断恢复和结果合并不会改变候选身份。

其余 `scripts/*` 命令覆盖官方模型审计、轨迹清洗与重采样、mount 搜索、IK/MPC/EasyIK/servo 实验、缓存与 provenance 检查、绘图、报告和批量 MuJoCo 渲染。应用作者从命令入口运行这些工作流，脚本内部以下划线开头的函数不是稳定公共接口。

<details>
<summary>实验复现 API reference</summary>

`design_optimization.experiment` 提供 `DataConfig`、`IKConfig`、`SearchConfig`、`ConstraintConfig`、`ExperimentConfig.from_yaml/write`、`seed_everything`、`save_checkpoint_atomic`、`load_checkpoint`、`append_jsonl`、`sha256_file` 和 `reproducibility_manifest`。

</details>

## 输入报告

四份方案与对比报告按原文件名保留，README 中的推荐参数和结论可回溯到这些输入与最终 evidence bundle：

- [双臂 IK 跟随方案四臂四种安装位姿报告](双臂IK跟随方案四臂四种安装位姿报告.pdf)
- [PiperX 双臂构型对比，26 条轨迹综合分析](<PiperX 双臂构型对比 · 综合分析报告（26 条轨迹） - 飞书云文档.pdf>)
- [Seal_Bag 轨迹上 PiperX 双臂三种安装构型对比](<Seal_Bag 轨迹上 PiperX 双臂三种安装构型的跟随能力对比 - 飞书云文档.pdf>)
- [单轨迹优化分析](单轨迹优化分析.pdf)

## 验证

代码修改后先运行测试，再复现最终证据。`--no-video` 适合快速验证求解和 summary；正式发布不传该开关。

```powershell
python -m pytest -q
python -m scripts.run_piperx_recommended_v31 --no-video
python -m scripts.run_piperx_recommended_v31
python -m scripts.build_piperx_complete_follow_report
```

最终发布以 raw 注册目标、summary schema v2、NPZ 逐帧数组、scene manifest、30 fps/2230 帧 MP4、视频旁车与 PDF 的交叉一致性为准。
