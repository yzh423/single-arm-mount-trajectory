# single-arm-mount

这里的第一类工作流从一条可复现实验命令开始：选择任务族和源 take，完成双手原始轨迹注册与 60 Hz 重采样，把源手位姿通过任务级 source-hand-to-PiperX-TCP 标定映射为跟踪目标，在状态与扫掠边硬碰撞门下求解成对严格 IK，再按机器人动力学限制无损重定时，最后输出 MuJoCo 视频和带哈希的 JSON、NPZ、XML 证据。

“完全跟随”只表示标定 TCP 目标在 1 mm / 0.5° 门限内全部到达。原始手轨迹是否被保留、原始时间戳是否可执行、重定时后的执行时长、状态碰撞与扫掠边碰撞，都是独立可审计的结论。仓库还包含 13 个 vendor-native 机械臂模型的单臂证据工作流，模型、网格、TCP、关节限位和来源修订必须先通过资格门，再进入轨迹求解与真实网格渲染。

```powershell
python -m scripts.run_piperx_recommended_v31 `
  --family 8-11/Fold_Box `
  --source-take 161044 `
  --maximum-candidates 16 `
  --output-dir reports/piperx_two_task_complete_follow/fold_box
```

上面的命令只重跑 Fold_Box。发布前还要重跑 Seal_Bag，并对两项任务执行完整 bundle 验证。最终已验证结果如下：

| 任务 | 标定 TCP 严格覆盖 | 原始时长 | 重定时执行时长 | 视频 | MuJoCo 碰撞 |
|---|---:|---:|---:|---:|---:|
| Fold_Box 161044 | 1061 / 1061 | 17.656956 s | 73.950623 s | 2220 帧，30 fps | 0 |
| Seal_Bag 161504 | 1757 / 1757 | 29.261361 s | 100.711602 s | 3022 帧，30 fps | 0 |

> **Caution:** 这里的零碰撞是当前 MuJoCo 模型对源结点、源扫掠入边、执行结点和执行扫掠入边的审计结果，不是真机安全许可。夹具、工件、线缆、外参误差、柔顺性和真实控制器时序仍需单独验证。

### Fixed-time 原始节奏版本

需要查看原始采集节奏时，使用独立的 fixed-time bundle。它逐元素复用原始 `source_time_s` 与 `source_qpos`，不调用重定时、不插入过渡帧，也不增加时长：

```powershell
python -m scripts.build_piperx_two_task_fixed_time_bundle
python -m scripts.build_piperx_two_task_fixed_time_bundle --validate-only
python -m scripts.build_piperx_fixed_time_report
```

| 任务 | fixed-time 时长 | 标定 TCP 严格覆盖 | 碰撞帧 | 视频 |
|---|---:|---:|---:|---:|
| Fold_Box 161044 | 17.656956 s | 1061 / 1061 | 0 | [MP4](reports/piperx_two_task_fixed_time/fold_box/8-11_Fold_Box_161044_fixed_time.mp4) |
| Seal_Bag 161504 | 29.261361 s | 1757 / 1757 | 0 | [MP4](reports/piperx_two_task_fixed_time/seal_bag/8-11_Seal_Bag_161504_fixed_time.mp4) |

[Fixed-time 报告](reports/piperx_two_task_fixed_time/PiperX双任务Fixed-Time完全跟随报告.pdf)和 [manifest](reports/piperx_two_task_fixed_time/fixed_time_manifest.json)记录时间/qpos 恒等性、哈希及逐帧解码结果。

> **Caution:** fixed-time 版本只证明原始节奏下的运动学跟随。Fold_Box 峰值为 13.924315 rad/s、799.124061 rad/s²，Seal_Bag 峰值为 8.815960 rad/s、543.330325 rad/s²，均不满足当前 1 rad/s 与 4 rad/s² 动力学限制，不能直接下发真机。

## Reproducing a PiperX task

当目标是重现一个任务而不是手工拼装内部对象时，从 `scripts.run_piperx_recommended_v31` 的命令行入口开始。入口按顺序加载源轨迹、注册、重采样、解析 mount、构建双臂场景、应用工具映射、运行严格 IK、重定时、渲染并写出证据。

### 选择任务与输出策略

`scripts.run_piperx_recommended_v31.parse_args` 定义任务族、take、采样率、候选预算、mount override、工具旋转 override、目标 conditioning 和视频策略。`scripts.run_piperx_recommended_v31.run` 执行一次完整任务，`scripts.run_piperx_recommended_v31.main` 则把命令行参数交给 `run` 并打印输出路径。

Fold_Box 与 Seal_Bag 的推荐重跑命令是：

```powershell
python -m scripts.run_piperx_recommended_v31 `
  --family 8-11/Fold_Box `
  --source-take 161044 `
  --maximum-candidates 16 `
  --output-dir reports/piperx_two_task_complete_follow/fold_box

python -m scripts.run_piperx_recommended_v31 `
  --family 8-11/Seal_Bag `
  --source-take 161504 `
  --maximum-candidates 16 `
  --output-dir reports/piperx_two_task_complete_follow/seal_bag
```

`--no-video` 只适合求解诊断。最终发布任务不能使用它，因为 validator 要求 MP4、provenance、逐文件 SHA-256 和完整解码结果同时存在。

`--condition-targets` 是 opt-in 行为。它把目标基准改为经过有界 Savitzky-Golay SE(3) conditioning 的轨迹。当前发布基准没有启用它，最终 summary 的 `source.target_basis` 必须是 `registered_resampled_calibrated_tcp`。

> **Caution:** CLI 的 `--left-tool-offset-wxyz` 与 `--right-tool-offset-wxyz` 只替换任务级旋转对。配置中的左右 local translation 和 `WristAdaptationSpec` 仍继续生效。

### 用稳定任务身份装载双手轨迹

同一个动作名称可以出现在不同采集日期，因此 `TaskFamily` 用 `<date>/<task>` 作为稳定身份。`FactoryBimanualTask` 承载源 CSV 的行号、时间、左右位置、左右四元数、有效性和可选夹爪角度。

```python
from factory_bimanual.task_family import TaskFamily

fold = TaskFamily.parse("8-11/Fold_Box")
seal = TaskFamily(date="8-11", task="Seal_Bag")

assert fold.key == "8-11/Fold_Box"
assert seal.key == "8-11/Seal_Bag"
```

源轨迹由运行器加载为 `FactoryBimanualTask`。它拒绝空 CSV、非递增时间、非 `vr_world` 坐标域、零范数四元数以及超出允许值的平移跳变。左右手的源数据不会因后续 TCP 标定而被覆盖，最终 NPZ 另外保存 `raw_left_hand_*` 和 `raw_right_hand_*` 数组。

<details>
<summary>Reference: TaskFamily 与 FactoryBimanualTask</summary>

`TaskFamily` 位于 `factory_bimanual.task_family`：

- `TaskFamily(date, task)` 验证两个值都是单一路径组件。
- `TaskFamily.parse("8-11/Fold_Box")` 解析稳定任务键。
- `TaskFamily.key` 返回 `8-11/Fold_Box`。

`FactoryBimanualTask` 位于 `factory_bimanual.source_data`，核心字段包括：

- `source_path`、`source_row_index`、`time_s`、`coordinate_frame`
- `left_position_m`、`left_quaternion_wxyz`、`left_valid`
- `right_position_m`、`right_quaternion_wxyz`、`right_valid`
- `left_gripper_angle_rad`、`right_gripper_angle_rad`

</details>

### 注册到同一个世界坐标系

当源双手轨迹处于 VR 世界坐标时，`RigidTaskRegistration` 定义一个共享刚体变换，`register_task` 对左右位置和姿态应用完全相同的注册。共享注册保留双手之间的几何关系，避免为左右手分别拟合坐标变换。

```python
import numpy as np
from factory_bimanual.registration import RigidTaskRegistration

registration = RigidTaskRegistration(
    rotation_world_from_vr=np.eye(3),
    translation_world_m=np.array([0.0, 0.0, 0.75]),
)
assert registration.matrix.shape == (4, 4)
```

真实运行中，`register_task(task, registration)` 返回带注册证据的任务。`RigidTaskRegistration.inverse()` 可用于检查往返变换，`matrix` 提供可落盘的 4x4 表示。

`resample_task_60hz` 随后把注册轨迹重建到精确 60 Hz 时间轴。位置逐轴插值，四元数用 SLERP，端点保留，有效性按对应源行传播，夹爪通道按时间插值。

```python
from scripts.run_piperx_recommended_v31 import resample_task_60hz

def resample_registered_task(registered_task):
    return resample_task_60hz(registered_task, rate_hz=60.0)
```

上例中的 `registered_task` 代表 `register_task` 的返回值。完整运行器还会保留源行索引，使每个重采样目标都能追溯到原始 CSV。

### 解析推荐 mount 与求解策略

当任务身份和注册确定后，`load_recommended_config` 加载 `configs/piperx_recommended_v31.json`，返回 `PiperXRecommendedConfig`。配置同时封装严格容差、DLS、执行动力学、mount funnel、全局重启预算和每任务 mount 规范。

```python
from factory_bimanual.piperx_recommended import load_recommended_config
from factory_bimanual.task_family import TaskFamily

config = load_recommended_config()
family = TaskFamily.parse("8-11/Fold_Box")
spec = config.mounts[family.key]

assert config.accept.position_tolerance_m == 0.001
assert round(config.accept.orientation_tolerance_rad, 8) == 0.00872665
assert spec.source_take == "161044"
```

`PiperXRecommendedConfig` 的接受门是 1 mm / 0.5°，全局 anchor restart 为 40。最终两任务的 DLS warm-start 上限为 200 次迭代，执行上限为 1 rad/s 和 4 rad/s²。

`RecommendedMountSpec` 保存源或注册世界中的左右基座位置、安装形态、yaw、任务级旋转、local translation、腕部适配和选择证据。`world_mount_for_family` 把它解析为 `WorldMount`，并在需要时通过任务注册把 source-frame mount 变换到 registered world。

```python
from factory_bimanual.piperx_recommended import world_mount_for_family

world_mount = world_mount_for_family(
    config,
    family,
    registration.rotation_world_from_vr,
    registration.translation_world_m,
)

assert world_mount.family == family
assert world_mount.left_xyz_m.shape == (3,)
assert world_mount.right_xyz_m.shape == (3,)
```

最终 Fold_Box mount 为：

- 左基座 `[-0.24857021833998647, 0.429238866791272, 0.7570900119999999]` m
- 右基座 `[-0.34857021833998647, -0.17076113320872802, 0.7570900119999999]` m
- 共同高度 `0.7570900119999999` m，左右 yaw 都为 `0°`
- 模式 `upright_table`，基座间距 `0.6082762530298219` m

最终 Seal_Bag mount 为：

- 左基座 `[-0.35, 0.25, 0.81]` m
- 右基座 `[-0.30, -0.45, 0.81]` m
- 左右 yaw 都为 `15°`
- 模式 `upright_table`，基座间距 `0.70178344238091` m

<details>
<summary>Reference: 推荐配置对象</summary>

`PiperXRecommendedConfig` 位于 `factory_bimanual.piperx_recommended`，聚合：

- `accept`：位置、姿态和 branch guard
- `dls`：阻尼、步长、截断和迭代预算
- `execution`：速度、加速度、settle、decision 和源采样率
- `funnel`：工作空间与 mount 搜索预算
- `anchor_restarts` 和 `mounts`

`RecommendedMountSpec` 表示版本化的任务策略，`WorldMount` 表示进入场景构建器的已解析世界坐标 mount。`WorldMount.as_scene_mount()` 输出 family、形态、模式、XY、yaw、Z、基座距离、take、坐标域和选择方法。

</details>

## Mapping hands to calibrated TCP targets

当原始手坐标不能直接代表 PiperX 末端工具中心时，要先明确跟踪参考系。当前结果跟踪的是 `registered_resampled_calibrated_tcp`，同时把注册后的 raw hand 位姿保存在 NPZ 中，避免在报告中把两个参考系混为一谈。

### 固定旋转映射

`apply_fixed_tool_rotation` 把一个归一化 source-hand-to-TCP 旋转右乘到整条源手姿态。这个旋转对每个任务、每一侧保持固定，不逐帧拟合。

```python
from factory_bimanual.tool_frame_calibration import apply_fixed_tool_rotation

def map_fixed_tool_rotations(registered_task, spec):
    left_tcp_quaternion = apply_fixed_tool_rotation(
        registered_task.left_quaternion_wxyz,
        spec.left_tool_offset_quaternion_wxyz,
    )
    right_tcp_quaternion = apply_fixed_tool_rotation(
        registered_task.right_quaternion_wxyz,
        spec.right_tool_offset_quaternion_wxyz,
    )
    return left_tcp_quaternion, right_tcp_quaternion
```

Seal_Bag 的左右旋转在全部 1757 个注册 60 Hz 目标上保持固定。Fold_Box 也先应用固定旋转，再对右腕开头一秒添加有界局部轴适配。

### 固定 local translation 映射

`apply_fixed_tool_translation` 接收一个 source-hand local frame 中的固定向量。每个采样点先用对应的源手姿态旋转这个向量，再加到世界位置，所以它不是固定世界位移。

```python
from factory_bimanual.tool_frame_calibration import apply_fixed_tool_translation

def map_fixed_tool_translations(registered_task, spec):
    left_tcp_position = apply_fixed_tool_translation(
        registered_task.left_position_m,
        registered_task.left_quaternion_wxyz,
        spec.left_tool_translation_m,
    )
    right_tcp_position = apply_fixed_tool_translation(
        registered_task.right_position_m,
        registered_task.right_quaternion_wxyz,
        spec.right_tool_translation_m,
    )
    return left_tcp_position, right_tcp_position
```

> **Note:** `translation_m` 在每个源手局部坐标系中恒定。手姿态改变时，世界坐标位移随之旋转。

Fold_Box 的 fixed local translations 是：

- 左 `[-0.01399595, 0.00244380, -0.02057040]` m
- 右 `[0.00861553, 0.01174536, 0.02031795]` m

Seal_Bag 的 fixed local translations 是：

- 左 `[-0.00347928, 0.00355632, 0.00867452]` m
- 右 `[0.00188397, -0.00019278, -0.00981904]` m

Validator 会从 NPZ 读取 raw hand position、raw hand quaternion 和 summary 中的 translation，重新调用 `apply_fixed_tool_translation`，再逐元素检查得到的 calibrated TCP position 是否与发布目标一致。

### 开头有界翻腕回正

当固定工具映射仍让 Fold_Box 开头进入高风险腕部分支时，`factory_bimanual.piperx_recommended.WristAdaptationSpec` 声明一段可审计的 local-axis 角度调度。`apply_bounded_wrist_adaptation` 执行该调度，并同时返回逐帧角度数组。

```python
from factory_bimanual.piperx_recommended import WristAdaptationSpec
from factory_bimanual.tool_frame_calibration import (
    apply_bounded_wrist_adaptation,
    apply_fixed_tool_rotation,
)

def map_fold_box_wrist(registered_task, spec):
    right_tcp_quaternion = apply_fixed_tool_rotation(
        registered_task.right_quaternion_wxyz,
        spec.right_tool_offset_quaternion_wxyz,
    )
    wrist = WristAdaptationSpec(
        side="right",
        axis="x",
        angle_deg=-12.5,
        hold_until_s=22 / 60,
        return_until_s=1.0,
    )
    mapped, angle_deg = apply_bounded_wrist_adaptation(
        right_tcp_quaternion,
        registered_task.time_s,
        wrist,
    )
    assert max(abs(float(value)) for value in angle_deg) <= 12.5
    return mapped, angle_deg
```

Fold_Box 在 `t <= 22/60 s` 保持右腕 local x `-12.5°`，随后线性回到 `0°`，在 `t = 1.0 s` 后不再添加适配。实现硬限制为 `15°`，最终配置使用 `12.5°`。

> **Caution:** Fold_Box 的 0.5° 跟踪误差是相对于经过 fixed SE(3) 和开头腕部调度后的 calibrated TCP target，不是相对于开头一秒的 raw hand orientation。

Seal_Bag 的 `wrist_adaptation` 为 `None`，整条轨迹只使用固定 SE(3) 映射。

### 在一个组合点构造跟踪目标

`condition_complete_follow_targets` 是工具映射的组合点。它按固定顺序应用 local translation、固定旋转、可选 `WristAdaptationSpec`，最后才处理可选 bounded Savitzky-Golay conditioning。

```python
from scripts.run_piperx_recommended_v31 import condition_complete_follow_targets

def build_calibrated_targets(task_60hz, spec):
    tool_offsets = {
        "left": spec.left_tool_offset_quaternion_wxyz,
        "right": spec.right_tool_offset_quaternion_wxyz,
    }
    tool_translations = {
        "left": spec.left_tool_translation_m,
        "right": spec.right_tool_translation_m,
    }
    target_task, target_quaternions, conditioning_audit = (
        condition_complete_follow_targets(
            task_60hz,
            tool_offsets=tool_offsets,
            tool_translations=tool_translations,
            wrist_adaptation=spec.wrist_adaptation,
            apply_conditioning=False,
        )
    )
    assert conditioning_audit.window == 0
    return target_task, target_quaternions, conditioning_audit
```

发布结果要求 `apply_conditioning=False`。如果显式启用 `--condition-targets`，position deviation 上限为 5 mm，orientation deviation 上限为 1°，但这种 target basis 不通过当前两任务发布 validator。

### 同时记录 raw hand 与 calibrated TCP

当求解完成后，`build_complete_summary` 把目标基准、raw hand 保留标记、工具旋转、local translation、腕部调度、接受门、候选预算、重定时和动力学证据写入 summary。轨迹 NPZ 则保存 raw hand、calibrated target、actual TCP、误差、qpos、碰撞和逐帧调度。

```python
import json
from pathlib import Path

summary_path = Path(
    "reports/piperx_two_task_complete_follow/fold_box/"
    "8-11_Fold_Box_161044_complete_follow.summary.json"
)
summary = json.loads(summary_path.read_text(encoding="utf-8"))

assert summary["schema"] == "piperx-complete-follow-v2"
assert summary["source"]["target_basis"] == (
    "registered_resampled_calibrated_tcp"
)
assert summary["source"]["raw_hand_trace_preserved"] is True
assert summary["metrics"]["complete_source_pose_coverage"] == 1.0
assert summary["metrics"]["collision_frames"] == 0
assert summary["metrics"]["execution_collision_frames"] == 0
```

在最终 summary 中，`source.raw_hand_trace_preserved` 必须为 `true`，`source.tracking_reference` 必须是 `task-level calibrated PiperX TCP`，`tool_frame.tracking_error_reference` 必须是 `calibrated TCP target`。

<details>
<summary>Reference: 工具映射 API</summary>

- `apply_fixed_tool_rotation(source_quaternions_wxyz, offset_quaternion_wxyz)`：归一化输入并应用固定旋转。
- `apply_fixed_tool_translation(source_positions_m, source_quaternions_wxyz, translation_m)`：把固定 local translation 逐帧旋转到世界坐标。
- `factory_bimanual.piperx_recommended.WristAdaptationSpec`：声明 side、axis、angle、hold 和 return 时间。
- `apply_bounded_wrist_adaptation(source_quaternions_wxyz, time_s, spec)`：返回映射四元数和精确角度调度。
- `condition_complete_follow_targets(...)`：组合 fixed SE(3)、可选腕部适配和可选 conditioning。
- `build_complete_summary(...)`：把 raw 与 calibrated 表示、协议和结果写入一个可验证 summary。

</details>

## Finding a collision-free complete path

当目标轨迹准备好后，求解必须把单臂 IK、双臂配对、状态碰撞、扫掠边碰撞、分支连续性和执行动力学分成不同门。严格 IK 覆盖为 100% 并不自动说明原时间戳可执行，也不自动说明路径无碰撞。

### 生成局部与全局严格 IK 候选

`CandidateGeneratorConfig` 定义位置与姿态容差、DLS 参数、去重距离、全局 seed、候选上限、受约束 fallback、分层 seed 和腕部风险排序。`MuJoCoCandidateGenerator` 在真实 PiperX 模型和原生 6 DOF 关节限位上生成候选。

```python
from factory_bimanual.mujoco_candidate_generator import (
    CandidateGeneratorConfig,
    MuJoCoCandidateGenerator,
)

def make_candidate_generator(
    model, data, contract, name_map, target_task, config
):
    if len(target_task.time_s) < 1:
        raise ValueError("target task must contain at least one frame")
    candidate_config = CandidateGeneratorConfig(
        position_tolerance_m=config.accept.position_tolerance_m,
        orientation_tolerance_rad=config.accept.orientation_tolerance_rad,
        damping=config.dls.damping,
        step_scale=config.dls.step_scale,
        maximum_step_rad=config.dls.maximum_step_rad,
        position_error_clip_m=config.dls.position_error_clip_m,
        orientation_error_clip_rad=config.dls.orientation_error_clip_rad,
        max_iterations=config.dls.max_iterations,
        global_seed_count=config.anchor_restarts,
        maximum_candidates=16,
        constrained_fallback_enabled=True,
        constrained_fallback_seed_count=20,
        constrained_fallback_max_iterations=300,
        stratified_seed_enabled=True,
        wrist_risk_enabled=True,
        dedup_rad=0.004363323129985824,
        rolling_early_stop_candidates=4,
    )
    return MuJoCoCandidateGenerator(
        model,
        data,
        contract,
        name_map=name_map,
        config=candidate_config,
    )
```

正常连续帧先尝试 local warm start。warm start 无严格解，或双臂配对后无法形成无碰撞连接时，runner 扩展 stratified global candidates。候选评分还使用 pose cost、joint-limit margin 和 PiperX J4/J5 wrist risk。

### 把状态与扫掠边都设为硬门

`MuJoCoPairedCollisionChecker` 对成对左右关节状态调用 MuJoCo 接触审计，并沿相邻结点之间的插值边重复检查。`ClearanceReport` 另外记录命名几何对的最小间距、要求 margin 和限制对。

```python
def audit_pair(
    checker,
    left_q,
    right_q,
    previous_left_q,
    previous_right_q,
):
    state_report = checker.state(left_q, right_q)
    edge_report = checker.transition(
        (previous_left_q, previous_right_q),
        (left_q, right_q),
    )
    return state_report, edge_report
```

`checker.clearance(left_q, right_q, margin_m=0.015)` 返回 `ClearanceReport`，可用于额外的 15 mm 命名 cross-arm gap 审计。当前发布零碰撞门检查几何接触，安全距离膨胀需要在真机前重新配置和验证。

> **Caution:** 仅检查离散 IK 结点会漏掉相邻姿态之间的碰撞。发布路径同时要求 `state(...)` 和 `transition(...)` 有效。

### 在完整轨迹上协调两侧状态机

`CompleteFollowRunner` 为左右侧维护独立 warm-start 状态机，在配对阶段统一处理碰撞和分支连续性。它先组合左右候选，再硬拒绝碰撞状态和碰撞入边，然后按插入步数、最大关节增量、腕部风险、pose cost 和 joint-limit margin 排序。

```python
from factory_bimanual.complete_follow import CompleteFollowRunner

def solve_complete_path(
    model,
    target_task,
    mapped_quaternions,
    config,
    maximum_candidates_per_side=16,
):
    runner = CompleteFollowRunner(
        model,
        target_task,
        mapped_quaternions,
        config,
        maximum_candidates_per_side=maximum_candidates_per_side,
    )
    return runner.run()
```

如果 local 和 global candidates 都耗尽，`CompleteFollowInfeasibleError` 会报告精确 source row。调用方可以捕获这个异常并读取 `error.row`；runner 不会把最后一个碰撞解当作 fallback。

```python
from factory_bimanual.complete_follow import CompleteFollowInfeasibleError

def run_with_infeasible_row(run_complete_follow):
    try:
        return run_complete_follow()
    except CompleteFollowInfeasibleError as error:
        print(f"infeasible source row: {error.row}")
        raise
```

> **Unlike** 只追求逐帧 IK 覆盖的流程，`CompleteFollowRunner` 要求每个被选状态都能从前一状态通过无碰撞扫掠边连接。

### 无损重定时与动力学边界

当 joint-space 路径可达且无碰撞，但原始时间戳超过 1 rad/s 或 4 rad/s² 时，`retime_complete_source_path` 保留全部 source target 和顺序，只插入必要的 branch transition states，并局部拉伸违反动力学上限的区段。

```python
from factory_bimanual.complete_follow import retime_complete_source_path

def retime_pair_path(
    source_pair_q,
    source_time_s,
    periodic_mask,
    state_valid,
    transition_valid,
):
    retiming = retime_complete_source_path(
        source_pair_q,
        source_time_s,
        periodic=periodic_mask,
        branch_guard_rad=0.3,
        maximum_velocity_rad_s=1.0,
        maximum_acceleration_rad_s2=4.0,
        state_valid=state_valid,
        transition_valid=transition_valid,
    )
    source_order = retiming.execution_source_index[
        retiming.source_execution_index
    ].tolist()
    assert source_order == list(range(len(source_pair_q)))
    assert retiming.maximum_velocity_rad_s <= 1.0 + 1e-9
    assert retiming.maximum_acceleration_rad_s2 <= 4.0 + 1e-9
    return retiming
```

`CompleteRetiming` 保存 execution q、execution timestamps、source index 映射、execution state、source-to-execution 索引、fixed-time 接受、插入帧数、delay、time scale、最大速度和最大加速度。

速度和加速度审计包含零速度起步与零速度停止边界。因此最终两任务虽然 calibrated TCP 覆盖都是 100%，但原时间戳固定时序只有首帧通过：Fold_Box 为 `1/1061`，Seal_Bag 为 `1/1757`。

最终重定时结果：

| 任务 | 执行时长 | 相对源时间延迟 | vmax | amax | 动力学门 |
|---|---:|---:|---:|---:|---:|
| Fold_Box | 73.950623048 s | 56.293667506 s | 1.0000000000000475 rad/s | 4.000000000001643 rad/s² | 通过浮点容差 |
| Seal_Bag | 100.711602008 s | 71.450240883 s | 1.0000000000001532 rad/s | 4.000000000000061 rad/s² | 通过浮点容差 |

> **Caution:** `retime_complete_source_path` 在时间拉伸前就验证源状态和源边。无效状态或边会立即失败，不能靠放慢速度隐藏碰撞。

### 读取共享结果对象

`CompleteFollowResult` 是渲染、summary 和 validator 共用的证据对象。它把 source 与 execution 两条时间轴、qpos、实际 TCP、误差、碰撞、候选数、global rescue 和动力学指标放在同一个不可含糊的数据结构中。

```python
import numpy as np

def assert_strict_result(result, expected_source_frames):
    assert result.source_qpos.shape[0] == expected_source_frames
    assert result.execution_qpos.shape[0] == len(result.execution_time_s)
    assert result.position_error_m["left"].max() <= 0.001
    assert result.orientation_error_rad["right"].max() <= np.deg2rad(0.5)
```

最终最大误差：

| 任务 | L position | R position | L orientation | R orientation |
|---|---:|---:|---:|---:|
| Fold_Box | 0.999604 mm | 0.999865 mm | 0.492460° | 0.495387° |
| Seal_Bag | 0.999965 mm | 0.999980 mm | 0.499972° | 0.494466° |

<details>
<summary>Reference: 完整路径求解 API</summary>

- `CandidateGeneratorConfig`：严格 IK 生成策略。
- `MuJoCoCandidateGenerator`：local warm start、stratified global seeds、fallback 和风险排序。
- `MuJoCoPairedCollisionChecker`：state、side state、transition 和命名 clearance 审计。
- `ClearanceReport`：最小间距、逐对距离、margin、valid 和 limiting pair。
- `CompleteFollowRunner`：逐帧候选、成对选择、global rescue、FK 测量、碰撞与重定时协调。
- `CompleteFollowInfeasibleError`：严格且无碰撞的 connected pair 耗尽时的显式失败。
- `retime_complete_source_path`：保留 source targets 的无损重定时。
- `CompleteRetiming`：仅表示重定时证据。
- `CompleteFollowResult`：表示完整求解、误差、碰撞和动力学证据。

</details>

## Validating and presenting evidence

当求解结果准备好后，发布对象不是单个 coverage 数字，而是一条可以独立复算的证据链：scene、trajectory、summary、MP4、provenance、task validator、two-task manifest、CSV 和 PDF。

### 渲染执行时间轴

`VideoRenderConfig` 定义分辨率、fps、相机、轨迹样式、标题和是否插值 execution states。`render_mujoco_mp4` 使用 MuJoCo official PiperX meshes 渲染，`write_video_provenance` 为每个编码帧记录 execution knot 或 active incoming edge，`decode_check_mp4` 则重新打开容器并检查帧数、fps、时长和分辨率。

```python
from pathlib import Path
from factory_bimanual.video import decode_check_mp4

published_videos = (
    (
        Path(
            "reports/piperx_two_task_complete_follow/fold_box/"
            "8-11_Fold_Box_161044_complete_follow.mp4"
        ),
        2220,
        73.966667,
    ),
    (
        Path(
            "reports/piperx_two_task_complete_follow/seal_bag/"
            "8-11_Seal_Bag_161504_complete_follow.mp4"
        ),
        3022,
        100.700000,
    ),
)

for video_path, expected_frames, expected_duration_s in published_videos:
    check = decode_check_mp4(
        video_path,
        expected_frames=expected_frames,
        expected_resolution=(1280, 720),
        expected_duration_s=expected_duration_s,
    )
    assert check.frame_count == expected_frames
    assert check.fps == 30.0
```

`decode_check_mp4` 对 expected duration 使用一帧时长的容差。正式生成仍由 `render_mujoco_mp4` 使用 execution timeline、线性插值和官方 PiperX meshes；`write_video_provenance` 把最终端点、execution knot 与 active incoming edge 的审计范围逐帧绑定。`render_mujoco_mp4` 内部也会执行解码检查，发布前再运行上面的只读检查可独立确认容器属性。

最终视频属性：

- Fold_Box：1280x720，30 fps，2220 帧，解码时长 73.966667 s
- Seal_Bag：1280x720，30 fps，3022 帧，解码时长 100.700000 s

> **Note:** 视频使用 execution timeline，并在相邻 retimed execution knots 之间线性插值。provenance 对移动中的编码帧引用其 active incoming edge 的碰撞审计。

### 重算一个任务 bundle

`validate_task_bundle` 不相信 summary 中的 headline。它重新读取 NPZ，检查 family、take、帧数、target basis、raw hand evidence、固定 translation、腕部调度、误差上限、动态门、source 与 execution 碰撞、provenance 和视频解码。

```python
from scripts.validate_piperx_two_task_bundle import validate_task_bundle

fold_record = validate_task_bundle(
    "reports/piperx_two_task_complete_follow/fold_box/"
    "8-11_Fold_Box_161044_complete_follow.summary.json",
    family="8-11/Fold_Box",
    take="161044",
    frames=1061,
    decode_video=True,
)

assert fold_record["pose_frames"] == 1061
assert fold_record["collision_frames"] == 0
```

Summary 中每个 artifact path 按 basename 在 summary 同目录解析，然后同时核对 SHA-256 与 byte size。仅复制 summary，或依赖 summary 中某个外部绝对路径，都不能通过验证。

### 把两项任务合成发布 manifest

`validate_bundle` 要求 bundle 根目录下恰好有 Fold_Box 与 Seal_Bag 两个任务。`write_outputs` 把验证记录转换为便携相对路径，并生成 `two_task_manifest.json` 与 `two_task_summary.csv`。

```python
from pathlib import Path
from scripts.validate_piperx_two_task_bundle import validate_bundle, write_outputs

bundle_root = Path("reports/piperx_two_task_complete_follow")
records = validate_bundle(bundle_root, decode_video=True)
manifest_path, csv_path = write_outputs(bundle_root, records)

assert manifest_path.name == "two_task_manifest.json"
assert csv_path.name == "two_task_summary.csv"
```

发布命令是：

```powershell
python -m scripts.validate_piperx_two_task_bundle `
  --bundle-root reports/piperx_two_task_complete_follow
```

`--skip-video-decode` 只跳过新一轮 MP4 全量解码，仍会交叉检查已记录的 decode metadata 和 provenance。正式发布验证不使用该选项。

### 生成证据门控 PDF

`load_report_data` 只接受 schema 正确、任务集合恰好为 Fold_Box 与 Seal_Bag、1 mm / 0.5°、calibrated TCP coverage 100% 且发布碰撞为零的 manifest。`build_report` 在这个门后生成中文 PDF，包含覆盖率、误差、时长、迭代、哈希、限制和真机前置条件。

```python
from scripts.build_piperx_two_task_report import load_report_data, build_report

records = load_report_data(
    "reports/piperx_two_task_complete_follow/two_task_manifest.json"
)
assert set(records) == {"Fold_Box", "Seal_Bag"}

build_report(
    "reports/piperx_two_task_complete_follow/two_task_manifest.json",
    "reports/piperx_two_task_complete_follow/two_task_experiment_log.json",
    "reports/piperx_two_task_complete_follow/"
    "PiperX双任务严格完全跟随实验报告.pdf",
)
```

该 PDF 是本地交付物，不属于 GitHub 发布集合。视频、manifest、CSV、summary、trajectory、scene、provenance 和 QA frames 仍可按 `.gitignore` allowlist 发布。

<details>
<summary>Reference: 验证与报告 API</summary>

- `VideoRenderConfig`：MP4 渲染参数。
- `render_mujoco_mp4`：执行时间轴 MuJoCo 渲染。
- `write_video_provenance`：frame-level execution 与碰撞审计 sidecar。
- `decode_check_mp4`：独立 MP4 decode audit。
- `validate_task_bundle`：单任务完整复算。
- `validate_bundle`：精确双任务集合门。
- `write_outputs`：便携 manifest 与 CSV 输出。
- `load_report_data`：PDF 前置证据门。
- `build_report`：生成最终中文报告。

</details>

## Iterating on mounts without losing provenance

当某个 mount 在特定窗口失败时，不直接改 JSON 并覆盖旧结果。`MountCandidate`、确定性邻域、probe、append-only log 和 resume 共同把失败方案保留为可复查实验历史。

### 生成确定性 local mount 邻域

`MountCandidate` 保存左右 XYZ、左右 yaw、模式、名称和 origin。它的 `key` 是去掉显示名称后的规范 JSON 哈希，保证相同几何方案得到相同标识。

`local_mount_candidates` 围绕 incumbent 依次生成左右 X/Y、共享 Z 和左右 yaw 的单坐标邻域，并按 candidate key 稳定排序。

```python
from scripts.optimize_piperx_two_task_follow import (
    MountCandidate,
    local_mount_candidates,
)

incumbent = MountCandidate(
    name="seal_upright",
    mode="upright_table",
    left_xyz_m=(-0.35, 0.25, 0.81),
    right_xyz_m=(-0.30, -0.45, 0.81),
    left_yaw_deg=15.0,
    right_yaw_deg=15.0,
    origin="validated_incumbent",
)

candidates = local_mount_candidates(
    incumbent,
    xy_step_m=0.01,
    z_step_m=0.01,
    yaw_step_deg=5.0,
    round_index=1,
)
```

共享 Z 邻域始终同时调整左右基座高度，避免产生不符合 `WorldMount` 共同高度约束的方案。

### 用同一 target context 执行 probe

`probe_candidate` 把 candidate 转换为 `WorldMount`，构建真实双 PiperX scene，并在同一 registered calibrated target context 下检查指定左右 source rows 的 global strict candidates。

```python
from scripts.optimize_piperx_two_task_follow import probe_candidate

def run_mount_probe(
    candidate,
    *,
    world_mount,
    target_task,
    target_quaternions,
    config,
):
    probe = probe_candidate(
        candidate,
        base_mount=world_mount,
        task=target_task,
        mapped=target_quaternions,
        config=config,
        output_dir=".tmp/piperx_two_task_mount_search",
        rows_by_side={"left": [0, 530, 1060], "right": [0, 530, 1060]},
        maximum_candidates=16,
    )
    assert probe["status"] == "probe_complete"
    return probe
```

Probe 记录 candidate key、输入参数、逐侧逐帧候选数、strict hit、orientation evidence 和 scene path。它是筛选门，不替代完整 1061 或 1757 帧求解。

### 追加记录并安全恢复

`append_experiment_record` 先读取既有 schema，再把一条记录追加到 `records`，写入临时文件后原子替换。`pending_candidates` 根据 candidate key、完成状态和可选 probe signature 过滤已完成方案。

```python
from pathlib import Path
from scripts.optimize_piperx_two_task_follow import (
    append_experiment_record,
    pending_candidates,
)

def record_mount_probe(log_path, probe, candidates):
    log_path = Path(log_path)
    append_experiment_record(log_path, probe)
    remaining = pending_candidates(
        candidates,
        log_path,
        resume=True,
        probe_signature=probe["probe_signature"],
    )
    assert candidates[0].key not in {item.key for item in remaining}
    return remaining
```

这条工作流让失败窗口成为实验记录，而不是消失在新的配置值后面。完整结果仍必须回到 `CompleteFollowRunner` 和 `validate_task_bundle` 的硬门。

<details>
<summary>Reference: mount 迭代 API</summary>

- `MountCandidate`：可哈希的 mount 方案。
- `local_mount_candidates`：确定性 X/Y、共享 Z、yaw 邻域。
- `probe_candidate`：指定 source rows 的严格候选 probe。
- `append_experiment_record`：append-only 原子日志。
- `pending_candidates`：按 candidate key 与 probe signature 恢复未完成工作。

</details>

## Qualifying native robot models

当单臂比较扩展到不同厂商时，不能只把关节链缩放成近似形态。每个结果必须关联到 vendor-native 模型、真实 mesh/package roots、TCP 约定、关节限位权威和来源修订。

### 用 ModelEntry 声明 13 个运行时模型

`ModelEntry` 描述模型路径、active joints、TCP parent、工具偏移、预期 DOF、mesh/package roots、base link、TCP authority 和 joint-limit authority。`MODELS` 是 13 个运行时条目的唯一注册表。

```python
from scripts.strict_urdf_model_audit import MODELS, ModelEntry

piperx = MODELS["piperx"]

assert isinstance(piperx, ModelEntry)
assert piperx.path.as_posix().endswith(
    "third_party/official_robot_models/piperx/PiperX.urdf"
)
assert piperx.joints == tuple(f"joint{i}" for i in range(1, 7))
assert piperx.tcp_parent == "ee_frame"
assert piperx.tool_offset_m == 0.0
assert piperx.expected_dof == 6
assert piperx.tcp_authority == "model_ee_frame_115mm"
```

当前 13 个 runtime variants 是：

1. `doosan`
2. `xarm6`
3. `ur5`
4. `kinova_gen3_lite`
5. `arx_x5`
6. `big_yam`
7. `franka_panda`
8. `franka_panda_locked_j3`
9. `i2rt_yam`
10. `nero`
11. `openarm`
12. `piperx`
13. `willow`

PiperX 的精确 runtime 坐标是：

- 模型：`third_party/official_robot_models/piperx/PiperX.urdf`
- active joints：`joint1` 到 `joint6`
- TCP parent：`ee_frame`
- tool translation：`[0.0, 0.0, 0.0]` m
- mesh root：`third_party/official_robot_models/piperx/meshes`
- TCP authority：`model_ee_frame_115mm`

### 加载并资格审计 native model

`load_native_spec` 读取 URDF 或 MJCF，保留 vendor visual layer，解析 package meshes，必要时把 DAE 确定性转换为可供 MuJoCo 使用的 STL，并应用已声明的 locked joint ranges。它不进行 morphology scaling。

`qualify` 在模型上附加 audited tracking TCP，编译 MuJoCo spec，验证根 body、mesh、active joints、DOF、关节限位、TCP transform、随机 FK、workspace span 和有限变换。

```python
from scripts.strict_urdf_model_audit import load_native_spec, qualify

spec = load_native_spec(MODELS["piperx"])
model = spec.compile()
audit = qualify("piperx", MODELS["piperx"])

assert audit["status"] == "pass"
assert audit["active_dof"] == 6
assert audit["native_geometry_dimensions"] is True
```

> **Caution:** 发布所有 runtime model assets 时必须包含 URDF/MJCF、mesh 和 package roots。只上传顶层 URDF 会让 `load_native_spec` 在另一台机器上因缺少资源而失败。

### 关联上游来源与 snapshot 状态

`OFFICIAL_MODELS` 记录公开上游仓库、固定 revision 和 variant。`MODEL_SOURCES` 在此基础上补入 Big YAM，以及没有公开 revision 的 Nero 与 Willow workspace-vendored snapshot metadata。

```python
from scripts.official_model_manifest import OFFICIAL_MODELS, MODEL_SOURCES

assert OFFICIAL_MODELS["piperx"]["revision"] == (
    "f6642ce0d7872c686f29c99e9e10cd23d1d49313"
)
assert MODEL_SOURCES["nero"]["repository"] == "workspace-vendored-snapshot"
assert set(MODELS) == set(MODEL_SOURCES)
```

`build_gate` 合并 runtime model SHA-256、来源、TCP authority、joint-limit authority、active joints 和 ranking eligibility。ARX X5 的 joint limit authority 标记为未验证 placeholder，因此 gate 会要求排名披露，但 vendor-directory execution 仍可独立判断。

```python
from scripts.build_official_model_provenance_gate import build_gate

gate = build_gate()

assert gate["experiment_execution_allowed"] is True
assert len(gate["robots"]) == 13
```

<details>
<summary>Reference: native model qualification API</summary>

- `ModelEntry`：运行时模型、关节、TCP、mesh/package 和权威字段。
- `MODELS`：13 个运行时 native variants。
- `load_native_spec`：不缩放几何的 vendor model importer。
- `qualify`：compile、mesh、joint、TCP 和 workspace audit。
- `OFFICIAL_MODELS`：公开上游仓库与 pinned revision。
- `MODEL_SOURCES`：完整 13 模型来源，包括 workspace-vendored snapshots。
- `build_gate`：机器可读的来源、TCP、joint-limit 和排名资格门。

</details>

## Publishing reproducible outputs

当实验在本地完成后，GitHub 内容由明确 allowlist 决定，不是 workspace 镜像。模型依赖要完整，双臂可复算证据要成套，单臂只选定报告和视频，四份指定 PDF 留在本地。

### 生成单臂求解与真实网格视频

`scripts.run_twelve_arm_two_single_tasks.run` 是批量子进程执行器，它把每个命令的 stdout 与 stderr 写入指定文件并返回退出码。模块的 `main` 组织模型审计、任务求解和渲染作业。

`scripts.render_strict_single_arm_task.main` 读取严格模型注册和求解缓存，使用真实 mesh 渲染单臂任务 MP4。两个入口共享 `MODELS` 权威，避免求解与视频使用不同几何。

```powershell
python -m scripts.run_twelve_arm_two_single_tasks
python -m scripts.render_strict_single_arm_task --help
```

在 Python 中调用 subprocess helper 的精确入口是 `scripts.run_twelve_arm_two_single_tasks.run`：

```python
from pathlib import Path
from scripts.run_twelve_arm_two_single_tasks import run

exit_code = run(
    ["python", "-m", "scripts.strict_urdf_model_audit"],
    Path("logs/model_audit.stdout.log"),
    Path("logs/model_audit.stderr.log"),
)
assert exit_code == 0
```

真实渲染 CLI 的精确入口是 `scripts.render_strict_single_arm_task.main`。命令行模块负责解析参数，直接从 Python 调用时传入它所声明的 CLI 参数后再进入 `main()`。

### 聚合报告、图表和对比视频

`scripts.build_ten_arm_two_single_task_outputs` 聚合选定 10 臂与两个单手任务，输出逐任务表、排名图、失败热图、PDF 和 10 臂对比视频。入口是该模块的 `main()`。

`scripts.build_twelve_arm_all_single_task_outputs` 聚合 12 臂与 12 个纯单手任务，输出 `all_results.csv`、summary、四类图、逐任务 12 臂对比视频、总体串联视频、HTML 和 PDF。入口同样是模块的 `main()`。

```powershell
python -m scripts.build_ten_arm_two_single_task_outputs
python -m scripts.build_twelve_arm_all_single_task_outputs
```

前者用于紧凑的两任务构型对比，后者用于完整单手任务矩阵。两者都消费已生成的严格结果，不能替代 `load_native_spec`、`qualify` 和 `build_gate` 的模型资格步骤。

### 用 .gitignore 定义发布 allowlist

`.gitignore` 当前允许整个 `third_party/official_robot_models/**`，并再次忽略任何嵌套 `.git` metadata。这样 13 个 runtime models 的模型文件、mesh、package roots 和确定性派生资源都能进入 GitHub，而嵌入式仓库元数据不会被发布。

关键规则是：

```gitignore
/third_party/*
!/third_party/official_robot_models/
!/third_party/official_robot_models/**
/third_party/**/.git/
```

双臂发布 allowlist 保留：

- [双任务 manifest](reports/piperx_two_task_complete_follow/two_task_manifest.json)
- [双任务汇总 CSV](reports/piperx_two_task_complete_follow/two_task_summary.csv)
- [检测与优化实验日志](reports/piperx_two_task_complete_follow/two_task_experiment_log.json)
- Fold_Box：[MP4](reports/piperx_two_task_complete_follow/fold_box/8-11_Fold_Box_161044_complete_follow.mp4)、[summary](reports/piperx_two_task_complete_follow/fold_box/8-11_Fold_Box_161044_complete_follow.summary.json)、[trajectory NPZ](reports/piperx_two_task_complete_follow/fold_box/8-11_Fold_Box_161044_complete_follow.trajectory.npz)、[scene JSON](reports/piperx_two_task_complete_follow/fold_box/8-11_Fold_Box_161044_complete_follow.scene.json)、[scene XML](reports/piperx_two_task_complete_follow/fold_box/8-11_Fold_Box_161044_complete_follow.scene.xml)、[provenance](reports/piperx_two_task_complete_follow/fold_box/8-11_Fold_Box_161044_complete_follow.provenance.json)、QA [start](reports/piperx_two_task_complete_follow/fold_box/qa_start.png) / [middle](reports/piperx_two_task_complete_follow/fold_box/qa_middle.png) / [end](reports/piperx_two_task_complete_follow/fold_box/qa_end.png)
- Seal_Bag：[MP4](reports/piperx_two_task_complete_follow/seal_bag/8-11_Seal_Bag_161504_complete_follow.mp4)、[summary](reports/piperx_two_task_complete_follow/seal_bag/8-11_Seal_Bag_161504_complete_follow.summary.json)、[trajectory NPZ](reports/piperx_two_task_complete_follow/seal_bag/8-11_Seal_Bag_161504_complete_follow.trajectory.npz)、[scene JSON](reports/piperx_two_task_complete_follow/seal_bag/8-11_Seal_Bag_161504_complete_follow.scene.json)、[scene XML](reports/piperx_two_task_complete_follow/seal_bag/8-11_Seal_Bag_161504_complete_follow.scene.xml)、[provenance](reports/piperx_two_task_complete_follow/seal_bag/8-11_Seal_Bag_161504_complete_follow.provenance.json)、QA [start](reports/piperx_two_task_complete_follow/seal_bag/qa_start.png) / [middle](reports/piperx_two_task_complete_follow/seal_bag/qa_middle.png) / [end](reports/piperx_two_task_complete_follow/seal_bag/qa_end.png)
- Authoritative source CSV：[Fold_Box 161044](data/factory/8-11/Fold_Box/handheld_20260811_161044.csv) 与 [Seal_Bag 161504](data/factory/8-11/Seal_Bag/handheld_20260811_161504.csv)

精选单臂报告保留：

- [mount IK fidelity pilot 报告](reports/single_arm/mount_ik_fidelity_pilot/pdf/mount_ik_fidelity_pilot_report.pdf)
- [ten-arm fixed-vs-legacy 报告](reports/single_arm/ten_arm_two_single_tasks_handbook_fixed_4096/ten_arm_fixed_vs_legacy_report.pdf)

精选单臂视频与清单保留：

- [13-arm Open_Box 30 s 视频](videos/single_arm/thirteen_arm_open_box_30s.mp4)
- [13-arm Open_Box 3D 30 s 视频](videos/single_arm/thirteen_arm_open_box_3d_30s.mp4)
- [单臂逐任务 manifest](videos/single_arm/per_arm_task_manifest.json)
- [严格视频作业 manifest](videos/single_arm/strict_video_job_manifest.json)

以下四份 PDF 明确保留在本地，不进入 GitHub：

1. `reports/piperx_two_task_complete_follow/PiperX双任务严格完全跟随实验报告.pdf`
2. `单轨迹优化分析.pdf`
3. `Seal_Bag 轨迹上 PiperX 双臂三种安装构型的跟随能力对比 - 飞书云文档.pdf`
4. `双臂IK跟随方案四臂四种安装位姿报告.pdf`

前三份根目录参考 PDF 由显式 ignore 规则排除，最终双任务报告 PDF 因未被 `reports` allowlist 反向包含而保持本地。已经被 Git 跟踪的文件仍需用 `git rm --cached` 从索引移除，ignore 规则本身不会自动取消跟踪。

> **Note:** [PiperX 双臂构型对比 · 综合分析报告（26 条轨迹）](<PiperX 双臂构型对比 · 综合分析报告（26 条轨迹） - 飞书云文档.pdf>) 不在上述四份排除清单中，并保留在 GitHub 发布集合。

### 用 .gitattributes 固定二进制语义

`.gitattributes` 把 MP4、NPZ、PDF 和 PNG 声明为 binary，避免换行规范化和文本 diff 破坏工件内容。

```gitattributes
*.mp4 binary
*.npz binary
*.pdf binary
*.png binary
```

这些规则不会执行压缩或 Git LFS 迁移。提交前仍要检查单文件大小、模型资产完整性、Git index 中是否含嵌套 `.git`，以及四份本地 PDF 是否未被跟踪。

<details>
<summary>Reference: 单臂输出与发布边界</summary>

- `scripts.run_twelve_arm_two_single_tasks.run`：带 stdout/stderr 文件的批量命令执行。
- `scripts.render_strict_single_arm_task.main`：strict registry 的真实 mesh MP4 CLI 入口。
- `scripts.build_ten_arm_two_single_task_outputs`：10 臂双任务报告、图和对比视频模块。
- `scripts.build_twelve_arm_all_single_task_outputs`：12 臂全单手矩阵报告、图、HTML 和视频模块。
- `.gitignore`：完整 runtime model assets 与精选证据 allowlist。
- `.gitattributes`：MP4、NPZ、PDF、PNG 的 binary 声明。

</details>

## Putting it together

完整发布流程先重跑两个任务，再执行无跳过的 validator，最后生成本地 PDF。运行顺序把求解、视频和报告建立在同一份 validated manifest 上。

```powershell
python -m scripts.run_piperx_recommended_v31 `
  --family 8-11/Fold_Box --source-take 161044 `
  --maximum-candidates 16 `
  --output-dir reports/piperx_two_task_complete_follow/fold_box

python -m scripts.run_piperx_recommended_v31 `
  --family 8-11/Seal_Bag --source-take 161504 `
  --maximum-candidates 16 `
  --output-dir reports/piperx_two_task_complete_follow/seal_bag

python -m scripts.validate_piperx_two_task_bundle `
  --bundle-root reports/piperx_two_task_complete_follow

python -m scripts.build_piperx_two_task_report
```

发布前核对最终事实：

- Fold_Box：`1061/1061` calibrated TCP targets，`73.950623 s` execution，`2220` video frames，`0` collision frames。
- Seal_Bag：`1757/1757` calibrated TCP targets，`100.711602 s` execution，`3022` video frames，`0` collision frames。
- 两任务接受门均为 `1 mm / 0.5°`。
- NPZ 同时含 raw hand evidence 与 calibrated TCP targets。
- Fold_Box 仅在开头一秒使用最大 `12.5°` 的右腕有界回正，Seal_Bag 无 wrist adaptation。
- `third_party/official_robot_models` 的 13 模型完整资产进入 GitHub。
- 精选单臂报告、视频和 manifest 进入 GitHub。
- 四份指定 PDF 只保留在本地。

如果任何一项任务出现未到达帧、状态碰撞、扫掠入边碰撞、动态上限失败、哈希不一致、raw/calibrated 映射不一致或视频解码不一致，`validate_piperx_two_task_bundle` 会拒绝生成新的可发布 manifest。
