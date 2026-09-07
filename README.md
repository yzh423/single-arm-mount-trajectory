# single-arm-mount

This repository turns recorded hand trajectories into robot mount comparisons and auditable MuJoCo motion evidence. A typical first action is a Python module command that selects a recording and mounting mode, then produces a scene, a trajectory archive, and metrics. The current PiperX controller-event v4 workflow reconstructs synchronized controller updates from repeated host observations, preserves those event intervals, and follows recorded targets after the configured fixed tool transform. Mount choice, pose accuracy, collision and topology checks, and joint dynamics are separate questions recorded as separate evidence.

The repository also contains single-arm design optimization, native robot model audits, and earlier factory bimanual experiments. Results are meaningful together with their recording, target mapping, time protocol, and evaluated window: full recordings, short prefixes, conditioned targets, and retimed executions are different experiments. Geometric tracking coverage does not certify a trajectory for physical execution.

From the repository root, with the dependencies and saved mount checkpoints described below, reproduce one complete recording:

```powershell
python -m scripts.run_piperx_controller_event_v4 --trajectory 8-11/Seal_Bag/161504 --mode baseline
```

## Reproducing controller-event tracking

Start with one recorded take and one saved mounting configuration so that the resulting errors have an explicit source and geometry.

### Prepare the Python environment

To run the current solver, use Python 3.11 (as specified in [environment-arm-design.yml](environment-arm-design.yml)) and install the core dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the environment's Python for subsequent commands, or activate it before using `python`. Python examples below run from this same root. [requirements.txt](requirements.txt) includes MuJoCo, NumPy, SciPy, Pandas, Matplotlib, OpenCV, ReportLab, Trimesh, and pytest. [requirements-full.txt](requirements-full.txt) adds Torch, PyYAML, Pillow, imageio-ffmpeg, HDF5, and Qt for the wider research and viewer workflows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-full.txt
```

Keep [data/factory](data/factory/) and [third_party/official_robot_models](third_party/official_robot_models/) available. The CLI discovers recordings under `data/factory`; its `--source-root` option selects mount-search checkpoints, not a different CSV dataset. Offscreen rendering also needs a working MuJoCo graphics backend and video encoder in the execution environment.

> **Note:** Importing `factory_bimanual.robot_contracts` constructs contracts for all five configured robot models, so a PiperX command can require other vendored model assets even when its scene uses only PiperX geometry.

### Solve a recording using its selected mount

For a small comparison window, evaluate the first 300 controller events of `8-11/Seal_Bag/161504` in a separate output directory:

```powershell
python -m scripts.run_piperx_controller_event_v4 --trajectory 8-11/Seal_Bag/161504 --mode baseline --source-prefix 300 --output reports/piperx_controller_event_v4_prefix300
```

The [v4 runner](scripts/run_piperx_controller_event_v4.py) calls `solve_event_shard`, writes a `.scene.xml`, `.trajectory.npz`, and `.summary.json` under `shards/8-11/Seal_Bag/161504/baseline/`, and prints the summary path, paired coverage, safety counts, and derivative maxima. Omitting `--source-prefix` evaluates the full event sequence. The output identity is `EVENT_SOLVER_PROTOCOL = "piperx-controller-event-raw-fixed-time-v4"`.

| Option | Meaning |
|---|---|
| `--trajectory` | Required discovered recording key, such as `8-11/Seal_Bag/161504` |
| `--mode` | Required `baseline`, `upright_table`, `horizontal_wall`, or `inverted` |
| `--source-root` | Checkpoint root; defaults to `reports/piperx_multitask_fixed_time_mount_study` |
| `--output` | Destination root; defaults to `reports/piperx_controller_event_v4` |
| `--source-prefix` | Event-count window, with a minimum of two events and a maximum of the available sequence |
| `--enforce-official-dynamics` | Opt in to the runner's 3 rad/s velocity and 5 rad/s² acceleration bounds |

> **Caution:** The runner consumes an existing `selected_mount` from `checkpoints/<date>/<task>/<take>/<mode>.json` and fails when it is unavailable; solving a v4 shard does not perform a fresh v4 mount optimization.

The [mount comparison section](#comparing-mounting-configurations) explains how those checkpoints are created. Use distinct output roots when comparing enforcement settings or source windows: each run writes the same mode-specific filenames within its chosen root.

> **Note:** Only runs requested with `--enforce-official-dynamics` invoke the final `validate_enforced_dynamics` gate, which rejects non-finite velocity/acceleration evidence or magnitudes above 3 rad/s and 5 rad/s² (with a 1e-9 tolerance); strict paired pose coverage remains a separate metric.

## Reading the current evidence

Compare percentages only after choosing the same recording, target contract, and evaluated window.

### Full recordings

For whole-recording conclusions, use the four summaries under [reports/piperx_controller_event_v4](reports/piperx_controller_event_v4/). The available full runs cover two recordings and two mounting modes:

| Recording | Host poll rows | Controller events | `baseline` paired coverage | `upright_table` paired coverage |
|---|---:|---:|---:|---:|
| `8-11/Seal_Bag/161504` | 2,439 | 1,872 | 100.000% | 89.850% |
| `8-11/Fold_Box/161044` | 1,478 | 755 | 94.834% | 76.159% |

All four full runs have `collision_frames = 0`, `edge_collision_frames = 0`, and `topology_invalid_frames = 0`. They all record `dynamics_enforced=false`. The directory does not contain complete `horizontal_wall` or `inverted` runs for these recordings, so it does not establish a full four-mount comparison.

Read the evidence directly without starting a solver:

```python
import json
from pathlib import Path

root = Path("reports/piperx_controller_event_v4/shards/8-11/Seal_Bag/161504")
for path in sorted(root.rglob("*.summary.json")):
    result = json.loads(path.read_text(encoding="utf-8"))
    print(result["mode"], result["controller_event_rows"],
          f'{100 * result["both_accept_coverage"]:.3f}%',
          result["dynamics_enforced"])
```

### Equal 300-event mount comparisons

For a comparison across all four mounting modes, use [reports/piperx_controller_event_v4_prefix300](reports/piperx_controller_event_v4_prefix300/). It contains eight shards, each limited to the first 300 controller events of its recording:

| Recording, prefix300 only | `baseline` | `upright_table` | `horizontal_wall` | `inverted` |
|---|---:|---:|---:|---:|
| `8-11/Seal_Bag/161504` | 100.000% | 59.667% | 0.000% | 0.000% |
| `8-11/Fold_Box/161044` | 87.000% | 41.667% | 8.333% | 0.000% |

All eight prefix shards also have zero state-collision, edge-collision, and topology-invalid counts, and `dynamics_enforced=false`. Their [bundle manifest](reports/piperx_controller_event_v4_prefix300/bundle_manifest.json) and videos describe this 300-event scope:

| Recording | Synchronized four-panel video |
|---|---|
| Seal_Bag, first 300 events | [Watch mount comparison](reports/piperx_controller_event_v4_prefix300/videos/comparisons/8-11_Seal_Bag_161504_four_mount_fixed_time.mp4) |
| Fold_Box, first 300 events | [Watch mount comparison](reports/piperx_controller_event_v4_prefix300/videos/comparisons/8-11_Fold_Box_161044_four_mount_fixed_time.mp4) |

> **Note:** The four-mount videos and matrix are prefix300 results; the full-run evidence above contains only baseline and upright-table modes.

### Pose acceptance, safety, and joint dynamics

To interpret a successful row, first check both source-validity masks and both arms' pose errors. [build_shard_arrays](scripts/build_piperx_multitask_fixed_time_bundle.py) constructs per-arm acceptance at 1 mm translation and 0.5° orientation, then sets `both_accept = left_accept & right_accept`. `aggregate_shard` computes coverage over every evaluated row, including failures and invalid observations. It also reports the longest rejected run and a separately named valid-source-only coverage.

> **Caution:** `validate_acceptance_evidence` recomputes left, right, and paired acceptance masks from source-valid flags and the strict 1 mm / 0.5° errors; a reported aggregate coverage cannot legitimize altered per-frame acceptance.

Safety is checked independently through state collisions, swept transitions, and topology. Joint velocity and acceleration are further independent arrays. A zero safety count therefore does not mean every target was followed, and a high pose coverage does not mean joint dynamics passed.

> **Caution:** `validate_zero_safety_evidence` checks the saved NPZ collision, swept-edge, and topology arrays; zero-valued JSON summary counts cannot override unsafe trajectory evidence.

| Full-run baseline | Paired coverage | Maximum joint velocity | Maximum joint acceleration |
|---|---:|---:|---:|
| Seal_Bag | 100.000% | 9.295 rad/s | 540.063 rad/s² |
| Fold_Box | 94.834% | 9.760 rad/s | 543.740 rad/s² |

Both full baselines exceed the runner's recorded reference limits of 3 rad/s and 5 rad/s². These are geometric fixed-time tracking results with dynamics enforcement disabled, not trajectories certified for physical execution.

To recompute metrics from the saved Seal_Bag trajectory:

```python
from pathlib import Path
import numpy as np
from scripts.build_piperx_multitask_fixed_time_bundle import aggregate_shard

root = Path("reports/piperx_controller_event_v4/shards/8-11/Seal_Bag/161504/baseline")
with np.load(next(root.glob("*.trajectory.npz")), allow_pickle=False) as archive:
    payload = {name: archive[name] for name in archive.files}
metrics = aggregate_shard(payload)
print(metrics["both_accept_frames"], metrics["source_frames"])
print(metrics["both_accept_coverage"], metrics["longest_hold_frames"])
```

### Diagnostic report

For a shareable explanation of the timing, target, mount, and dynamics findings, open the [Chinese v4 diagnostic PDF](output/pdf/PiperX双手Fixed-Time-v4根因诊断与优化报告.pdf). It presents full baseline results plus the two-task prefix300 comparisons.

```powershell
python -m scripts.build_piperx_controller_event_v4_report
```

> **Note:** The report builder's `build` function uses fixed full/prefix roots, the two task keys above, and the prior raw-source audit; `--output` changes only the PDF destination, not its input dataset or experimental scope.

> **Note:** `validate_report_scope` requires eight exact prefix windows and four full baseline/upright shards across the two tasks, checks solved/total/omitted event counts, and rejects a short prefix presented as a complete recording.

## Keeping recordings and targets aligned

The source row, event time, registration, and tool transform jointly define the target whose error is measured.

### Select a recording and reconstruct its events

When working in Python, discover the real `TrajectorySpec` instead of inventing a filename, checksum, or row count. `TaskFamily` identifies the date/task pair; `TrajectorySpec` adds the take, canonical source hash, path, and host-poll row count.

```python
from pathlib import Path
from factory_bimanual.task_family import TaskFamily
from factory_bimanual.multitask_fixed_time_study import discover_dual_hand_trajectories
from factory_bimanual.source_data import load_factory_task

family = TaskFamily(date="8-11", task="Seal_Bag")
spec = next(item for item in discover_dual_hand_trajectories(Path("data/factory"))
            if item.family == family and item.take == "161504")
task = load_factory_task(spec.path, spec.family.key,
                         timing_mode="controller_updates",
                         max_translation_jump_m=0.20,
                         repair_invalid_pose_rows=True)
print(spec.key, spec.row_count, len(task.time_s), task.timing_source)
print(task.source_row_index[:5])
```

The returned `FactoryBimanualTask` carries paired positions, `wxyz` quaternions, validity masks, optional gripper measurements, timestamps, and original poll-row indices. The full Seal_Bag recording has 2,439 poll rows and 1,872 controller events.

> **Note:** `load_factory_task` defaults to `host_poll`; v4 explicitly requests `controller_updates`, removes paired repeated-frame observations, and retains their original CSV row indices.

Event selection requires matching left/right frame-change masks. It uses the mean of the two receive times and permits a default skew of 1 ms. Missing timing columns, asynchronous updates, excessive receive skew, non-increasing time, and invalid coordinate frames raise `ValueError`.

> **Caution:** Numerically repairing non-finite invalid poses leaves their validity masks intact, so repaired rows still cannot count as accepted tracking.

### Validate a saved event trajectory

Before comparing a saved path with its source, bind it to the selected event timeline and poll-row mapping:

```python
from pathlib import Path
import numpy as np
from factory_bimanual.multitask_fixed_time_study import discover_dual_hand_trajectories
from scripts.run_piperx_controller_event_v4 import validate_event_shard

spec = next(item for item in discover_dual_hand_trajectories(Path("data/factory"))
            if item.key == "8-11/Seal_Bag/161504")
root = Path("reports/piperx_controller_event_v4_prefix300/shards/8-11/Seal_Bag/161504/baseline")
with np.load(next(root.glob("*.trajectory.npz")), allow_pickle=False) as archive:
    payload = {name: archive[name] for name in archive.files}
print(validate_event_shard(payload, spec, "baseline", source_prefix=300))
bad = dict(payload, fixed_time_s=payload["fixed_time_s"] + 0.001)
try:
    validate_event_shard(bad, spec, "baseline", source_prefix=300)
except ValueError as error:
    print(error)  # event shard fixed time differs from source events
```

The successful call returns `True`; the altered schedule is rejected. `SHARD_SCHEMA` identifies the shared array format, while `EVENT_SOLVER_PROTOCOL` identifies the event-v4 solver contract. `event_validation_spec(spec, event_count)` creates a replacement row-count spec without mutating the original recording metadata. `event_initializer_rows` maps checkpoint initializer probes, expressed in host-poll rows, onto event indices using `searchsorted` and clipping. Reusing poll-row numbers as event indices selects different observations.

> **Note:** Manifest and report callers pass the discovered source spec to `validate_archive_summary`, binding the NPZ protocol, source/fixed time, mandatory arrays, summary counts/coverage, raw fixed-tool targets, source-valid masks, and original source-row mapping before accepting the summary.

> **Unlike** the original `validate_shard`, which compares against raw host CSV `t`, `validate_event_shard` compares against reconstructed controller receive events; the validators are not interchangeable.

Both protocols normalize the timeline to a common zero origin and require the execution schedule to equal the chosen source schedule. Neither permits retiming under its fixed-time claim.

### Apply the shared registration and fixed tool mapping

To place a paired recording into a robot scene, use one proper rigid transformation for both hands. A `RigidTaskRegistration` rotates and translates the shared frame; `register_task` returns a `RegisteredBimanualTask` while preserving relative bimanual geometry.

This example follows the runner's shared centering/height registration, then prepares raw fixed-tool targets:

```python
from pathlib import Path
import numpy as np
from factory_bimanual.multitask_fixed_time_study import discover_dual_hand_trajectories
from factory_bimanual.source_data import load_factory_task
from factory_bimanual.registration import RigidTaskRegistration, register_task
from scripts.run_piperx_multitask_fixed_time_mount_study import prepare_family_follow_targets

spec = next(item for item in discover_dual_hand_trajectories(Path("data/factory"))
            if item.key == "8-11/Seal_Bag/161504")
source = load_factory_task(spec.path, spec.family.key,
                           timing_mode="controller_updates",
                           max_translation_jump_m=0.20,
                           repair_invalid_pose_rows=True)
points = np.vstack((source.left_position_m, source.right_position_m))
translation = [-points[:, 0].mean(), -points[:, 1].mean(), 0.90 - points[:, 2].min()]
registered = register_task(source, RigidTaskRegistration(np.eye(3), translation))
task, mapped, audit = prepare_family_follow_targets(
    spec, registered, apply_conditioning=False, apply_wrist_adaptation=False)
print(task.left_position_m.shape, mapped["left"].shape)
print(audit.maximum_position_deviation_m, audit.maximum_orientation_deviation_rad)
```

> **Note:** Shared target preparation defaults to conditioning and wrist adaptation; v4 disables both while retaining configured fixed tool rotations and translations.

The family configuration is read through `load_recommended_config`; `world_mount_for_family` relates its recommended mount to the registration. [tool_frame_calibration](factory_bimanual/tool_frame_calibration.py) contains `CalibrationArtifact`, fixed rotation/translation helpers, and source fingerprints for that mapping. Here “raw” means no conditioning or time-varying wrist adaptation after the fixed tool transform, not an identity transform between handheld and robot TCP frames.

For earlier or explicitly adapted experiments, [trajectory_conditioning](factory_bimanual/trajectory_conditioning.py) exposes `bounded_savgol_se3`, [bounded_orientation_adaptation](factory_bimanual/bounded_orientation_adaptation.py) generates bounded orientation candidates, and [quaternion_trajectory](factory_bimanual/quaternion_trajectory.py) reconstructs held quaternions. These target-preparation choices require their own labeled metrics; their coverage cannot replace raw-target v4 coverage.

## Comparing mounting configurations

Choose installation geometry outside the trajectory solver, then test its ability to support a continuous paired-arm motion.

### Search and reuse mount checkpoints

To inspect the search workload before solving, write the planned job inventory:

```powershell
python -m scripts.run_piperx_multitask_fixed_time_mount_study --dry-run
```

This writes `study_plan.json` under the study output root. `STUDY_MODES` fixes the order below; `StudyConfig` fixes the 1 mm / 0.5° tolerances. `StudyJob` and `plan_jobs` associate each discovered recording with its mode.

| Mode | Installation family |
|---|---|
| `baseline` | Configured family baseline mount |
| `upright_table` | Upright tabletop installation |
| `horizontal_wall` | Horizontal wall installation |
| `inverted` | Inverted installation |

When a required checkpoint is missing, run the corresponding search job:

```powershell
python -m scripts.run_piperx_multitask_fixed_time_mount_study --trajectory 8-11/Seal_Bag/161504 --mode baseline
```

The [search runner](scripts/run_piperx_multitask_fixed_time_mount_study.py) saves its selected geometry and stage results. Its existing search protocol uses the earlier conditioned host-poll targets. The v4 runner then re-evaluates that saved mount under raw fixed-tool controller-event targets. These are two distinct operations: a v4 result does not establish that the mount was optimized under v4.

For search extensions, [orientation_mount_search](factory_bimanual/orientation_mount_search.py) generates orientation candidates and fingerprints, [per_task_mount_search](factory_bimanual/per_task_mount_search.py) ranks full finalists and selects safe layouts, [staged_mount_search](factory_bimanual/staged_mount_search.py) selects full fixed-time candidates and targeted indices, and [spacing_scan](factory_bimanual/spacing_scan.py) generates/ranks base-spacing candidates. Preserve full-run safety and failure evidence when promoting a sparse-search winner.

### Keep native geometry consistent between solver and renderer

To inspect the robot contract used by Seal_Bag's PiperX scene, retrieve its native joint and asset identities:

```python
from factory_bimanual.robot_contracts import get_robot_contract, robot_geometry_sha256

contract = get_robot_contract("piperx")
print(contract.dof_per_arm, contract.prefixed_joint_names("left"))
print(contract.source_urdf, robot_geometry_sha256("piperx"))
```

`BimanualRobotContract` contains native arm joints, joint limits, base/TCP links, and TCP offset. `ROBOT_CONTRACTS` configures `xarm6`, `franka_panda`, `i2rt_yam`, `piperx`, and `ur5`. The v4 solver passes its selected mount to [build_same_model_scene](factory_bimanual/scene_builder.py), which compiles the paired model and returns a `SceneManifest` recording the scene, base positions, spacing, and joint names.

For model development, [strict_urdf_model_audit](scripts/strict_urdf_model_audit.py) and [audit_thirteen_robot_models](scripts/audit_thirteen_robot_models.py) check native models; [build_official_model_provenance_gate](scripts/build_official_model_provenance_gate.py) records model/TCP provenance. Keep the vendored [official models](third_party/official_robot_models/) and their upstream notices/licenses with the assets. A scene's referenced meshes are only one part of runtime and provenance dependencies.

## Understanding tracking failures

A reachable isolated frame does not establish a collision-free, topology-valid, dynamically feasible path through the entire recording.

### Follow paired branches and retain rejected targets

When Seal_Bag or Fold_Box rejects a target, inspect the NPZ's `paired_failure_reason`, `dls_solve_mode`, per-arm errors, source-validity masks, and discontinuity arrays alongside its joint states.

```python
from pathlib import Path
import numpy as np

root = Path("reports/piperx_controller_event_v4/shards/8-11/Seal_Bag/161504/upright_table")
with np.load(next(root.glob("*.trajectory.npz")), allow_pickle=False) as archive:
    rejected = np.flatnonzero(~archive["both_accept"])
    for row in rejected[:5]:
        print(int(archive["source_poll_row_index"][row]),
              float(archive["source_time_s"][row]),
              archive["paired_failure_reason"][row])
```

The shared solver machinery uses [MuJoCoCandidateGenerator](factory_bimanual/mujoco_candidate_generator.py) for native-DOF candidates, [MuJoCoPairedCollisionChecker](factory_bimanual/mujoco_collision_adapter.py) for state and swept-transition collision gates, and [MuJoCoMountTopologyChecker](factory_bimanual/mount_topology.py) for paired-arm topology. [bimanual_collision](factory_bimanual/bimanual_collision.py) defines collision classifications and callback reports.

For alternate branch-selection research, [strict_bimanual_ik](factory_bimanual/strict_bimanual_ik.py) exposes `IKCandidate`, `BimanualIKConfig`, and `solve_strict_bimanual_path`; [collision_safe_follow](factory_bimanual/collision_safe_follow.py) adds explicit pose-tolerance tiers through `solve_collision_safe_follow`. Relaxed tiers must remain distinguishable from the strict 1 mm / 0.5° acceptance metric.

Rejected rows stay on the fixed timeline. `HOLD_FIXED_TIME` in a rendered panel identifies rejected pose tracking; collision-free stationary output can have zero accepted coverage.

### Measure and constrain motion on the immutable timeline

To study a bounded recovery step, use `bounded_joint_step` and retain its returned `BoundedJointStep` velocity as state for the next interval:

```python
import numpy as np
from factory_bimanual.fixed_time_tracking import bounded_joint_step, fixed_time_derivatives

step = bounded_joint_step(
    previous_q=np.array([0.0]), desired_q=np.array([0.2]),
    previous_velocity_rad_s=np.array([0.5]),
    dt_s=0.02, previous_dt_s=0.02,
    velocity_limit_rad_s=3.0, acceleration_limit_rad_s2=5.0,
    lower_rad=np.array([-1.0]), upper_rad=np.array([1.0]))
velocity, acceleration = fixed_time_derivatives(
    np.array([[0.0], step.q]), np.array([0.0, 0.02]),
    initial_velocity_rad_s=0.5)
print(step.q, step.limited, velocity[-1], acceleration[-1])
```

This one-joint example isolates the timing primitive; it does not perform collision checking or certify a robot path. `fixed_time_derivatives` uses the actual event intervals and returns frame-aligned velocity/acceleration evidence. For a MuJoCo trajectory, select the actuated joint columns rather than treating all `qpos` columns as arm joints.

> **Caution:** An acceleration-bounded step can continue moving while braking after a hold request and can raise when no feasible step exists; HOLD does not imply an instantaneous zero-velocity command.

[fixed_time_refinement](factory_bimanual/fixed_time_refinement.py) projects a joint path under fixed timestamps and derivative bounds. [fixed_time_evidence_audit](factory_bimanual/fixed_time_evidence_audit.py) separates task-space segment rates, pose errors, strict paired acceptance, and comparisons between target tracks. Refinement changes a candidate path; an audit measures the path and target contract that were actually saved.

To run a separate enforced-dynamics experiment on the same Seal_Bag window:

```powershell
python -m scripts.run_piperx_controller_event_v4 --trajectory 8-11/Seal_Bag/161504 --mode baseline --source-prefix 300 --enforce-official-dynamics --output reports/piperx_controller_event_v4_dynamics_prefix300
```

The command requests bounded motion; its resulting pose coverage must be measured anew. It does not inherit the 100% geometric baseline result.

> **Caution:** With dynamics enforcement enabled, failure to find a safe bounded step stops the solve, and returned trajectories still face the final derivative gate; this local solver failure does not prove that every continuous IK branch is physically infeasible.

### Distinguish alternative execution contracts

When studying execution changes, retain their time and target labels with the results. [complete_follow](factory_bimanual/complete_follow.py) exposes `CompleteFollowRunner` and `retime_complete_source_path`; [recommended_follow](factory_bimanual/recommended_follow.py) implements the recommended-v3.1 follower; [rescue_v31](factory_bimanual/rescue_v31.py) schedules branch rescue and transitions.

> **Unlike** fixed-time v4, complete-follow can retime a source-order path; complete spatial coverage under that workflow is not fixed-time coverage.

For controller comparisons, [native_dof_easyik](factory_bimanual/native_dof_easyik.py), [native_dof_mpc](factory_bimanual/native_dof_mpc.py), [easyik_runner](factory_bimanual/easyik_runner.py), and [mpc_runner](factory_bimanual/mpc_runner.py) provide native-DOF EasyIK/MPC controllers and runners. [controller_adapter](factory_bimanual/controller_adapter.py) connects scene state and targets; [branch_equivalence](factory_bimanual/branch_equivalence.py) compares initial branches before scheduling conditional MPC runs. Their output contracts require separate validation before comparison with v4.

## Producing shareable comparisons

Build videos from validated saved trajectories so that the same source window is visible in every mounting panel.

### Build a four-mode manifest

Once all four prefix shards exist for each included trajectory, assemble their rendering manifest:

```powershell
python -m scripts.build_piperx_controller_event_manifest --output reports/piperx_controller_event_v4_prefix300
```

[build_manifest](scripts/build_piperx_controller_event_manifest.py) checks the v4 protocol, no-retiming flag, zero safety counts, and identical event timelines across a recording's four modes. It then writes `bundle_manifest.json`.

> **Caution:** Building this manifest normalizes and rewrites the source summary JSON files, so this command is not read-only validation.

> **Note:** The manifest's gates do not require 100% pose coverage or passed dynamics; a renderable shard is not a deployable trajectory.

The full v4 directory currently lacks wall and inverted shards for both recordings and therefore cannot supply the required four-mode set. Use the prefix300 root for the existing four-panel deliverable.

### Render synchronized panels

To render the Seal_Bag comparison, select its manifest entry:

```powershell
python -m scripts.render_piperx_multitask_mount_comparisons --output reports/piperx_controller_event_v4_prefix300 --trajectory 8-11/Seal_Bag/161504
```

Omit `--trajectory` to render every recording in the manifest. `render_trajectory` coordinates `render_panel` and `compose_four_panel`, producing a synchronized 1280×720 MP4 at 30 fps. [mount_comparison_visuals](factory_bimanual/mount_comparison_visuals.py) defines comparison timelines, source-frame indices, and mount ranking; [video](factory_bimanual/video.py) handles timing, provenance, and `decode_check_mp4` checks.

> **Note:** Panel diagnostics retain source poll-row provenance and label rejected poses `HOLD_FIXED_TIME`, so a motionless collision-free panel can still represent zero accepted coverage.

> **Note:** Render-cache validity depends on hashes of the summary, trajectory archive, scene, and render protocol, not on the existence of an MP4 alone.

For additional report formats, [artifacts](factory_bimanual/artifacts.py) supplies `FrameDiagnostics`, `RunArtifactPaths`, and `RunArtifactWriter`; [report](factory_bimanual/report.py) writes common comparisons; [per_task_mount_report](factory_bimanual/per_task_mount_report.py) writes evidence-preserving metric rows; and [orientation_comparison_metrics](factory_bimanual/orientation_comparison_metrics.py) derives orientation metrics from immutable arrays. Generate summaries from the stored evidence before choosing a presentation format.

### Put the prefix workflow together

After the individual commands above, the complete two-task prefix comparison is one loop over saved mounts followed by manifest construction and rendering:

```powershell
$eventOutput = "reports/piperx_controller_event_v4_prefix300"
$eventTasks = @("8-11/Seal_Bag/161504", "8-11/Fold_Box/161044")
$eventModes = @("baseline", "upright_table", "horizontal_wall", "inverted")
foreach ($eventTask in $eventTasks) {
    foreach ($eventMode in $eventModes) {
        python -m scripts.run_piperx_controller_event_v4 --trajectory $eventTask --mode $eventMode --source-prefix 300 --output $eventOutput
        if ($LASTEXITCODE -ne 0) { throw "Event shard failed: $eventTask / $eventMode" }
    }
}
python -m scripts.build_piperx_controller_event_manifest --output $eventOutput
if ($LASTEXITCODE -ne 0) { throw "Event manifest failed" }
python -m scripts.render_piperx_multitask_mount_comparisons --output $eventOutput
if ($LASTEXITCODE -ne 0) { throw "Event rendering failed" }
```

This recomputes the eight prefix artifacts. The diagnostic PDF additionally requires four full runs (baseline and upright-table for each task) and its fixed prior-audit input, so the prefix loop alone does not satisfy all report inputs.

## Exploring the wider research toolkit

Use the additional workflows for their stated recording, geometry, and execution contracts while keeping the current v4 evidence identifiable.

### Reproduce the earlier conditioned host-poll study

For the historical 27-trajectory × 4-mode study, use [reports/piperx_multitask_fixed_time_mount_study](reports/piperx_multitask_fixed_time_mount_study/). Its 108 cells use the earlier conditioned host-poll fixed-time protocol. The [Chinese report](reports/piperx_multitask_fixed_time_mount_study/PiperX多任务Fixed-Time四构型对比报告.pdf), [aggregate CSV](reports/piperx_multitask_fixed_time_mount_study/aggregate.csv), [bundle manifest](reports/piperx_multitask_fixed_time_mount_study/bundle_manifest.json), [release manifest](reports/piperx_multitask_fixed_time_mount_study/release_manifest.json), and [27 comparison videos](reports/piperx_multitask_fixed_time_mount_study/videos/comparisons/) remain available as historical evidence.

```powershell
python -m scripts.build_piperx_multitask_fixed_time_bundle --validate-only
python -m scripts.run_piperx_multitask_fixed_time_mount_study --trajectory 8-12/PourRawMaterial/111542 --mode baseline
```

The first command validates the existing historical bundle; the second resumes a historical mount-search job. [PourRawMaterial's comparison video](reports/piperx_multitask_fixed_time_mount_study/videos/comparisons/8-12_PourRawMaterial_111542_four_mount_fixed_time.mp4) is an example of that earlier protocol, not a v4 recording result.

The historical [raw-source audit](reports/piperx_multitask_fixed_time_mount_study/literature_audit/) and [project optimization/literature report](output/pdf/PiperX双手多任务Fixed-Time项目优化与文献审计报告.pdf) examine differences between conditioned and raw-target evidence. Their 27×4 matrix and deployment conclusions must retain that scope. Use [audit_piperx_fixed_time_evidence](scripts/audit_piperx_fixed_time_evidence.py) for the older bundle audit, [build_piperx_multitask_mount_report](scripts/build_piperx_multitask_mount_report.py) for its report, and [build_piperx_multitask_release_manifest](scripts/build_piperx_multitask_release_manifest.py) for its release identity.

Earlier [two-task fixed-time](reports/piperx_two_task_fixed_time/), [complete-follow](reports/piperx_complete_follow/), [two-task complete-follow](reports/piperx_two_task_complete_follow/), and [recommended-v3.1](reports/piperx_recommended_v31/) artifacts are retained. Their corresponding [two-task validator](scripts/validate_piperx_two_task_bundle.py), [complete-follow report builder](scripts/build_piperx_complete_follow_report.py), and [recommended-v3.1 runner](scripts/run_piperx_recommended_v31.py) remain separate entry points. “v4.0” in an older complete-follow report title does not identify the controller-event v4 protocol.

### Stage a general multi-robot experiment

For the separate factory controller matrix, start by inspecting its planned workload and preflight outputs:

```powershell
python -m factory_bimanual.cli dry-run
python -m factory_bimanual.cli preflight
python -m factory_bimanual.cli short-prefix --rows 2
```

The [general CLI](factory_bimanual/cli.py) wires `FactoryBimanualExperiment` to `ProductionExecutors` and writes under `reports/factory_bimanual`. Its task defaults, source catalog, workspace-feasibility checks, and spacing validation are in [defaults](factory_bimanual/defaults.py), [factory_task_catalog](factory_bimanual/factory_task_catalog.py), [workspace_feasibility](factory_bimanual/workspace_feasibility.py), and [preflight](factory_bimanual/preflight.py). [isolation](factory_bimanual/isolation.py) snapshots and checks protected single-arm files around these experiments.

> **Note:** Preflight writes reports and compiled scenes; a full general-CLI run requires `full --confirm-full` and successful preflight.

```powershell
python -m factory_bimanual.cli full --confirm-full
```

These commands execute the general factory workflow; use `scripts.run_piperx_controller_event_v4` for the event-v4 reproduction demonstrated above.

### Retained single-arm reports and videos

To inspect single-arm work without running a search, open the retained deliverables:

| Study | Report | Video or artifact directory |
|---|---|---|
| Mount/IK fidelity pilot | [PDF](reports/single_arm/mount_ik_fidelity_pilot/pdf/mount_ik_fidelity_pilot_report.pdf) | [OpenArm](reports/single_arm/mount_ik_fidelity_pilot/videos/openarm.mp4), [xArm6](reports/single_arm/mount_ik_fidelity_pilot/videos/xarm6.mp4) |
| Ten arms, three pick episodes | [PDF](reports/single_arm/ten_arm_pick_right_left_three_episodes/report.pdf) | [Study artifacts](reports/single_arm/ten_arm_pick_right_left_three_episodes/) |
| Fixed-time versus legacy, ten arms | [PDF](reports/single_arm/ten_arm_two_single_tasks_handbook_fixed_4096/ten_arm_fixed_vs_legacy_report.pdf) | [Study artifacts](reports/single_arm/ten_arm_two_single_tasks_handbook_fixed_4096/) |

The [single-arm project manifest](SINGLE_ARM_PROJECT_MANIFEST.json) and [reports index](reports/README.md) provide additional navigation. Older report titles and recommendation language describe their own experiment revisions; the full/prefix scope tables in this README identify the controller-event v4 evidence.

### Extend geometry, search, and optimization

When adding robot models or alternative search policies, follow the native model through forward kinematics, candidate evaluation, and full-trajectory validation. The research modules group into these connected surfaces:

| Task | Modules and principal interfaces |
|---|---|
| Load native robot geometry | [robot_registry](design_optimization/robot_registry.py): `RobotSpec`, `load_robot_registry`; [urdf_chain](design_optimization/urdf_chain.py): `NativeSerialChain`, `load_native_chain`, `load_urdf_chain`, `load_mjcf_chain`, `fk_flange`, `sampled_maximum_reach` |
| Build parametric geometry and compute poses | [topology](design_optimization/topology.py): `TopologyTemplate`, `DesignBatch`, `load_templates`; [kinematics](design_optimization/kinematics.py): `build_designs`, `assemble_from_deltas`, `fk_tcp`, `geometric_jacobian`, joint-world positions |
| Solve and select IK branches | [ik](design_optimization/ik.py): `IKResult`, `solve_multistart`, `solve_trajectory_multistart`, `select_continuous_branches`; [trajectory](design_optimization/trajectory.py): `BimanualTrajectory`, `select_bimanual_collision_aware` |
| Map task frames and score clearance | [taskspace](design_optimization/taskspace.py): `BimanualTaskFrames`, transform/quaternion helpers, `world_to_base`; [collision](design_optimization/collision.py): `CollisionMetrics`, capsule clearances and penalties |
| Prepare reproducible data | [experiment](design_optimization/experiment.py): data/IK/search/constraint configs, checkpoints and `reproducibility_manifest`; [egodex](design_optimization/egodex.py): `EgoDexPoseDataset`; [local_pose_dataset](design_optimization/local_pose_dataset.py) and [local_pose_sampling](design_optimization/local_pose_sampling.py): cleaning, splits, trimming and relative sampling |
| Set installation bounds and candidate budgets | [installation_search_space](design_optimization/installation_search_space.py), [search_policy](design_optimization/search_policy.py), [mount_sweep](design_optimization/mount_sweep.py): feasible mounts, staged Sobol candidates and mount scores |
| Promote candidates through evaluation fidelities | [best_first_mount_search](design_optimization/best_first_mount_search.py): `SearchBudget`, `CandidateState`, `Frontier`; [hierarchical_mount_search](design_optimization/hierarchical_mount_search.py): global/local candidates; [fidelity_funnel](design_optimization/fidelity_funnel.py): shortlist and cross-fidelity metrics |
| Bound memory and reuse evaluated frames | [batched_search](design_optimization/batched_search.py): `safe_candidate_batch_size`, `evaluate_candidate_batches`; [incremental_mount_evaluator](design_optimization/incremental_mount_evaluator.py): `FrameResult`, `CandidateEvaluation`, `IncrementalMountEvaluator` |
| Rank designs and quantify robustness | [objectives](design_optimization/objectives.py): `PopulationMetrics`, `evaluate_population`; [pareto](design_optimization/pareto.py): nondominated sorting and NSGA-II; [robustness](design_optimization/robustness.py): Sobol perturbations and upper-tail CVaR |
| Learn approximations and refine paths | [surrogate](design_optimization/surrogate.py): `ObjectiveSurrogate`, ensemble prediction and UCB proposals; [reach](design_optimization/reach.py): numerical reach; [slp_refinement](design_optimization/slp_refinement.py): `refine_joint_window_slp` |
| Explore runtime policies and diagnostics | [realtime](design_optimization/realtime.py): bounded solve scheduling and proposal mailbox; [retiming](design_optimization/retiming.py): `retime_joint_path`; [branch_policy](design_optimization/branch_policy.py), [collision_classifier](design_optimization/collision_classifier.py), [episode_follow_metrics](design_optimization/episode_follow_metrics.py): learned proposals and failure metrics |

Native geometry auditing and parametric geometry optimization serve different purposes. A parametric winner needs native-model validation before its measured coverage can support a claim about a vendored robot. Likewise, collision proxies and learned proposals accelerate candidate evaluation; the final MuJoCo state and transition evidence still determines the reported safety counts.

### Locate task-specific runners and viewers

For a particular dataset or experiment revision, choose a script by the job it performs and inspect that script's arguments before running a long search:

| Work | Entry points |
|---|---|
| Single-arm formal studies | [fidelity pilot](scripts/run_mount_ik_fidelity_pilot.py), [ten-arm three-episode study](scripts/run_ten_arm_three_pick_episodes.py), [ten-arm two-task study](scripts/run_ten_arm_two_single_tasks.py), [twelve-arm Local study](scripts/run_twelve_arm_all_single_tasks.py) |
| Native model validation and replay | [strict model audit](scripts/strict_urdf_model_audit.py), [strict mount search](scripts/search_strict_urdf_mount.py), [strict task cache solver](scripts/solve_strict_urdf_task_cache.py), [single-arm renderer](scripts/render_strict_single_arm_task.py), [strict batch queue](scripts/run_strict_cache_batch.py) |
| External-domain preparation and search | [Local benchmark preparation](scripts/prepare_local_pose_benchmark.py), [DROID trace preparation](scripts/build_droid_5min_trace.py), [EgoDex pose download](scripts/download_egodex_pose_only.py), [external-domain dense search](scripts/run_external_domain_dense_search.py), [held-out validation](scripts/validate_external_domain_test.py) |
| Earlier PiperX search and calibration | [per-task mount search](scripts/run_piperx_factory_per_task_mount_search.py), [orientation comparison](scripts/run_piperx_seal_bag_orientation_comparison.py), [tool calibration](scripts/calibrate_piperx_tool_frames.py), [fixed-time mount search](scripts/search_piperx_fixed_time_mounts.py) |
| Controller and branch investigations | [recommended Doosan MPC](scripts/run_recommended_doosan_mpc.py), [robot MPC tuning](scripts/tune_robot_mpc.py), [rolling branch selection](scripts/rolling_multibranch_ik.py), [branch-plan validation](scripts/validate_branch_plan.py), [realtime trace analysis](scripts/analyze_realtime_trace.py) |
| Learning experiments | [branch policy training](scripts/train_branch_proposal_policy.py), [collision classifier training](scripts/train_collision_classifier_cuda.py), [surrogate training](scripts/train_surrogate_from_archive.py) |
| Inspect motion and scene assembly | [desktop episode viewer](tools/episode-3d-desktop/), [web episode viewer](tools/episode-3d-viewer/), [Python desktop viewer](scripts/episode_3d_desktop.py), [MuJoCo viewer launcher](factory_bimanual/launch_viewer.py), [assembly renderer](scripts/render_model_assembly_qa.py) |

For example, inspect the native single-arm solver's available inputs without launching a solve:

```powershell
python -m scripts.solve_strict_urdf_task_cache --help
python -m scripts.episode_3d_desktop --help
```

The remaining [scripts](scripts/) include study-specific figure/PDF builders, cache re-audits, CUDA geometry searches, collision-proxy calibration, and older renderers. Keep saved configurations and provenance with exported results so that single-arm studies, historical bimanual experiments, and raw controller-event v4 results remain traceable to the experiment that produced them.

### Public API inventory

When you need an exact class or function name beyond the worked examples, use this source-module index. It lists 926 public top-level class/function definitions across 256 first-party modules (673 distinct names); repeated names such as `main` are listed under each defining module. Class methods are represented by their owning class. These names cover the current and historical workflows above, so an entry alone does not establish a v4 execution contract.

<details>
<summary>design_optimization: 31 modules, 147 definitions</summary>

| Source module | Public classes and functions |
|---|---|
| [batched_search.py](design_optimization/batched_search.py) | `safe_candidate_batch_size`, `evaluate_candidate_batches` |
| [best_first_mount_search.py](design_optimization/best_first_mount_search.py) | `Fidelity`, `SearchBudget`, `CandidateState`, `Frontier`, `promotion_target`, `should_stop_after_success`, `representative_frame_indices`, `representative_frame_windows`, `select_diverse_regions`, `generate_local_population` |
| [branch_policy.py](design_optimization/branch_policy.py) | `BranchPolicyOutput`, `MultiBranchIKPolicy`, `wrapped_joint_error`, `branch_imitation_loss` |
| [collision_classifier.py](design_optimization/collision_classifier.py) | `periodic_joint_features`, `ConfigurationCollisionClassifier`, `binary_metrics` |
| [collision.py](design_optimization/collision.py) | `CollisionMetrics`, `segment_segment_distance`, `self_segment_pair_distances`, `self_capsule_clearance`, `dual_capsule_clearance`, `table_capsule_clearance`, `collision_penalty` |
| [egodex.py](design_optimization/egodex.py) | `EgoDexSplit`, `EgoDexPoseDataset` |
| [episode_follow_metrics.py](design_optimization/episode_follow_metrics.py) | `classify_solver_failures`, `planner_failure_diagnostics`, `failure_reason_diagnostics`, `episode_follow_metrics`, `follow_rank` |
| [experiment.py](design_optimization/experiment.py) | `DataConfig`, `IKConfig`, `SearchConfig`, `ConstraintConfig`, `ExperimentConfig`, `seed_everything`, `save_checkpoint_atomic`, `load_checkpoint`, `append_jsonl`, `sha256_file`, `reproducibility_manifest` |
| [fidelity_funnel.py](design_optimization/fidelity_funnel.py) | `select_diverse_candidates`, `cross_fidelity_metrics` |
| [hierarchical_mount_search.py](design_optimization/hierarchical_mount_search.py) | `MountSearchBudget`, `global_mount_candidates`, `diverse_region_indices`, `local_mount_candidates` |
| [ik.py](design_optimization/ik.py) | `IKResult`, `SelectedIKPath`, `reverse_ik_time`, `concatenate_ik_branches`, `rotation_log`, `pose_error`, `adaptive_damping`, `joint_limit_centering_velocity`, `solve_multistart`, `deterministic_seeds`, `solve_trajectory_multistart`, `select_continuous_branches` |
| [incremental_mount_evaluator.py](design_optimization/incremental_mount_evaluator.py) | `FrameResult`, `CandidateEvaluation`, `IncrementalMountEvaluator` |
| [installation_search_space.py](design_optimization/installation_search_space.py) | `InstallationSearchSpace`, `first_version_bounds`, `mount_rotation_matrix`, `tabletop_mount_feasible`, `mount_transform_from_record`, `torch_mount_rotation_matrices` |
| [kinematics.py](design_optimization/kinematics.py) | `axis_angle_matrix`, `build_designs`, `assemble_from_deltas`, `fk_flange`, `fk_tcp`, `geometric_jacobian`, `joint_world_positions`, `deterministic_joint_samples` |
| [local_pose_dataset.py](design_optimization/local_pose_dataset.py) | `CleanedPoseFrame`, `CleaningRecord`, `infer_hands`, `content_split`, `assign_stratified_splits`, `clean_pose_frame`, `prepare_local_pose_benchmark`, `trajectory_has_motion` |
| [local_pose_sampling.py](design_optimization/local_pose_sampling.py) | `trim_episode_edges`, `relative_pose_sample` |
| [mount_sweep.py](design_optimization/mount_sweep.py) | `sobol_mount_candidates`, `mount_quality_score` |
| [objectives.py](design_optimization/objectives.py) | `PopulationMetrics`, `topology_intersection_error`, `motor_clearance_violation`, `evaluate_population` |
| [pareto.py](design_optimization/pareto.py) | `ParetoRanking`, `constraint_dominates`, `nondominated_sort`, `crowding_distance`, `rank_population`, `nsga2_select`, `tournament_parents`, `make_offspring` |
| [reach.py](design_optimization/reach.py) | `numerical_max_flange_reach` |
| [realtime.py](design_optimization/realtime.py) | `ServoMode`, `RealtimeBudget`, `ServoObservation`, `SolveSchedule`, `BranchProposal`, `BranchProposalMailbox`, `schedule_bounded_solve`, `govern_cartesian_target` |
| [retiming.py](design_optimization/retiming.py) | `RetimedPath`, `retime_joint_path` |
| [robot_registry.py](design_optimization/robot_registry.py) | `RobotSpec`, `load_robot_registry` |
| [robustness.py](design_optimization/robustness.py) | `symmetric_sobol_perturbations`, `upper_tail_cvar` |
| [search_policy.py](design_optimization/search_policy.py) | `dimension_aware_candidate_budget`, `staged_sobol_candidates` |
| [slp_refinement.py](design_optimization/slp_refinement.py) | `SLPRefinementResult`, `refine_joint_window_slp` |
| [surrogate.py](design_optimization/surrogate.py) | `ObjectiveSurrogate`, `gaussian_nll`, `EnsemblePrediction`, `ensemble_predict`, `standardize_targets`, `propose_ucb_candidates` |
| [taskspace.py](design_optimization/taskspace.py) | `BimanualTaskFrames`, `mirrored_mount_transforms`, `quaternion_wxyz_to_matrix`, `make_transform`, `canonical_egodex_targets`, `world_to_base`, `world_to_base_population` |
| [topology.py](design_optimization/topology.py) | `TopologyTemplate`, `DesignBatch`, `load_templates` |
| [trajectory.py](design_optimization/trajectory.py) | `BimanualTrajectory`, `outer_elbow_penalty`, `select_bimanual_collision_aware` |
| [urdf_chain.py](design_optimization/urdf_chain.py) | `NativeSerialChain`, `load_urdf_chain`, `load_mjcf_chain`, `load_native_chain`, `fk_flange`, `sampled_maximum_reach` |

</details>

<details>
<summary>factory_bimanual: 51 modules, 188 definitions</summary>

| Source module | Public classes and functions |
|---|---|
| [artifacts.py](factory_bimanual/artifacts.py) | `FrameDiagnostics`, `RunArtifactPaths`, `RunArtifactWriter` |
| [bimanual_collision.py](factory_bimanual/bimanual_collision.py) | `CollisionClass`, `CollisionReport`, `CallbackCollisionChecker` |
| [bounded_orientation_adaptation.py](factory_bimanual/bounded_orientation_adaptation.py) | `OrientationAdaptationConfig`, `OrientationCandidate`, `orientation_offset_angles`, `orientation_adaptation_candidates` |
| [branch_equivalence.py](factory_bimanual/branch_equivalence.py) | `BranchDecision`, `compare_initial_branches` |
| [cli.py](factory_bimanual/cli.py) | `main` |
| [collision_safe_follow.py](factory_bimanual/collision_safe_follow.py) | `PoseToleranceTier`, `CollisionSafeFollowConfig`, `CollisionSafeFollowResult`, `solve_collision_safe_follow` |
| [complete_follow.py](factory_bimanual/complete_follow.py) | `CompleteRetiming`, `CompleteFollowResult`, `retime_complete_source_path`, `CompleteFollowInfeasibleError`, `CompleteFollowRunner` |
| [controller_adapter.py](factory_bimanual/controller_adapter.py) | `make_easyik`, `make_mpc`, `set_bimanual_targets` |
| [defaults.py](factory_bimanual/defaults.py) | `derive_initial_registration`, `TaskSpec` |
| [easyik_runner.py](factory_bimanual/easyik_runner.py) | `EasyIKBudget`, `EasyIKScene`, `EasyIKResult`, `run_easyik_task` |
| [executors.py](factory_bimanual/executors.py) | `ProductionExecutors` |
| [factory_task_catalog.py](factory_bimanual/factory_task_catalog.py) | `FactoryEpisode`, `TaskRepresentative`, `ExcludedFactoryInput`, `FactoryTaskCatalog`, `build_factory_task_catalog`, `write_dataset_manifest` |
| [fixed_time_evidence_audit.py](factory_bimanual/fixed_time_evidence_audit.py) | `TaskspaceSegmentRates`, `PoseTrackingErrors`, `TargetTrackComparison`, `taskspace_segment_rates`, `pose_tracking_errors`, `strict_pair_accept`, `compare_target_tracks` |
| [fixed_time_refinement.py](factory_bimanual/fixed_time_refinement.py) | `FixedTimeRefinementResult`, `refine_fixed_time_joint_path` |
| [fixed_time_run_contract.py](factory_bimanual/fixed_time_run_contract.py) | `source_time_schedule`, `validate_synchronized_mount` |
| [fixed_time_tracking.py](factory_bimanual/fixed_time_tracking.py) | `BoundedJointStep`, `fixed_time_derivatives`, `bounded_joint_step` |
| [four_cell_viewer.py](factory_bimanual/four_cell_viewer.py) | `assert_reference_demo_only`, `load_reference_model`, `lightweight_scene_text`, `mapped_relative_pose`, `LightweightFourCellIK`, `main` |
| [isolation.py](factory_bimanual/isolation.py) | `protected_paths`, `snapshot_protected_files`, `assert_protected_files_unchanged` |
| [launch_viewer.py](factory_bimanual/launch_viewer.py) | `main` |
| [mount_comparison_visuals.py](factory_bimanual/mount_comparison_visuals.py) | `comparison_timeline`, `source_frame_indices`, `hex_to_bgr`, `audited_mount_rank` |
| [mount_orientation.py](factory_bimanual/mount_orientation.py) | `mount_quaternions` |
| [mount_topology.py](factory_bimanual/mount_topology.py) | `MountTopologyConfig`, `MountTopologyReport`, `evaluate_mount_topology_positions`, `MuJoCoMountTopologyChecker` |
| [mpc_runner.py](factory_bimanual/mpc_runner.py) | `MPCScene`, `BimanualMPCResult`, `scheduled_mpc_runs`, `run_mpc_task` |
| [mujoco_candidate_generator.py](factory_bimanual/mujoco_candidate_generator.py) | `normalized_pose_residual`, `stratified_joint_seeds`, `piperx_wrist_risk`, `CandidateGeneratorConfig`, `MuJoCoCandidateGenerator` |
| [mujoco_collision_adapter.py](factory_bimanual/mujoco_collision_adapter.py) | `ClearanceReport`, `MuJoCoPairedCollisionChecker` |
| [multitask_fixed_time_study.py](factory_bimanual/multitask_fixed_time_study.py) | `TrajectorySpec`, `StudyConfig`, `discover_dual_hand_trajectories`, `validate_shard` |
| [native_dof_easyik.py](factory_bimanual/native_dof_easyik.py) | `NativeDofDualArmEasyIKPVT` |
| [native_dof_mpc.py](factory_bimanual/native_dof_mpc.py) | `UnsupportedModelControllerContract`, `MPCConfig`, `ArmState`, `NativeDofDualArmMPCPVT`, `NativeDofDualArmQPServoPVT` |
| [orientation_comparison_metrics.py](factory_bimanual/orientation_comparison_metrics.py) | `derive_orientation_metrics` |
| [orientation_mount_search.py](factory_bimanual/orientation_mount_search.py) | `OrientationSearchConfig`, `generate_mounts`, `mount_fingerprint`, `evenly_spaced` |
| [per_task_mount_report.py](factory_bimanual/per_task_mount_report.py) | `build_metric_rows`, `write_comparison_metrics` |
| [per_task_mount_search.py](factory_bimanual/per_task_mount_search.py) | `PerTaskSearchConfig`, `load_registered_representative`, `prefix_registered_task`, `candidate_fingerprint`, `rank_full_finalist`, `summarize_quality_arrays`, `select_safe_layout` |
| [piperx_recommended.py](factory_bimanual/piperx_recommended.py) | `StrictAcceptConfig`, `DLSConfig`, `ExecutionConfig`, `MountFunnelConfig`, `WristAdaptationSpec`, `RecommendedMountSpec`, `PiperXRecommendedConfig`, `WorldMount`, `load_recommended_config`, `world_mount_for_family` |
| [preflight.py](factory_bimanual/preflight.py) | `validate_spacing_evidence`, `run_preflight` |
| [quaternion_trajectory.py](factory_bimanual/quaternion_trajectory.py) | `QuaternionReconstruction`, `quaternion_poses_equal`, `reconstruct_held_quaternions` |
| [recommended_follow.py](factory_bimanual/recommended_follow.py) | `RecommendedFollowResult`, `RecommendedFollowRunner` |
| [registration.py](factory_bimanual/registration.py) | `RigidTaskRegistration`, `RegisteredBimanualTask`, `register_task` |
| [report.py](factory_bimanual/report.py) | `ReportPaths`, `write_comparison_report` |
| [rescue_v31.py](factory_bimanual/rescue_v31.py) | `StrictGate`, `shortest_joint_delta`, `wrist_branch_signature`, `trapezoidal_transition_time`, `minimum_jerk_transition`, `CandidateFrame`, `RescueScheduleConfig`, `RescueEvent`, `FollowSchedule`, `schedule_rescue_v31` |
| [robot_contracts.py](factory_bimanual/robot_contracts.py) | `BimanualRobotContract`, `robot_geometry_sha256`, `get_robot_contract` |
| [run_experiment.py](factory_bimanual/run_experiment.py) | `ExperimentConfig`, `ExperimentJob`, `FactoryBimanualExperiment` |
| [scene_builder.py](factory_bimanual/scene_builder.py) | `SceneManifest`, `build_same_model_scene` |
| [source_data.py](factory_bimanual/source_data.py) | `FactoryBimanualTask`, `load_factory_task` |
| [spacing_scan.py](factory_bimanual/spacing_scan.py) | `SpacingCandidateRow`, `SpacingSelection`, `generate_spacing_candidates`, `rank_spacing_candidates` |
| [staged_mount_search.py](factory_bimanual/staged_mount_search.py) | `rank_full_fixed_time_mount`, `select_full_fixed_time_mount`, `targeted_source_indices` |
| [strict_bimanual_ik.py](factory_bimanual/strict_bimanual_ik.py) | `IKCandidate`, `BimanualIKConfig`, `BimanualIKResult`, `solve_strict_bimanual_path` |
| [task_family.py](factory_bimanual/task_family.py) | `TaskFamily`, `family_from_path` |
| [tool_frame_calibration.py](factory_bimanual/tool_frame_calibration.py) | `source_file_fingerprint`, `proper_axis_rotations`, `apply_fixed_tool_rotation`, `apply_fixed_tool_translation`, `apply_bounded_wrist_adaptation`, `validate_local_refinement_deg`, `fixed_offset_quaternion`, `representative_quaternion_indices`, `rank_calibration_result`, `CalibrationArtifact` |
| [trajectory_conditioning.py](factory_bimanual/trajectory_conditioning.py) | `ConditioningAudit`, `ConditionedTrajectory`, `bounded_savgol_se3` |
| [video.py](factory_bimanual/video.py) | `RealtimeVideoTiming`, `VideoRenderConfig`, `interpolation_sample`, `is_replanned_transition`, `execution_status_label`, `VideoDecodeCheck`, `VideoRenderResult`, `build_realtime_timing`, `write_video_provenance`, `decode_check_mp4`, `render_mujoco_mp4` |
| [workspace_feasibility.py](factory_bimanual/workspace_feasibility.py) | `TrajectoryGeometry`, `trajectory_geometry`, `audit_workspace_feasibility` |

</details>

<details>
<summary>scripts: 174 modules, 591 definitions</summary>

| Source module | Public classes and functions |
|---|---|
| [analyze_axis_topology.py](scripts/analyze_axis_topology.py) | `unsigned_axis_angle_deg`, `line_distance`, `main` |
| [analyze_legacy_realtime_latency.py](scripts/analyze_legacy_realtime_latency.py) | `main` |
| [analyze_realtime_trace.py](scripts/analyze_realtime_trace.py) | `percentile`, `main` |
| [audit_all_local_table_clearance.py](scripts/audit_all_local_table_clearance.py) | `audit_episode`, `main` |
| [audit_egodex_input_dynamics.py](scripts/audit_egodex_input_dynamics.py) | `summarize`, `main` |
| [audit_episode_collision_free_candidates.py](scripts/audit_episode_collision_free_candidates.py) | `main` |
| [audit_external_domains.py](scripts/audit_external_domains.py) | `bucket` |
| [audit_outer_elbow_posture.py](scripts/audit_outer_elbow_posture.py) | `main` |
| [audit_parametric_topologies.py](scripts/audit_parametric_topologies.py) | `build_audit`, `main` |
| [audit_piperx_fixed_time_evidence.py](scripts/audit_piperx_fixed_time_evidence.py) | `run`, `main` |
| [audit_thirteen_robot_models.py](scripts/audit_thirteen_robot_models.py) | `canonical_chain_from_entry`, `audit_registry`, `main` |
| [benchmark_realtime_scaffold.py](scripts/benchmark_realtime_scaffold.py) | `main` |
| [benchmark_willow_trace.py](scripts/benchmark_willow_trace.py) | `quat_mul`, `quat_conjugate`, `quat_slerp`, `site_quaternion`, `percentile`, `load_trace`, `run_robot`, `main` |
| [build_droid_20min_trace.py](scripts/build_droid_20min_trace.py) | `path_for`, `fetch`, `duration`, `main` |
| [build_droid_5min_trace.py](scripts/build_droid_5min_trace.py) | `main` |
| [build_egodex_challenge_trace.py](scripts/build_egodex_challenge_trace.py) | `qangle`, `resample`, `main` |
| [build_fixed_vs_legacy_report.py](scripts/build_fixed_vs_legacy_report.py) | `load_legacy`, `load_fixed`, `write_csv`, `summaries`, `save`, `make_figures`, `build_pdf`, `main` |
| [build_mount_ik_fidelity_pilot_report.py](scripts/build_mount_ik_fidelity_pilot_report.py) | `validate_pilot_completeness`, `build_report_payload`, `write_report`, `main` |
| [build_official_model_provenance_gate.py](scripts/build_official_model_provenance_gate.py) | `build_gate`, `main` |
| [build_piperx_complete_follow_report.py](scripts/build_piperx_complete_follow_report.py) | `parse_args`, `build_report`, `main` |
| [build_piperx_controller_event_manifest.py](scripts/build_piperx_controller_event_manifest.py) | `validate_archive_summary`, `build_manifest`, `main` |
| [build_piperx_controller_event_v4_report.py](scripts/build_piperx_controller_event_v4_report.py) | `validate_report_scope`, `build`, `main` |
| [build_piperx_fixed_time_report.py](scripts/build_piperx_fixed_time_report.py) | `validate_report_manifest`, `build_report`, `main` |
| [build_piperx_literature_optimization_report.py](scripts/build_piperx_literature_optimization_report.py) | `build`, `main` |
| [build_piperx_multitask_fixed_time_bundle.py](scripts/build_piperx_multitask_fixed_time_bundle.py) | `build_shard_arrays`, `solve_selected_shard`, `aggregate_shard`, `validate_manifest`, `validate_bundle_artifacts`, `build_bundle`, `solve_shards`, `main` |
| [build_piperx_multitask_mount_report.py](scripts/build_piperx_multitask_mount_report.py) | `validate_report_manifest`, `build_report_claims`, `build_report`, `main` |
| [build_piperx_multitask_release_manifest.py](scripts/build_piperx_multitask_release_manifest.py) | `build_release_manifest`, `validate_release_manifest`, `main` |
| [build_piperx_recommended_v31_report.py](scripts/build_piperx_recommended_v31_report.py) | `parse_args`, `build_report`, `main` |
| [build_piperx_two_task_fixed_time_bundle.py](scripts/build_piperx_two_task_fixed_time_bundle.py) | `build_fixed_time_arrays`, `validate_fixed_time_arrays`, `build_task`, `validate_task`, `build_bundle`, `main` |
| [build_piperx_two_task_report.py](scripts/build_piperx_two_task_report.py) | `load_report_data`, `build_report`, `parse_args`, `main` |
| [build_strict_video_job_manifest.py](scripts/build_strict_video_job_manifest.py) | `task_names`, `main` |
| [build_ten_arm_three_episode_report.py](scripts/build_ten_arm_three_episode_report.py) | `provenance_for_report`, `load_rows`, `matrix`, `build_report`, `main` |
| [build_ten_arm_two_single_task_outputs.py](scripts/build_ten_arm_two_single_task_outputs.py) | `load_rows`, `build_data`, `figures`, `comparison_videos`, `report`, `main` |
| [build_twelve_arm_all_single_task_outputs.py](scripts/build_twelve_arm_all_single_task_outputs.py) | `load_rows`, `write_tables`, `save_figures`, `render_comparisons`, `build_report`, `main` |
| [calibrate_collision_capsules_cuda.py](scripts/calibrate_collision_capsules_cuda.py) | `rigid_alignment_rmse`, `classification_metrics`, `main` |
| [calibrate_piperx_tool_frames.py](scripts/calibrate_piperx_tool_frames.py) | `Context`, `calibrate`, `main` |
| [calibrate_thirteen_arm_collision_proxy.py](scripts/calibrate_thirteen_arm_collision_proxy.py) | `calibrate`, `main` |
| [clean_short_episodes.py](scripts/clean_short_episodes.py) | `EpisodeInfo`, `inspect_episode`, `scan_episodes`, `is_short`, `quarantine_path`, `clean`, `build_parser`, `main` |
| [compare_m0609_wrist_branches.py](scripts/compare_m0609_wrist_branches.py) | `wrist_flip`, `replay`, `main` |
| [derive_openarm_single_arm.py](scripts/derive_openarm_single_arm.py) | `derive` |
| [diagnose_piperx_complete_follow.py](scripts/diagnose_piperx_complete_follow.py) | `parse_args`, `run`, `main` |
| [diagnose_result.py](scripts/diagnose_result.py) | `main` |
| [download_egodex_pose_only.py](scripts/download_egodex_pose_only.py) | `HTTPRange`, `fetch_range`, `member_bytes`, `matrix_to_quat_wxyz`, `relative`, `pose_arrays`, `extract`, `consolidate`, `main` |
| [episode_3d_desktop.py](scripts/episode_3d_desktop.py) | `TcpSeries`, `Episode`, `quaternion_to_matrix`, `nearest_index`, `load_episode`, `EpisodeViewerWindow`, `build_parser`, `main` |
| [evaluate_mount_robustness_cuda.py](scripts/evaluate_mount_robustness_cuda.py) | `main` |
| [execute_piperx_seal_bag_orientation_shortlists.py](scripts/execute_piperx_seal_bag_orientation_shortlists.py) | `select_collision_free_attempt`, `attempt_stem`, `main` |
| [export_continuous_playback.py](scripts/export_continuous_playback.py) | `main` |
| [export_external_domain_samples.py](scripts/export_external_domain_samples.py) | `relative`, `evenly` |
| [export_formal_continuous_playback.py](scripts/export_formal_continuous_playback.py) | `finalists`, `transform_points`, `main` |
| [export_mount_roll_coverage.py](scripts/export_mount_roll_coverage.py) | `transform_points`, `main` |
| [formal_run_acceleration.py](scripts/formal_run_acceleration.py) | `input_fingerprint`, `search_fingerprint`, `solve_fingerprint`, `render_fingerprint`, `atomic_write_json`, `artifact_is_current`, `json_fingerprint_matches` |
| [generate_branch_oracle_dataset.py](scripts/generate_branch_oracle_dataset.py) | `main` |
| [high_quality_robot_scene.py](scripts/high_quality_robot_scene.py) | `align_robot_to_pedestal`, `base_mount_depth`, `decorate_high_quality_scene` |
| [inspect_result_collisions.py](scripts/inspect_result_collisions.py) | `main` |
| [local_refine_geometry_700_750.py](scripts/local_refine_geometry_700_750.py) | `main` |
| [migrate_piperx_fixed_time_artifacts_v2.py](scripts/migrate_piperx_fixed_time_artifacts_v2.py) | `migrate`, `main` |
| [mount_search_telemetry.py](scripts/mount_search_telemetry.py) | `StageTelemetry`, `SearchTelemetry` |
| [optimize_doosan_730_geometry.py](scripts/optimize_doosan_730_geometry.py) | `candidate_scene`, `evaluate`, `main` |
| [optimize_parametric_6r_droid.py](scripts/optimize_parametric_6r_droid.py) | `quat_rotation`, `sampled_tasks`, `seed_bank`, `follow_task`, `evaluate`, `main` |
| [optimize_piperx_two_task_follow.py](scripts/optimize_piperx_two_task_follow.py) | `MountCandidate`, `local_mount_candidates`, `append_experiment_record`, `pending_candidates`, `seal_bag_seed_candidates`, `probe_candidate`, `parse_args`, `run`, `main` |
| [optimize_universal_base_parametric.py](scripts/optimize_universal_base_parametric.py) | `rot`, `tasks`, `attempt`, `evaluate`, `main` |
| [plan_collision_free_ik_branches.py](scripts/plan_collision_free_ik_branches.py) | `main` |
| [plan_realtime_trace_ik_branches.py](scripts/plan_realtime_trace_ik_branches.py) | `main` |
| [plot_collision_classifier_comparison.py](scripts/plot_collision_classifier_comparison.py) | `main` |
| [plot_collision_proxy_audit.py](scripts/plot_collision_proxy_audit.py) | `main` |
| [plot_continuous_ablation.py](scripts/plot_continuous_ablation.py) | `load`, `main` |
| [plot_controller_gap.py](scripts/plot_controller_gap.py) | `main` |
| [plot_final_study.py](scripts/plot_final_study.py) | `main` |
| [plot_mount_roll_profiles.py](scripts/plot_mount_roll_profiles.py) | `main` |
| [plot_piperx_multitask_mount_results.py](scripts/plot_piperx_multitask_mount_results.py) | `coverage_matrix`, `winner_counts`, `generate_figures`, `main` |
| [plot_surrogate_ablation.py](scripts/plot_surrogate_ablation.py) | `rows`, `main` |
| [plot_ten_arm_two_task_results.py](scripts/plot_ten_arm_two_task_results.py) | `load_rows`, `export`, `write_source_data`, `robot_order`, `plot_ranking`, `plot_coverage_heatmap`, `plot_errors`, `plot_failures`, `plot_mounts`, `main` |
| [prepare_local_pose_benchmark.py](scripts/prepare_local_pose_benchmark.py) | `run`, `main` |
| [profile_piperx_fixed_time.py](scripts/profile_piperx_fixed_time.py) | `main` |
| [reaudit_805_two_task_caches.py](scripts/reaudit_805_two_task_caches.py) | `main` |
| [refine_fold_box_piperx_collision_free_mount.py](scripts/refine_fold_box_piperx_collision_free_mount.py) | `safe`, `main` |
| [refine_fold_box_piperx_mount.py](scripts/refine_fold_box_piperx_mount.py) | `main` |
| [refine_parametric_6r_droid.py](scripts/refine_parametric_6r_droid.py) | `main` |
| [refine_piperx_complete_follow_mount.py](scripts/refine_piperx_complete_follow_mount.py) | `run`, `main` |
| [refine_seal_bag_right_mount_shortlist.py](scripts/refine_seal_bag_right_mount_shortlist.py) | `main` |
| [refine_selected_failure_window_slp.py](scripts/refine_selected_failure_window_slp.py) | `main` |
| [render_all_arm_task_videos.py](scripts/render_all_arm_task_videos.py) | `slug`, `project` |
| [render_all_ten_arm_fixed_4096_videos.py](scripts/render_all_ten_arm_fixed_4096_videos.py) | `main` |
| [render_continuous_playback.py](scripts/render_continuous_playback.py) | `main` |
| [render_factory_dual_piperx_fixed_time.py](scripts/render_factory_dual_piperx_fixed_time.py) | `comparison_planning_indices`, `interpolate_joint_path`, `TaskSpec`, `task_spec`, `load_locked_piperx_calibration`, `fixed_time_timing_audit`, `validate_saved_trajectory`, `mount_separation_for_run`, `solve_fixed_time_motion`, `run_task`, `render_saved_run`, `main` |
| [render_factory_dual_piperx_fold_box.py](scripts/render_factory_dual_piperx_fold_box.py) | `main` |
| [render_factory_dual_xarm6_fold_box.py](scripts/render_factory_dual_xarm6_fold_box.py) | `SharedRetiming`, `AccelerationRetiming`, `bounded_smooth_pose_series`, `retime_for_joint_acceleration`, `shared_retimed_intervals`, `improvement_has_plateaued`, `shared_mount_height_candidates`, `synchronous_mount_score`, `map_source_quaternions_to_tcp`, `classify_ik_failure`, `pose_error`, `evaluate_joint_path`, `refine_fixed_time_bimanual_path`, `mapped_quaternions_at_joint_midpoint`, `solve`, `subset`, `solve_strict_single_arm_method`, `solve_multibranch_single_arm_method`, `audit_bimanual_collisions`, `failure_windows`, `run_fold_box`, `main` |
| [render_factory_dual_xarm6_follow.py](scripts/render_factory_dual_xarm6_follow.py) | `solve_position_follow`, `main` |
| [render_factory_dual_xarm6_insert_into_bottle.py](scripts/render_factory_dual_xarm6_insert_into_bottle.py) | `improvement_has_plateaued`, `shared_mount_height_candidates`, `synchronous_mount_score`, `map_source_quaternions_to_tcp`, `classify_ik_failure`, `pose_error`, `mapped_quaternions_at_joint_midpoint`, `solve`, `subset`, `solve_strict_single_arm_method`, `solve_multibranch_single_arm_method`, `audit_bimanual_collisions`, `enforce_paired_collision_safety`, `evaluate_realized_bimanual`, `failure_windows`, `main` |
| [render_factory_dual_xarm6_pour_raw_material.py](scripts/render_factory_dual_xarm6_pour_raw_material.py) | `improvement_has_plateaued`, `shared_mount_height_candidates`, `synchronous_mount_score`, `map_source_quaternions_to_tcp`, `classify_ik_failure`, `pose_error`, `mapped_quaternions_at_joint_midpoint`, `solve`, `subset`, `solve_strict_single_arm_method`, `solve_multibranch_single_arm_method`, `audit_bimanual_collisions`, `failure_windows`, `main` |
| [render_factory_dual_xarm6_se3_follow.py](scripts/render_factory_dual_xarm6_se3_follow.py) | `RunOptions`, `parse_run_options`, `output_path_for_robot`, `mount_selection_method`, `use_collision_safe_pair_planner`, `improvement_has_plateaued`, `shared_mount_height_candidates`, `synchronous_mount_score`, `map_source_quaternions_to_tcp`, `classify_ik_failure`, `pose_error`, `mapped_quaternions_at_joint_midpoint`, `prepare_follow_targets`, `solve`, `subset`, `solve_strict_single_arm_method`, `solve_multibranch_single_arm_method`, `solve_collision_safe_bimanual_method`, `audit_bimanual_collisions`, `failure_windows`, `main` |
| [render_fold_box_piperx_front_view.py](scripts/render_fold_box_piperx_front_view.py) | `front_azimuth_from_mount`, `main` |
| [render_model_assembly_qa.py](scripts/render_model_assembly_qa.py) | `main` |
| [render_piperx_multitask_mount_comparisons.py](scripts/render_piperx_multitask_mount_comparisons.py) | `render_panel`, `compose_four_panel`, `render_trajectory`, `main` |
| [render_result_frame.py](scripts/render_result_frame.py) | `main` |
| [render_seal_bag_front_view.py](scripts/render_seal_bag_front_view.py) | `report_directory_for_robot`, `main` |
| [render_strict_single_arm_task.py](scripts/render_strict_single_arm_task.py) | `playback_frame_count`, `interpolated_failure_reason`, `path_render_indices`, `interpolate_joint_positions`, `interpolate_quaternion_wxyz`, `font`, `sphere`, `connector`, `orientation_triad`, `configure_visual_only_render`, `main` |
| [render_thirteen_arm_mujoco_panels.py](scripts/render_thirteen_arm_mujoco_panels.py) | `vals`, `build` |
| [rerender_805_two_task_videos.py](scripts/rerender_805_two_task_videos.py) | `main` |
| [result_provenance.py](scripts/result_provenance.py) | `validate_provenance`, `aggregate_scores` |
| [retime_realtime_trace.py](scripts/retime_realtime_trace.py) | `main` |
| [rolling_multibranch_ik.py](scripts/rolling_multibranch_ik.py) | `BranchCandidate`, `BranchSelection`, `RecedingHorizonPath`, `RetimedPath`, `build_collision_free_pair_layers`, `select_minimum_retime_path`, `select_global_feasible_path`, `transition_limit_rad`, `select_rolling_branch`, `select_receding_horizon_path` |
| [run_805_native_strict_rerank.py](scripts/run_805_native_strict_rerank.py) | `main` |
| [run_805_two_task_strict_validation.py](scripts/run_805_two_task_strict_validation.py) | `main` |
| [run_doosan_realtime_mpc_50hz.py](scripts/run_doosan_realtime_mpc_50hz.py) | `tuned_profile` |
| [run_expanded_validation.py](scripts/run_expanded_validation.py) | `main` |
| [run_external_domain_dense_search.py](scripts/run_external_domain_dense_search.py) | `main` |
| [run_formal_resampling.py](scripts/run_formal_resampling.py) | `main` |
| [run_i2rt_cap_formal.py](scripts/run_i2rt_cap_formal.py) | `run`, `main` |
| [run_mount_ik_fidelity_pilot.py](scripts/run_mount_ik_fidelity_pilot.py) | `build_parser`, `stage_counts`, `pilot_fingerprint`, `stage_input_fingerprint`, `load_stage_records`, `coarse_fingerprint`, `coarse_algorithm_fingerprint`, `official_model_contract`, `coarse_model_contract`, `load_completed_robot_result`, `full_episode_optimism_metrics`, `main` |
| [run_piperx_controller_event_v4.py](scripts/run_piperx_controller_event_v4.py) | `validate_enforced_dynamics`, `validate_zero_safety_evidence`, `validate_acceptance_evidence`, `event_validation_spec`, `validate_event_shard`, `event_initializer_rows`, `solve_event_shard`, `main` |
| [run_piperx_factory_per_task_mount_search.py](scripts/run_piperx_factory_per_task_mount_search.py) | `ModeJob`, `atomic_json`, `plan_mode_jobs`, `evaluate_stage_resumable`, `run_mode`, `main` |
| [run_piperx_multitask_fixed_time_mount_study.py](scripts/run_piperx_multitask_fixed_time_mount_study.py) | `StudyJob`, `plan_jobs`, `family_representative_spec`, `rank_mount_result`, `atomic_json`, `study_status_path`, `prepare_family_follow_targets`, `run_job`, `run_study`, `write_dry_run`, `main` |
| [run_piperx_recommended_v31.py](scripts/run_piperx_recommended_v31.py) | `parse_args`, `candidate_rank_key`, `scene_mount_kwargs`, `resample_task_60hz`, `smooth_follow_targets`, `condition_complete_follow_targets`, `build_summary`, `build_complete_summary`, `run`, `main` |
| [run_piperx_seal_bag_orientation_comparison.py](scripts/run_piperx_seal_bag_orientation_comparison.py) | `run_mode`, `main` |
| [run_recommended_doosan_mpc.py](scripts/run_recommended_doosan_mpc.py) | `run`, `main` |
| [run_strict_cache_batch.py](scripts/run_strict_cache_batch.py) | `main` |
| [run_strict_mount_search_batch.py](scripts/run_strict_mount_search_batch.py) | `main` |
| [run_strict_render_batch.py](scripts/run_strict_render_batch.py) | `main` |
| [run_ten_arm_three_pick_episodes.py](scripts/run_ten_arm_three_pick_episodes.py) | `job_fingerprint`, `coarse_fingerprint`, `selected_episodes`, `coarse_result_complete`, `run_one`, `main` |
| [run_ten_arm_two_single_tasks.py](scripts/run_ten_arm_two_single_tasks.py) | `main` |
| [run_ten_arm_two_single_tasks_current_rerun.py](scripts/run_ten_arm_two_single_tasks_current_rerun.py) | `main` |
| [run_thirteen_arm_dense_search.py](scripts/run_thirteen_arm_dense_search.py) | `select_proxy_shortlist`, `load_official_search_templates`, `build_parser`, `load_search_samples`, `yaw_only_mount_space`, `expand_yaw_only_mounts`, `select_task_samples`, `select_robots`, `main` |
| [run_twelve_arm_all_single_tasks.py](scripts/run_twelve_arm_all_single_tasks.py) | `main` |
| [run_twelve_arm_stick_battery_formal.py](scripts/run_twelve_arm_stick_battery_formal.py) | `run`, `write_progress`, `main` |
| [run_twelve_arm_two_single_tasks.py](scripts/run_twelve_arm_two_single_tasks.py) | `build_parser`, `run`, `write_progress`, `main` |
| [run_two_arm_two_task_formal.py](scripts/run_two_arm_two_task_formal.py) | `run`, `main` |
| [run_ur5_realtime_mpc_50hz.py](scripts/run_ur5_realtime_mpc_50hz.py) | `tuned_profile` |
| [run_xarm6_realtime_mpc_50hz.py](scripts/run_xarm6_realtime_mpc_50hz.py) | `tuned_profile` |
| [scan_doosan_spacing.py](scripts/scan_doosan_spacing.py) | `scene_for`, `evaluate` |
| [scan_robot_installation.py](scripts/scan_robot_installation.py) | `make_scene`, `evaluate`, `main` |
| [search_fold_box_piperx_fixed_time_mount.py](scripts/search_fold_box_piperx_fixed_time_mount.py) | `synchronized_mount_candidates`, `select_full_audited_mount`, `main` |
| [search_fold_box_piperx_mount.py](scripts/search_fold_box_piperx_mount.py) | `layered_sample_indices`, `coarse_mount_candidates`, `rank_mount_candidate`, `rank_paired_mount_candidate`, `select_paired_mount`, `local_paired_refinements`, `collision_aware_pair_refinements`, `select_mount_pair`, `search`, `main` |
| [search_fold_box_piperx_noncrossing_mount.py](scripts/search_fold_box_piperx_noncrossing_mount.py) | `main` |
| [search_fold_box_piperx_paired_mount.py](scripts/search_fold_box_piperx_paired_mount.py) | `advance_connected_pairs`, `deterministic_pair_mounts`, `evaluate_pair`, `bounded_full_planning_indices`, `evaluate_full_pair`, `main` |
| [search_piperx_fixed_time_mounts.py](scripts/search_piperx_fixed_time_mounts.py) | `xarm6_style_mount_candidates`, `local_mount_candidates`, `select_stage_finalists`, `screen_task`, `refine_task`, `refine_again_task`, `validate_task_finalists`, `main` |
| [search_seal_bag_right_mount.py](scripts/search_seal_bag_right_mount.py) | `visual_wrist_flip_mask`, `exhaustive_tabletop_right_mounts`, `longest_failure_run`, `whole_trajectory_window_indices`, `generate_right_mount_candidates`, `right_mount_score`, `main` |
| [search_strict_urdf_mount.py](scripts/search_strict_urdf_mount.py) | `build_parser`, `resolved_best_first_budget`, `resolved_hierarchy_budget`, `registered_targets`, `main` |
| [select_safe_pareto_candidate.py](scripts/select_safe_pareto_candidate.py) | `main` |
| [smoke_doosan_realtime_mpc.py](scripts/smoke_doosan_realtime_mpc.py) | `main` |
| [smoke_gpu_design.py](scripts/smoke_gpu_design.py) | `main` |
| [smoke_gpu_ik.py](scripts/smoke_gpu_ik.py) | `main` |
| [smoke_xarm6_ur5_realtime_mpc.py](scripts/smoke_xarm6_ur5_realtime_mpc.py) | `run_robot`, `main` |
| [solve_strict_urdf_task_cache.py](scripts/solve_strict_urdf_task_cache.py) | `vendor_allowed_collision_pairs`, `target_poses_for`, `targets_for`, `optimization`, `build_model`, `MountModelTemplate`, `build_mount_model_template`, `active_ancestor_map`, `active_chain_distance`, `hold_invalid_frames`, `evaluate_q_path`, `collision_flags`, `main` |
| [status_learning_runs.py](scripts/status_learning_runs.py) | `last_matching`, `main` |
| [strict_mujoco_ik.py](scripts/strict_mujoco_ik.py) | `joint_periodic_mask`, `wrapped_joint_delta`, `continuity_jump`, `continuity_recovery_step`, `realized_velocity_violation`, `joint_discontinuity_from_recovery_modes`, `PositionPathResult`, `PosePathResult`, `generate_pose_candidate_layers`, `solve_pose_path_layered`, `solve_pose_path`, `solve_pose_path_multibranch`, `solve_position_path` |
| [strict_mujoco_model.py](scripts/strict_mujoco_model.py) | `active_chain_length_m`, `sampled_maximum_tcp_reach_m` |
| [strict_trajectory_sources.py](scripts/strict_trajectory_sources.py) | `RelativeTrajectory`, `resample_trajectory`, `place_relative_positions`, `minimum_safe_anchor_z`, `local_task_anchor_z`, `task_anchor_rotations`, `load_relative_task_trajectory` |
| [strict_urdf_model_audit.py](scripts/strict_urdf_model_audit.py) | `ModelEntry`, `load_native_spec`, `qualify`, `main` |
| [summarize_collision_formal.py](scripts/summarize_collision_formal.py) | `main` |
| [summarize_formal_benchmark.py](scripts/summarize_formal_benchmark.py) | `collect`, `main` |
| [summarize_joint_geometry_base.py](scripts/summarize_joint_geometry_base.py) | `result_path`, `main` |
| [summarize_pareto_runs.py](scripts/summarize_pareto_runs.py) | `main` |
| [summarize_recommended_velocity.py](scripts/summarize_recommended_velocity.py) | `main` |
| [summarize_task_level_topology.py](scripts/summarize_task_level_topology.py) | `main` |
| [sweep_mount_roll_cuda.py](scripts/sweep_mount_roll_cuda.py) | `main` |
| [train_branch_proposal_policy.py](scripts/train_branch_proposal_policy.py) | `main` |
| [train_collision_classifier_cuda.py](scripts/train_collision_classifier_cuda.py) | `main` |
| [train_surrogate_from_archive.py](scripts/train_surrogate_from_archive.py) | `main` |
| [tune_robot_mpc.py](scripts/tune_robot_mpc.py) | `evaluate`, `main` |
| [validate_branch_plan.py](scripts/validate_branch_plan.py) | `main` |
| [validate_continuous_egodex_cuda.py](scripts/validate_continuous_egodex_cuda.py) | `select_challenge_episode`, `richest_contiguous_window`, `main` |
| [validate_continuous_pareto_candidate_cuda.py](scripts/validate_continuous_pareto_candidate_cuda.py) | `main` |
| [validate_doosan_730_candidate.py](scripts/validate_doosan_730_candidate.py) | `main` |
| [validate_egodex_geometry_cuda.py](scripts/validate_egodex_geometry_cuda.py) | `main` |
| [validate_episode_pointwise_candidate_cuda.py](scripts/validate_episode_pointwise_candidate_cuda.py) | `main` |
| [validate_external_domain_test.py](scripts/validate_external_domain_test.py) | `main` |
| [validate_parametric_6r_droid.py](scripts/validate_parametric_6r_droid.py) | `main` |
| [validate_pareto_candidate_cuda.py](scripts/validate_pareto_candidate_cuda.py) | `main` |
| [validate_piperx_two_task_bundle.py](scripts/validate_piperx_two_task_bundle.py) | `validate_task_bundle`, `validate_bundle`, `write_outputs`, `parse_args`, `main` |
| [validate_thirteen_arm_test.py](scripts/validate_thirteen_arm_test.py) | `base_matrix` |
| [video_timeline.py](scripts/video_timeline.py) | `validation_timeline`, `showcase_timeline` |
| [view_mount_comparison.py](scripts/view_mount_comparison.py) | `main` |
| [watch_piperx_factory_per_task_mount_search.py](scripts/watch_piperx_factory_per_task_mount_search.py) | `status_payload`, `should_restart`, `main` |

</details>
