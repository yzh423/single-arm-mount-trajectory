# single-arm-mount

This repository turns recorded robot end-effector trajectories into reproducible mount-and-IK studies. Choose a task dataset and robot model, run the corresponding search or fixed-time study, then inspect validated shards, aggregate metrics, figures, videos, and reports under [`reports/`](reports/). The primary workflow is the PiperX dual-hand multi-task study: both hands are solved as one synchronized system across four physical mount configurations, and every source timestamp is preserved without retiming.

The central mental model is an evidence pipeline rather than a single solver call. Source validation and registration define the targets. Per-task mount search and branch-aware IK propose solutions. Frame collision, swept-edge collision, and mount-topology checks reject unsafe states. Bundle validation proves that every expected trajectory and mount cell is present and traceable. Plot, render, and report scripts consume the validated evidence. General single-arm design-optimization tools share this repository as a parallel workflow.

```powershell
python -m scripts.build_piperx_multitask_fixed_time_bundle --validate-only
```

## Choose a workflow

Start with the PiperX workflow when the input contains synchronized left- and right-hand trajectories and the original timing must remain fixed. Use the single-arm workflow when comparing robot models, installation positions, geometry, Pareto trade-offs, or robustness for one arm at a time. This separation exists because paired Fixed-time evidence and single-arm design exploration answer different questions and must not share success criteria.

### PiperX dual-hand multi-task Fixed-time study

When both recorded hands must be evaluated as one synchronized system without changing task timing, use the published PiperX Fixed-time experiment. It covers 27 dual-hand trajectories from 12 task families and compares four mount configurations in a fixed order: `baseline`, `upright_table`, `horizontal_wall`, and `inverted`.

The study contract is fixed at 1 mm position error and 0.5 degree orientation error against the published conditioned TCP targets. Before IK, the recorded TCP stream receives a deterministic 9-frame calibration/conditioning pass bounded to at most 5 mm translation and 1 degree orientation change from the recorded target. This conditioning suppresses capture noise; it is not retiming and does not add, remove, reorder, or shift timestamps. A frame counts as successful only when both hands pass both strict pose limits at the same source timestamp. Retiming is disabled, and the solved timeline must equal the exact relative source schedule after subtracting its first timestamp to use a zero origin.

Safety is not a ranking preference. Every published shard must satisfy all three hard gates:

- `collision_frames == 0`
- `edge_collision_frames == 0`
- `topology_invalid_frames == 0`

The current validated bundle contains all 108 trajectory and mount cells. The baseline and upright-table configurations have substantially higher average strict coverage than the wall and inverted configurations. This is a measured workspace and orientation limitation under the exact timing and pose constraints, not a missing-data artifact. See [Limitations](#limitations) before interpreting coverage as task-level deployability.

Published evidence:

- [Final 38-page PDF report](reports/piperx_multitask_fixed_time_mount_study/PiperX多任务Fixed-Time四构型对比报告.pdf)
- [Aggregate metrics for all 108 cells](reports/piperx_multitask_fixed_time_mount_study/aggregate.csv)
- [Validated bundle manifest](reports/piperx_multitask_fixed_time_mount_study/bundle_manifest.json)
- [Content-addressed release manifest](reports/piperx_multitask_fixed_time_mount_study/release_manifest.json)
- [Comparison figures](reports/piperx_multitask_fixed_time_mount_study/figures/)
- [All 27 synchronized four-mount MuJoCo comparison videos](reports/piperx_multitask_fixed_time_mount_study/videos/comparisons/)

The trajectory `8-12/PourRawMaterial/111542` is the running example in this README. Its upright-table result accepts both hands on 1918 of 5942 original frames, or 32.28%, while all four of its mount cells record zero frame collisions, zero swept-edge collisions, and zero topology-invalid frames. Its [four-mount Fixed-time comparison video](reports/piperx_multitask_fixed_time_mount_study/videos/comparisons/8-12_PourRawMaterial_111542_four_mount_fixed_time.mp4) uses the same source schedule in all four panels.

### Run or resume the PiperX study

When reproducing or extending the published experiment, begin from the versioned parameter contract rather than reconstructing settings from command history. The recommended configuration is [`configs/piperx_recommended_v31.json`](configs/piperx_recommended_v31.json); it stores calibrated family-specific tool frames, mount poses, solver budgets, and bounded wrist adaptations.

Use Python 3.11. Install `requirements.txt` for the Fixed-time study, report, and tests; install `requirements-full.txt` when running the optional repository-wide single-arm, conversion, and desktop UI tools.

```powershell
python -m pip install -r requirements.txt
# Optional full workspace:
python -m pip install -r requirements-full.txt
```

```powershell
# Discover the experiment matrix without solving it
python -m scripts.run_piperx_multitask_fixed_time_mount_study --dry-run

# Run or resume per-trajectory mount search
python -m scripts.run_piperx_multitask_fixed_time_mount_study

# Build formal shards, aggregate, and validate the complete bundle
python -m scripts.build_piperx_multitask_fixed_time_bundle
python -m scripts.build_piperx_multitask_fixed_time_bundle --validate-only

# Verify the published PDF, figures, videos, provenance, and implementation hashes
python -m scripts.build_piperx_multitask_release_manifest --validate-only
```

Rebuild figures, real MuJoCo comparison videos, and the PDF from validated evidence:

```powershell
python -m scripts.plot_piperx_multitask_mount_results
python -m scripts.render_piperx_multitask_mount_comparisons
python -m scripts.build_piperx_multitask_mount_report
python -m scripts.build_piperx_multitask_release_manifest
```

Use `--trajectory` and `--mode` on the study and bundle scripts for a focused investigation. Focused output is useful for debugging, but it is not a replacement for validating the complete 27 by 4 publication bundle.

### General single-arm design optimization

When the question concerns one arm's robot choice, geometry, installation, or performance trade-offs instead of synchronized two-hand following, use the repository-wide single-arm workflow under [`design_optimization/`](design_optimization/). It provides robot registries, URDF kinematics, mount and geometry search, hierarchical and best-first search, Pareto scoring, robustness checks, collision classification, trajectory-following metrics, real-time evaluation, and optional retiming experiments.

When building a single-arm search around audited robot geometry, import the package-level public API instead of depending on topology internals. `load_templates` reads the robot audit into named `TopologyTemplate` values containing joint axes, link deltas, home rotation, joint bounds, and audited tool metadata; `DesignBatch` carries a topology-preserving batch of candidate geometries that share one template. These three names are the complete `design_optimization.__all__` surface and are loaded lazily so lightweight imports do not initialize the Torch runtime until they are used.

Single-arm executable entry points live under [`scripts/`](scripts/). Examples include:

- [`scripts/run_mount_ik_fidelity_pilot.py`](scripts/run_mount_ik_fidelity_pilot.py), a mount and IK fidelity comparison
- [`scripts/run_ten_arm_two_single_tasks.py`](scripts/run_ten_arm_two_single_tasks.py), a multi-robot task study
- [`scripts/run_twelve_arm_all_single_tasks.py`](scripts/run_twelve_arm_all_single_tasks.py), a broader single-arm matrix
- [`scripts/search_strict_urdf_mount.py`](scripts/search_strict_urdf_mount.py), strict URDF-based mount search
- [`scripts/render_strict_single_arm_task.py`](scripts/render_strict_single_arm_task.py), single-arm result rendering

The retained single-arm publication area is [`reports/single_arm/`](reports/single_arm/). Useful starting points include the [mount and IK fidelity report](reports/single_arm/mount_ik_fidelity_pilot/pdf/mount_ik_fidelity_pilot_report.pdf), its [OpenArm video](reports/single_arm/mount_ik_fidelity_pilot/videos/openarm.mp4), its [xArm6 video](reports/single_arm/mount_ik_fidelity_pilot/videos/xarm6.mp4), and the [ten-arm three-episode report](reports/single_arm/ten_arm_pick_right_left_three_episodes/report.pdf).

## Prepare trustworthy trajectory data

Optimization begins only after model identity, timestamps, and coordinate frames are explicit. The factory pipeline rejects malformed input because silent timing repair would invalidate every later Fixed-time comparison.

### Load synchronized dual-hand data

Before registration or IK can be trusted, both hands and their shared schedule must enter the pipeline as one validated value. `FactoryBimanualTask` and `load_factory_task` in [`factory_bimanual/source_data.py`](factory_bimanual/source_data.py) load left and right positions, quaternions, and source timestamps. Timestamps must be finite and strictly increasing. The loader verifies pose-array shape and can repair explicitly supported invalid pose rows without changing the source schedule.

For `8-12/PourRawMaterial/111542`, this stage produces one synchronized `FactoryBimanualTask` with 5942 source rows. Both hand targets and their shared timestamps remain paired as the same running value through registration, mount comparison, IK, and evidence generation.

> **Caution:** Fixed-time is an identity contract. Equal duration is insufficient. `retiming_applied` must be false, and every solved timestamp must equal its source timestamp.

### Discover the complete task matrix

When a result is meant to represent the full dataset rather than a hand-picked recording, discovery must make both inclusion and exclusion explicit. `FactoryEpisode`, `TaskRepresentative`, and `FactoryTaskCatalog` discover eligible recordings and record exclusions. `discover_dual_hand_trajectories` turns that catalog into the formal `TrajectorySpec` matrix used by the multi-task study.

The manifest is the publication authority for coverage. A complete PiperX bundle requires 27 trajectories, four modes per trajectory, and 108 unique cells. Missing, duplicated, stale, or mismatched cells fail validation.

### Register source coordinates

Before comparing mounts, every recording must describe targets in the same physical frame and under the correct task-family tool convention. Registration utilities in [`factory_bimanual/registration.py`](factory_bimanual/registration.py) map source coordinates into the shared factory and world frames. Task-family utilities then apply configured tool translations, tool rotations, and target conventions. Source data, registration, and target contract are included in cache fingerprints so changed tool frames cannot reuse stale search results.

For the running PourRawMaterial trajectory, family-specific tool rotations are part of that fingerprint. This means the 32.28% upright-table result cannot be loaded from a cache created under an older wrist or target-axis convention.

### Use authoritative robot geometry

When reachability, IK, and collision claims depend on link dimensions and joint limits, the solver and renderer must share authoritative robot geometry. `RobotRegistry`, robot contracts, `URDFChain`, and kinematics utilities under [`design_optimization/`](design_optimization/) define robot identity for single-arm studies. PiperX formal solving and rendering use native MuJoCo model adapters under [`factory_bimanual/`](factory_bimanual/).

Official and vendored robot assets are retained under [`third_party/official_robot_models/`](third_party/official_robot_models/). Model provenance and licenses remain with those assets. Do not substitute display meshes or approximate link geometry for formal collision evidence.

## Compare installation configurations

Mount comparison combines calibrated task-family defaults with per-trajectory search because one family-wide pose does not place every recording in the same reachable and collision-free workspace. Unlike score-only selection, the final choice excludes incomplete safety evidence before comparing IK coverage.

### Resolve calibrated family-specific mounts

When recordings from different task families use different tool directions or workspace regions, resolve their calibrated defaults before starting per-trajectory search. `PiperXRecommendedConfig`, `RecommendedMountSpec`, and `WorldMount` in [`factory_bimanual/piperx_recommended.py`](factory_bimanual/piperx_recommended.py) map each family to tool frames, mount placement, solver budgets, and optional wrist adaptation. The configuration is versioned so a result can be traced to the exact target and installation contract.

### Construct physical mount orientations

When comparing table, wall, and inverted installations, change the physical base orientation while holding the recorded task schedule constant. `mount_quaternions` and the mount-mode constants construct upright-table, horizontal-wall, inverted, and supported forward orientations. Each mode changes the robot base frame, not the trajectory timebase.

> **Caution:** The older synchronized-mount validator assumes upright bases. It cannot validate wall or inverted study modes. The multi-task workflow uses mount-aware topology checks for every mode.

### Search and select safe layouts

When family defaults do not place a particular recording in a reachable and collision-free region, search that trajectory's layout without weakening the final safety contract. `PerTaskSearchConfig`, `candidate_fingerprint`, `rank_full_finalist`, and `select_safe_layout` in [`factory_bimanual/per_task_mount_search.py`](factory_bimanual/per_task_mount_search.py) manage staged search and caching. Orientation, spacing, workspace, coarse, dense, and local refinement modules progressively reduce the candidate set before formal audit.

Only fully audited candidates are eligible for final selection. Selection raises when no collision-free, topology-valid layout exists. Incomplete evidence cannot be treated as a successful mount merely because its sparse IK score is high.

This distinction is visible on `8-12/PourRawMaterial/111542`: upright-table reaches 32.28% strict both-hand coverage, horizontal-wall reaches 4.98%, and baseline and inverted reach 0%. The modes retain the same 5942 deterministically conditioned targets and original timestamps, so the difference measures installation and IK capability rather than a different playback schedule.

## Follow both hands at fixed source time

Dual-hand following is a paired path problem because a left-arm solution is not valid evidence unless the right arm succeeds safely at the same source timestamp. Solving each arm independently can create branch discontinuities, cross-arm collisions, or visually unsynchronized behavior.

### Apply the study contract

Before any solver result can enter the publication bundle, it must be evaluated against one immutable experiment matrix and tolerance contract. `TrajectorySpec`, `StudyConfig`, and `STUDY_MODES` in [`factory_bimanual/multitask_fixed_time_study.py`](factory_bimanual/multitask_fixed_time_study.py) define that matrix. The published mode order is fixed, and the position and orientation tolerances are locked to 0.001 m and 0.5 degree against the conditioned targets. Every shard records the conditioning contract and its bounded deviation from the raw recorded TCP stream so the distinction remains auditable.

### Solve paired branch-aware IK

When both arms admit multiple IK branches, choose branches jointly so a locally valid single-arm posture cannot break bimanual continuity or safety. `IKCandidate`, `BimanualIKConfig`, `BimanualIKResult`, and `solve_strict_bimanual_path` in [`factory_bimanual/strict_bimanual_ik.py`](factory_bimanual/strict_bimanual_ik.py) search paired branches while enforcing continuity, joint limits, velocity constraints, collision checks, and swept-transition constraints.

Failure labels keep branch loss, state collision, transition collision, pose rejection, and dynamics separate. A rejected frame is not automatically an IK convergence failure.

On the running PourRawMaterial trajectory, the upright-table shard records 1926 left-accepted frames and 1919 right-accepted frames, but only the 1918 simultaneous frames count. This comparison shows why the published 32.28% metric is paired coverage rather than the average or union of two single-arm scores.

### Refine and recover without changing time

When an exact next-frame connection fails, recovery may preserve a safe executable path, but it must not alter the timestamp or relabel the missed target as success. The fixed-time refinement, complete-follow, collision-safe-follow, and rescue modules therefore attempt exact connections first, then bounded safe recovery. A safe hold can preserve collision safety when the next target is unreachable, but the held frame remains a strict-follow failure.

> **Caution:** Missing IK must not be relabeled as collision-free success when the controller holds the last safe posture.

> **Note:** Bounded orientation adaptation is attempted only after an exact orientation connection fails. Adapted tracking is reported separately and is not exact tracking.

### Report dynamics separately

When pose accuracy passes but the recorded schedule demands excessive joint motion, the result must expose a dynamics failure rather than slow the trajectory. Native-DOF and controller adapters produce executable joint-space trajectories and compute velocity and acceleration evidence on the unchanged source schedule. Dynamic feasibility is reported independently from pose coverage.

## Enforce safety as a hard gate

Formal safety combines geometric collision and installation topology because clear link geometry can still violate the intended left/right mounting relationship. Unlike a post-hoc penalty, either failure rejects initialization, rescue, and recovery before coverage can improve the candidate's rank.

### Classify frame and transition collisions

When endpoint postures appear clear, the motion between them can still cross robot or environment geometry, so both states and transitions require explicit evidence. `CollisionClass`, `CollisionReport`, and `CallbackCollisionChecker` classify self-collision, cross-arm collision, table collision, base collision, and swept-transition collision. MuJoCo adapters audit native PiperX geometry at every solved frame and at interpolated transition samples.

Frame collision and swept-edge collision are stored separately. This identifies trajectories whose endpoint postures are clear but whose transition passes through geometry.

### Enforce mount topology

When collision geometry is clear but the arms cross their intended sides or the bases violate the installation relationship, the candidate is still invalid. `MountTopologyConfig`, `MountTopologyReport`, and `MuJoCoMountTopologyChecker` in [`factory_bimanual/mount_topology.py`](factory_bimanual/mount_topology.py) enforce intended left/right base ordering and arm-side relationship. Initialization, paired rescue, and recovery all use the conjunction of collision and topology checks.

> **Caution:** A candidate pair must pass both MuJoCo collision checking and mount-topology checking. Passing either checker alone is insufficient.

All four `8-12/PourRawMaterial/111542` cells pass this conjunction: each reports `collision_frames == 0`, `edge_collision_frames == 0`, and `topology_invalid_frames == 0`. The upright-table coverage gain is therefore not purchased by accepting the slight collisions that a frame-only visual inspection can miss.

## Build auditable evidence

Each trajectory and mount cell becomes a formal shard so that aggregate coverage, safety claims, and rendered media can be recomputed from per-frame evidence instead of copied from logs or screenshots.

### Validate per-frame shards

When a summary number must remain independently auditable, retain the per-frame arrays from which it is computed. `build_shard_arrays`, `solve_selected_shard`, `aggregate_shard`, and `validate_shard` in [`scripts/build_piperx_multitask_fixed_time_bundle.py`](scripts/build_piperx_multitask_fixed_time_bundle.py) store source time, joint state, both-hand pose errors, strict acceptance, collision flags, topology validity, velocity, acceleration, and failure reasons.

Summary metrics are recomputed from these arrays. Raw left/right TCP validity masks are retained; invalid source poses are forced to reject and are disclosed separately from the valid-source denominator. Reports do not infer success from prose, screenshots, or video overlays.

For `8-12/PourRawMaterial/111542`, four shards preserve the same 5942-frame source schedule. The upright-table arrays contain the 1918 paired acceptances used to compute 32.28%, while the collision and topology arrays prove the three zero-count safety claims for that cell.

### Reject stale caches

When source data, tool conventions, mounts, robot geometry, collision settings, or solver behavior changes, reusing an older result would make the evidence internally inconsistent. Search checkpoints and candidates therefore include exact source hashes, budgets, mount parameters, tool-frame contracts, safety settings, and implementation protocols in their fingerprints. Formal summaries additionally bind the selected mount, source prefix, robot URDF hash, scene settings, and strict solver contract. Matching only a schema or protocol name is insufficient.

### Assemble and validate the bundle

When results are ready for comparison or publication, validate completeness and provenance across the entire matrix rather than trusting whichever shards happen to exist. `build_bundle`, `solve_shards`, `validate_manifest`, and `validate_bundle_artifacts` verify the complete 27 by 4 matrix. [`bundle_manifest.json`](reports/piperx_multitask_fixed_time_mount_study/bundle_manifest.json) records formal evidence paths and validation state.

`RunArtifactPaths` and `RunArtifactWriter` under [`factory_bimanual/artifacts.py`](factory_bimanual/artifacts.py) constrain output locations and write structured run artifacts atomically, reducing the risk that an interrupted run appears complete.

## Inspect and publish results

Plots, videos, and reports are downstream views of the validated bundle so that presentation cannot alter the experiment that produced the numbers. Unlike independent video editing, every comparison panel is indexed from the formal source schedule and acceptance arrays.

### Read the aggregate and figures

When comparing task families and mount modes across all 108 cells, start from the validated aggregate rather than reading values from video overlays. [`aggregate.csv`](reports/piperx_multitask_fixed_time_mount_study/aggregate.csv) is the compact comparison table. The figure directory contains:

- [strict both-hand coverage](reports/piperx_multitask_fixed_time_mount_study/figures/coverage_heatmap.png)
- [collision and topology audit](reports/piperx_multitask_fixed_time_mount_study/figures/collision_topology_heatmap.png)
- [maximum accepted-frame position error](reports/piperx_multitask_fixed_time_mount_study/figures/maximum_position_error_heatmap.png)
- [maximum accepted-frame orientation error](reports/piperx_multitask_fixed_time_mount_study/figures/maximum_orientation_error_heatmap.png)
- [longest hold ratio](reports/piperx_multitask_fixed_time_mount_study/figures/longest_hold_heatmap.png)
- [velocity evidence](reports/piperx_multitask_fixed_time_mount_study/figures/velocity_heatmap.png)
- [acceleration evidence](reports/piperx_multitask_fixed_time_mount_study/figures/acceleration_heatmap.png)
- [coverage, dynamics, and safety deployability gate](reports/piperx_multitask_fixed_time_mount_study/figures/deployability_heatmap.png)
- [winning-mount counts](reports/piperx_multitask_fixed_time_mount_study/figures/winner_counts.png)

### Compare synchronized videos

When visually comparing installations, every panel must represent the same source instant so apparent synchronization cannot be created during rendering. `comparison_timeline` and `source_frame_indices` in [`factory_bimanual/mount_comparison_visuals.py`](factory_bimanual/mount_comparison_visuals.py) enforce one common four-panel schedule. The renderer loads the validated MuJoCo scene and formal joint arrays for each panel.

> **Note:** The comparison renderer does not shorten or retime a mount mode. It samples each formal result onto one common 30 fps visualization timeline; all four panels still use the same zero-origin relative source schedule and duration.

The [comparison-video directory](reports/piperx_multitask_fixed_time_mount_study/videos/comparisons/) contains one MP4 for each of the 27 trajectories. Each filename includes the task family and take.

The running example is published as [`8-12_PourRawMaterial_111542_four_mount_fixed_time.mp4`](reports/piperx_multitask_fixed_time_mount_study/videos/comparisons/8-12_PourRawMaterial_111542_four_mount_fixed_time.mp4). Its four panels provide a visual comparison of the same baseline 0%, upright-table 32.28%, horizontal-wall 4.98%, and inverted 0% formal results without shortening any mode.

### Rebuild the PDF

When producing the final report, require the same complete, non-retimed evidence contract used by the plots and videos. [`scripts/build_piperx_multitask_mount_report.py`](scripts/build_piperx_multitask_mount_report.py) enforces that manifest requirement. The final [PiperX multi-task four-mount report](reports/piperx_multitask_fixed_time_mount_study/PiperX多任务Fixed-Time四构型对比报告.pdf) contains the experiment contract, task-level tables, figures, safety audit, and synchronized visual comparisons.

## Extend single-arm optimization

The single-arm surfaces are useful when the research question is robot selection or mount optimization rather than synchronized dual-hand Fixed-time following.

### Search at increasing fidelity

When the single-arm candidate space is too large for strict native-model evaluation everywhere, narrow it in stages of increasing fidelity. `InstallationSearchSpace` and the mount-sweep, best-first, hierarchical, batched, incremental, fidelity-funnel, and SLP refinement modules explore candidates at increasing computational cost. Coarse reachability narrows the space before strict native-model evaluation.

### Keep evaluation dimensions separate

When one candidate improves reach but worsens safety, dynamics, robustness, or latency, a single blended score can hide the trade-off. Task-space, objective, Pareto, robustness, collision, episode-follow, real-time, and retiming modules therefore score these properties separately.

### Inspect single-arm artifacts

When numerical metrics need visual diagnosis or a reproducible presentation artifact, inspect the same single-arm result through the episode viewers and export scripts. They provide browser and desktop 3D inspection, montage generation, and reproducible video bundles. Retained examples and reports remain discoverable under [`reports/single_arm/`](reports/single_arm/), while official model assets remain under [`third_party/official_robot_models/`](third_party/official_robot_models/).

## Limitations

Strict coverage is intentionally narrow. It requires both hands to meet 1 mm and 0.5 degree simultaneously at the original source timestamp. Safe holds, bounded wrist adaptation, orientation relaxation, collision-free motion, and successful rendering do not convert a rejected target into strict success.

Wall and inverted mounts have low or zero strict coverage on many trajectories because their reachable workspace and orientation branches differ from the recorded task geometry. The bundle keeps these cells and their failure evidence instead of dropping them. The report compares installation capability; it does not claim complete following for every task and mount.

The published collision result applies to the checked native MuJoCo geometry, transition sampling, clearance settings, and mount-topology contract. It does not replace hardware commissioning, calibration, torque limits, environmental collision checking, or emergency-stop validation.

Dynamic failures are not repaired by retiming in this study. Velocity and acceleration results describe execution at the recorded schedule. Use a separate retiming experiment only when changing task timing is acceptable, and do not compare it as Fixed-time evidence.

Under the joint publication gate of 100% strict both-hand coverage, PiperX velocity/acceleration limits, and zero safety violations, the current result is 0 of 108 cells. Coverage winners in the report therefore describe relative fixed-time pose-following capability, not deployment readiness.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`factory_bimanual/`](factory_bimanual/) | Dual-hand data, mount search, paired IK, collision, topology, artifacts, and visualization contracts |
| [`design_optimization/`](design_optimization/) | General single-arm robot, kinematics, search, evaluation, Pareto, robustness, and visualization tools |
| [`scripts/`](scripts/) | Reproducible experiment, validation, plotting, rendering, and report entry points |
| [`configs/`](configs/) | Versioned robot, task-family, mount, tool-frame, and solver parameters |
| [`reports/piperx_multitask_fixed_time_mount_study/`](reports/piperx_multitask_fixed_time_mount_study/) | Current 27-trajectory PiperX four-mount evidence bundle |
| [`reports/single_arm/`](reports/single_arm/) | Retained single-arm reports, metrics, figures, and selected videos |
| [`third_party/official_robot_models/`](third_party/official_robot_models/) | Vendored official robot models and provenance |
| [`tests/`](tests/) | Unit and integration tests for contracts, search, IK, safety, bundle validation, and media |

## Publication boundary

The GitHub repository includes all 27 formal dual-hand source CSVs, the generated multi-task report, validated evidence, 27 comparison videos and their provenance sidecars, retained single-arm reports and videos, and all required `third_party` models. Licenses and provenance in third-party model directories must remain intact; unresolved vendor redistribution status is disclosed in [`third_party/official_robot_models/NOTICE.md`](third_party/official_robot_models/NOTICE.md) rather than guessed.

The following four reference PDFs are local input material only and are intentionally excluded from GitHub. They are not published artifacts and are not linked from the repository:

- `PiperX双任务严格完全跟随实验报告.pdf`
- `单轨迹优化分析.pdf`
- `Seal_Bag 轨迹上 PiperX 双臂三种安装构型的跟随能力对比 - 飞书云文档.pdf`
- `双臂IK跟随方案四臂四种安装位姿报告.pdf`

Keep these files out of commits and release bundles. Their conclusions may inform implementation choices, but the repository's published claims must remain traceable to the validated shards, aggregate, manifest, figures, videos, and final multi-task report listed above.
