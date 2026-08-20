# Fold Box Dual PiperX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PiperX to the factory-bimanual robot contracts, independently select two Fold Box mounts, and render a dynamically bounded full SE(3) follow video.

**Architecture:** Extend the immutable robot contract and parameterize the existing Fold Box solver by contract instead of duplicating it. A dedicated deterministic mount-search script produces a JSON winner consumed by a thin PiperX rendering wrapper.

**Tech Stack:** Python, NumPy, SciPy, MuJoCo, OpenCV, pytest.

## Global Constraints

- Native PiperX six-joint limits and 3.0 rad/s velocity limits are authoritative.
- TCP is `gripper_base` + 0.13 m along local +Z.
- Both mounts are upright, share base Z, and base Z is no higher than 0.87 m.
- Pose smoothing is capped at 3 mm and 1°; IK tolerances are 1 mm and 1.5°.
- Joint acceleration is limited to 60 rad/s²; jerk hard limits remain disabled.
- Existing xArm6 Fold Box outputs are preserved.
- This workspace has no Git metadata; each tested increment is a local save point rather than a commit.

---

### Task 1: PiperX factory-bimanual contract

**Files:**
- Modify: `factory_bimanual/robot_contracts.py`
- Modify: `factory_bimanual/controller_profiles.json`
- Modify: `tests/factory_bimanual/test_robot_contracts.py`
- Modify: `tests/factory_bimanual/test_scene_builder.py`

**Interfaces:**
- Produces: `ROBOT_CONTRACTS["piperx"]` with six joints, `base_link`, `gripper_base`, native limits, and `(0,0,.13)` TCP offset.

- [ ] Write failing tests for contract order, limits, TCP parent/offset, scene compilation, and a six-value 3.0 rad/s controller profile.
- [ ] Run `python -m pytest tests/factory_bimanual/test_robot_contracts.py tests/factory_bimanual/test_scene_builder.py -q` and confirm failure because PiperX is absent.
- [ ] Add the immutable PiperX contract and controller profile without changing other robots.
- [ ] Re-run the focused tests and require zero failures.

### Task 2: Contract-parameterized Fold Box solver

**Files:**
- Modify: `scripts/render_factory_dual_xarm6_fold_box.py`
- Modify: `tests/factory_bimanual/test_fold_box_mount_config.py`

**Interfaces:**
- Produces: `solve_multibranch_single_arm_method(..., robot_name="xarm6")`, `mapped_quaternions_at_joint_midpoint(..., robot_name="xarm6")`, `audit_bimanual_collisions(..., robot_name="xarm6")`, and `run_fold_box(robot_name, selected_mount, output, velocity_limit_rad_s)`.
- Preserves: zero-argument `main()` behavior for xArm6.

- [ ] Write failing tests using a temporary PiperX dual scene and assert every helper resolves `left_joint1..6`, `right_joint1..6`, and the PiperX contract rather than xArm6.
- [ ] Run the focused tests and confirm the hard-coded xArm6 lookup fails.
- [ ] Thread `robot_name`/contract through candidate generation, midpoint calibration, collision audit, scene building, summary, and output metadata.
- [ ] Re-run Fold Box and rolling-multibranch tests; require xArm6 regressions to remain green.

### Task 3: Deterministic PiperX Fold Box mount search

**Files:**
- Create: `scripts/search_fold_box_piperx_mount.py`
- Create: `tests/factory_bimanual/test_fold_box_piperx_mount_search.py`

**Interfaces:**
- Produces: `layered_sample_indices(frame_count, positions) -> np.ndarray`, `rank_mount_candidate(record) -> tuple`, and `search(output_json: Path) -> dict`.
- Output: `reports/factory_bimanual/fold_box_dual_piperx/fold_box_piperx_mount_search.json`.

- [ ] Write failing tests proving sampled indices include endpoints/extrema, both base Z values are exactly 0.87 m, yaw-only mounts have zero tilt, and ranking prioritizes synchronous strict coverage then collision and required time.
- [ ] Implement a deterministic coarse search around the registered left/right target clouds, followed by bounded XY/yaw refinement of the best candidates.
- [ ] Evaluate sparse strict 6D IK with native joint limits, state collision filtering, and table-footprint constraints; store all candidates and the selected mount.
- [ ] Run focused mount-search tests, then execute `python -m scripts.search_fold_box_piperx_mount` and verify the JSON has a finite selected mount.

### Task 4: PiperX full solve and renderer

**Files:**
- Create: `scripts/render_factory_dual_piperx_fold_box.py`
- Create: `tests/factory_bimanual/test_fold_box_piperx_renderer.py`

**Interfaces:**
- Consumes: selected mount JSON and `run_fold_box(...)`.
- Produces: PiperX scene, MP4, summary JSON, trajectory NPZ, and provenance JSON under `reports/factory_bimanual/fold_box_dual_piperx/`.

- [ ] Write failing tests asserting the wrapper uses `robot_name="piperx"`, 3.0 rad/s, 60 rad/s², shared 0.87 m base Z, interpolation, and a PiperX-specific output stem.
- [ ] Implement the thin wrapper and explicit missing/invalid mount-result errors.
- [ ] Run focused tests.
- [ ] Execute the full renderer and retain explicit failure windows if synchronous strict coverage is below 100%.

### Task 5: Independent final audit and comparison

**Files:**
- Create: `reports/factory_bimanual/fold_box_dual_piperx/fold_box_piperx_vs_xarm6.json`

**Interfaces:**
- Consumes: final PiperX and xArm6 trajectory NPZ/summary files.
- Produces: a comparison record with coverage, durations, velocity/acceleration peaks, smoothing caps, collisions, failure windows, and decoded video metadata.

- [ ] Recompute joint speed and centered acceleration directly from final qpos/time arrays; do not trust stored booleans alone.
- [ ] Recompute smoothing deviations and strict TCP residual maxima.
- [ ] Decode every MP4 frame and confirm 1280×720 at 60 FPS.
- [ ] Run PiperX/factory focused tests, then `python -m pytest -q`; report unrelated existing failures separately.
- [ ] Inspect front-view frames near the beginning, middle, and end before reporting completion.
