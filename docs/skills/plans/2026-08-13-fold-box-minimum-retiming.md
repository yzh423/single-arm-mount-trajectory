# Fold Box Minimum Retiming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use global minimum necessary retiming on Fold Box to remove source-time speed failures without changing TCP targets or mount.

**Architecture:** Reuse `select_minimum_retime_path` for per-arm global branch selection. Merge the two per-arm required interval arrays into one shared bimanual execution clock and render the exact selected IK path on that clock.

**Tech Stack:** Python, NumPy, MuJoCo, pytest, OpenCV.

## Global Constraints

- Joint speed limit is exactly 3.14 rad/s.
- TCP positions, quaternions, frame order, mount and shared base height remain unchanged.
- No total-duration cap; every execution interval is at least its source interval.
- State and swept-edge collision checks remain enabled.
- No acceleration, jerk, torque or MPC feasibility claim.

---

### Task 1: Fold Box global retimed path mode

**Files:**
- Modify: `scripts/render_factory_dual_xarm6_fold_box.py`
- Test: `tests/factory_bimanual/test_fold_box_mount_config.py`

**Interfaces:**
- Consumes: `select_minimum_retime_path(...) -> RetimedPath`
- Produces: `solve_multibranch_single_arm_method(..., global_retimed=True)` with per-side `required_dt_s`.

- [ ] Write a failing test that the Fold Box solver calls the global minimum-retime selector in retimed mode.
- [ ] Run the focused test and confirm it fails because the mode is absent.
- [ ] Add the mode, using safety fraction 1.0, 35° maximum joint step, 45° maximum wrist step, and existing swept collision callback.
- [ ] Evaluate realized velocity using `path.required_dt_s` in retimed mode.
- [ ] Run `python -m pytest tests/test_rolling_multibranch_ik.py tests/factory_bimanual/test_fold_box_mount_config.py -q`.
- [ ] Commit is unavailable because the workspace has no `.git` repository; record the tested increment in the task summary.

### Task 2: Shared bimanual execution clock and diagnostics

**Files:**
- Modify: `scripts/render_factory_dual_xarm6_fold_box.py`
- Test: `tests/factory_bimanual/test_fold_box_mount_config.py`

**Interfaces:**
- Consumes: left/right `required_dt_s` arrays.
- Produces: `execution_intervals = maximum(source, left_required, right_required)` and retiming summary fields.

- [ ] Write a failing test for shared maximum interval merging and exact duration accounting.
- [ ] Run the focused test and confirm the helper is absent.
- [ ] Implement a small pure helper returning shared intervals, added duration, retimed mask and maximum edge addition.
- [ ] Make retimed frames successful when pose, orientation, speed and collision checks pass.
- [ ] Save source and required interval arrays in NPZ and summary JSON.
- [ ] Run the focused and rolling-path test suites.
- [ ] Commit is unavailable because the workspace has no `.git` repository.

### Task 3: Full Fold Box execution and verification

**Files:**
- Update generated artifacts under `reports/factory_bimanual/fold_box_dual_xarm6/`.

**Interfaces:**
- Consumes: complete renderer and Fold Box CSV.
- Produces: MP4, trajectory NPZ, summary JSON and scene manifest.

- [ ] Run `python -m scripts.render_factory_dual_xarm6_fold_box`.
- [ ] Verify every execution interval is at least its source interval.
- [ ] Recompute joint speeds independently and assert maximum ≤3.14 rad/s.
- [ ] Assert collision count is zero and retimed successful frames carry no cannot-follow reason.
- [ ] Decode every video frame with OpenCV and verify 1280×720 at 60 fps.
- [ ] Compare source duration, execution duration, coverage and failure windows against the fixed-time baseline.
- [ ] Run the complete focused regression suite and report exact results and remaining limitations.
