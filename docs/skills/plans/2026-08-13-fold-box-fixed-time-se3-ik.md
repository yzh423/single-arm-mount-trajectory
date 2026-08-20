# Fold Box Fixed-Time SE(3) IK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Fold Box fixed-time IK failures by reconstructing held quaternion samples, carrying real joint velocity into branch scoring, and reporting precise failure classes.

**Architecture:** A focused trajectory utility reconstructs only hold-then-jump quaternion samples and returns immutable diagnostics. The rolling branch selector accepts the previous realized joint velocity. The Fold Box renderer composes both features, audits candidate-edge feasibility, and persists before/after evidence.

**Tech Stack:** Python, NumPy, MuJoCo, pytest, OpenCV renderer.

## Global Constraints

- Execution duration remains exactly 23.509543792 s.
- TCP tolerances remain 0.001 m and 1.5 degrees.
- Joint velocity remains at or below 3.14 rad/s.
- Mount, shared Z=0.87 m, source CSV, robot limits, and front camera remain unchanged.
- First/last target poses, frame count, timestamps, and collision-free execution are mandatory.

---

### Task 1: Quaternion Hold Reconstruction

**Files:**
- Create: `factory_bimanual/quaternion_trajectory.py`
- Create: `tests/factory_bimanual/test_quaternion_trajectory.py`

**Interfaces:**
- Produces: `reconstruct_held_quaternions(time_s, quaternion_wxyz, *, stationary_rad=1e-9, motion_rad=1e-6) -> QuaternionReconstruction`
- Produces: `QuaternionReconstruction(quaternion_wxyz, changed, change_rad)`

- [ ] Write tests proving endpoint preservation, nonuniform-time SLERP, unit norm, unchanged constant-rate motion, and input validation.
- [ ] Run `python -m pytest tests/factory_bimanual/test_quaternion_trajectory.py -q`; expect import failure.
- [ ] Implement adjacent-sign normalization, shortest-path SLERP, and hold-then-jump replacement only.
- [ ] Re-run the focused test; expect all tests to pass.

### Task 2: Velocity-Aware Rolling Branch Cost

**Files:**
- Modify: `scripts/rolling_multibranch_ik.py`
- Modify: `tests/test_rolling_multibranch_ik.py`

**Interfaces:**
- `select_rolling_branch(..., initial_velocity_rad_s: np.ndarray | None = None)`
- `select_receding_horizon_path` tracks the latest realized velocity and passes it to every window.

- [ ] Add a test where two paths are position-feasible but only one continues the supplied initial velocity smoothly.
- [ ] Run that test; expect a signature error or the zero-velocity branch to win.
- [ ] Initialize frontier velocity from the optional input, validate its shape/finite values, and carry realized velocity across receding windows and bounded recovery.
- [ ] Run `python -m pytest tests/test_rolling_multibranch_ik.py -q`; expect all tests to pass.

### Task 3: Fold Box Integration and Diagnostics

**Files:**
- Modify: `scripts/render_factory_dual_xarm6_fold_box.py`
- Modify: `tests/factory_bimanual/test_fold_box_mount_config.py`

**Interfaces:**
- Fold Box uses reconstructed mapped quaternion targets for IK and rendering.
- NPZ adds original target quaternion, reconstruction mask/change, and intrinsic edge feasibility arrays.
- Failure classification returns `SOURCE TIMING INFEASIBLE`, `RECOVERY PROPAGATION`, `POSE UNREACHABLE`, or `COLLISION BLOCKED` only when strict tracking fails.

- [ ] Add integration tests for reconstruction configuration, unchanged mount, strict-success reason `ok`, and mutually exclusive failure categories.
- [ ] Run focused tests; expect missing integration behavior.
- [ ] Integrate reconstruction after TCP quaternion mapping, compute candidate-layer edge categories, correct reason priority, and extend summary/NPZ provenance.
- [ ] Run all focused Fold Box, trajectory, and rolling IK tests; expect pass.

### Task 4: Full Planning, Safety Audit, and Video

**Files:**
- Update generated artifacts under `reports/factory_bimanual/fold_box_dual_xarm6/`.

**Interfaces:**
- Produces updated MP4, summary JSON, trajectory NPZ, provenance JSON, and scene files.

- [ ] Archive the current formal artifacts with a `.prese3` suffix.
- [ ] Run `python scripts/render_factory_dual_xarm6_fold_box.py` with `PYTHONPATH=.`.
- [ ] Assert exact duration, unchanged endpoints/mount, max velocity <=3.14 rad/s, zero state/edge collision, decodable 1280x720 60 FPS video, and non-regression from 86.8635%/258 frames.
- [ ] Compare source orientation speed, intrinsic infeasible edges, recovery propagation, strict coverage, and visible failure windows before/after.
