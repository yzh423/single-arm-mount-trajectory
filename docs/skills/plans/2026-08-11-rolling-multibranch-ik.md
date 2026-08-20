# Rolling Multi-Branch IK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 12-frame rolling multi-branch IK planner that reduces greedy branch dead ends while preserving strict real-URDF pose and collision standards.

**Architecture:** Add a focused, NumPy-only layered branch planner module and integrate it with MuJoCo candidate generation in `strict_mujoco_ik.py`. Keep collision evaluation and cache/report generation in the existing strict pipeline, and expose planner diagnostics in the NPZ cache and JSON audit.

**Tech Stack:** Python 3.12, NumPy, MuJoCo, unittest/pytest.

## Global Constraints

- Position tolerance remains 1 mm and orientation tolerance remains 1.5 degrees.
- Real URDF/MJCF Jacobians and collision geometry remain authoritative.
- Table, pedestal, self-collision, candidate-search size, and whole-episode success criteria are not weakened.
- Use an eight-candidate beam, 12-frame horizon, timestamp-derived velocity limits, and a 25-degree per-frame safety cap.
- Formal 12-arm execution remains stopped until three representative episodes pass validation.

---

### Task 1: Layered branch planner

**Files:**
- Create: `scripts/rolling_multibranch_ik.py`
- Create: `tests/test_rolling_multibranch_ik.py`

**Interfaces:**
- Produces: `BranchCandidate`, `transition_limit_rad(dt_s, velocity_limit_rad_s, cap_rad)`, `select_rolling_branch(layers, initial_q, periodic, dt_s, velocity_limit_rad_s, cap_rad, beam_width)`.

- [ ] Write failing tests proving timestamp/cap behavior, periodic transitions, future-safe branch selection, and acceleration preference.
- [ ] Run `E:\Anaconda\python.exe -m pytest tests/test_rolling_multibranch_ik.py -q`; expect import or assertion failures.
- [ ] Implement immutable candidate records and deterministic layered dynamic programming with hard pose/collision/velocity ordering and soft acceleration, margin, residual, and travel costs.
- [ ] Re-run the focused tests; expect all pass.

### Task 2: MuJoCo candidate frontier integration

**Files:**
- Modify: `scripts/strict_mujoco_ik.py`
- Modify: `tests/test_strict_mujoco_ik.py`

**Interfaces:**
- Consumes: planner interfaces from Task 1.
- Produces: extended `PosePathResult` diagnostics `velocity_violation`, `acceleration_warning`, `branch_count`, `chosen_branch_index`, `recovery_mode`, `singularity_margin`, and `joint_limit_margin`.

- [ ] Write a failing synthetic-chain test in which frame-greedy IK selects a dead-end branch but the rolling solver follows the complete pose path.
- [ ] Run the focused test and confirm it fails for the missing rolling behavior.
- [ ] Extract deterministic per-frame IK candidate generation, deduplicate candidates by wrapped joint distance, evaluate Jacobian singularity and joint-limit margin, then commit the first edge of each 12-frame rolling solution.
- [ ] Retain limited-step recovery for pose-valid oversized transitions and hold unreachable/collision-invalid states.
- [ ] Run `E:\Anaconda\python.exe -m pytest tests/test_strict_mujoco_ik.py tests/test_rolling_multibranch_ik.py -q`; expect all pass.

### Task 3: Time constraints, cache diagnostics, and fingerprinting

**Files:**
- Modify: `scripts/solve_strict_urdf_task_cache.py`
- Modify: `scripts/run_twelve_arm_two_single_tasks.py`
- Modify: `tests/test_formal_run_acceleration.py`
- Modify: `tests/test_episode_follow_metrics.py`

**Interfaces:**
- Consumes: extended `PosePathResult` from Task 2.
- Produces: cache arrays and audit summaries for planner diagnostics; task fingerprint includes planner code and exact planner parameters.

- [ ] Write failing tests for diagnostic array persistence and planner-code cache invalidation.
- [ ] Pass source `time_s` into the rolling solver with explicit planner parameters.
- [ ] Save diagnostic arrays and summary counts without changing primary whole-episode metrics.
- [ ] Add `scripts/rolling_multibranch_ik.py`, `design_optimization/episode_follow_metrics.py`, horizon, beam, candidate count, velocity limit, and cap to `_task_fingerprint`.
- [ ] Run the focused cache/fingerprint tests; expect all pass.

### Task 4: Verification and representative real-URDF gate

**Files:**
- Create: `reports/single_arm/rolling_multibranch_validation.json`

**Interfaces:**
- Consumes: stopped-run caches, current limited-step baseline, and rolling planner results.
- Produces: reproducible three-task comparison containing successful frames, discontinuities, collisions, RMSE, maximum velocity/acceleration, and runtime.

- [ ] Run `E:\Anaconda\python.exe -m pytest -q`; require zero failures.
- [ ] Run Doosan `open-box-2`, Franka Panda `tube-left-upright`, and ARX X5 `pick-from-high-left` without overwriting stopped-run evidence.
- [ ] Write the comparison JSON and verify no new table/self-collision frames, no successful-frame regression, and discontinuity reduction on at least two tasks.
- [ ] Keep the formal run stopped if any gate fails; otherwise report readiness before restarting it.

## Execution note

This directory is not a Git repository, so the plan omits commit commands while retaining one red-green test cycle per independently reviewable task.
