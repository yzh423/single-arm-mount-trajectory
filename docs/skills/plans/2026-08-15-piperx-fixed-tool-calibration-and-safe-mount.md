# PiperX Fixed Tool Calibration and Safe Mount Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one locked cross-task PiperX coordinate calibration, collision-aware Mounts, and verified fixed-time videos with zero state and swept collisions.

**Architecture:** Isolate calibration, clearance measurement, Mount scoring, bounded IK generation, and paired path selection behind tested interfaces. All search and render entry points consume the same calibration artifact and collision vocabulary.

**Tech Stack:** Python, NumPy, SciPy, MuJoCo, pytest, OpenCV.

## Global Constraints

- Preserve the official PiperX URDF flange, gripper, and `ee_frame` transforms.
- Use one left/right fixed coordinate calibration for Fold Box and Seal Bag.
- Never smooth, reconstruct, or modify mapped orientation per frame.
- Keep Mounts upright with shared Z; scan independent XY/yaw and separation about 0.40–0.80 m, Z 0.79–1.00 m.
- Treat state collision, swept collision, and 15 mm simulation clearance as final hard constraints.
- Do not treat unreachable or held frames as collision-safe success.
- Preserve source timestamps exactly for fixed-time outputs.

---

### Task 1: Fixed cross-task orientation calibration

**Files:**
- Create: `factory_bimanual/tool_frame_calibration.py`
- Create: `scripts/calibrate_piperx_tool_frames.py`
- Create: `tests/factory_bimanual/test_tool_frame_calibration.py`
- Modify: `scripts/render_factory_dual_xarm6_se3_follow.py`

**Interfaces:**
- Produces `proper_axis_rotations()`, `apply_fixed_tool_rotation()`, `CalibrationArtifact`, `load_locked_calibration()`, and a JSON calibration artifact.

- [ ] Write failing tests for 24 proper axis rotations, fixed relative-rotation preservation, ±15 degree bounds, shared cross-task artifact fingerprints, and rejection of frame-zero mapping.
- [ ] Run `python -m pytest tests/factory_bimanual/test_tool_frame_calibration.py -q` and confirm failures identify missing interfaces.
- [ ] Implement quaternion/matrix mapping and artifact validation without per-frame smoothing.
- [ ] Implement representative-frame discrete pair search and bounded deterministic local refinement.
- [ ] Run the focused test and both PiperX runner/search tests.
- [ ] Run calibration and save the locked artifact under `reports/factory_bimanual/piperx_tool_frame_calibration.json`.

### Task 2: Clearance-aware collision adapter

**Files:**
- Modify: `factory_bimanual/mujoco_collision_adapter.py`
- Create: `tests/factory_bimanual/test_piperx_clearance.py`

**Interfaces:**
- Produces `clearance(left_q, right_q) -> ClearanceReport` and swept minimum-clearance reporting.

- [ ] Write failing tests for the five named PiperX geometry pair families and 15 mm hard margin.
- [ ] Implement deterministic geometry grouping and nearest-distance queries while retaining penetration classes.
- [ ] Add swept-edge minimum clearance over the same interpolation used by collision checking.
- [ ] Run collision adapter and collision-safe-follow tests.

### Task 3: Bounded and stratified PiperX IK candidates

**Files:**
- Modify: `factory_bimanual/mujoco_candidate_generator.py`
- Modify: `factory_bimanual/strict_bimanual_ik.py`
- Create: `tests/factory_bimanual/test_piperx_candidate_diversity.py`

**Interfaces:**
- Candidate diagnostics add normalized residual, shoulder/elbow/wrist stratum, J4/J5 risk, and clearance-ready metadata.

- [ ] Write failing tests for deterministic stratified seeds, bound-respecting iterations, tolerance-normalized residuals, and J4/J5 ±89 degree penalties.
- [ ] Replace DLS clipping with bounded trial steps and retain SLSQP only as a declared fallback.
- [ ] Rank candidates by normalized pose residual and risk without losing distinct kinematic strata.
- [ ] Run candidate, IK, and paired-follow tests.

### Task 4: Joint safe branch graph and failure taxonomy

**Files:**
- Modify: `factory_bimanual/collision_safe_follow.py`
- Modify: `scripts/render_factory_dual_xarm6_se3_follow.py`
- Modify: `scripts/render_factory_dual_piperx_fixed_time.py`
- Modify: `tests/factory_bimanual/test_collision_safe_follow.py`

**Interfaces:**
- Paired result reports connectable ratio, state/swept clearance, longest failure run, and mutually attributable failure categories.

- [ ] Write failing tests proving independent-safe arms can form an unsafe pair and that unsafe state/edge pairs never enter the graph.
- [ ] Add the 15 mm clearance gate and full paired dynamic-program cost tuple.
- [ ] Add bounded recovery and explicit failure classification.
- [ ] Verify fixed-time timing remains immutable.

### Task 5: Lexical Mount search around the PiperX reference region

**Files:**
- Modify: `factory_bimanual/staged_mount_search.py`
- Modify: `scripts/search_piperx_fixed_time_mounts.py`
- Modify: `tests/factory_bimanual/test_staged_piperx_mount_search.py`

**Interfaces:**
- Mount records expose all six lexical score fields and complete-evidence flags.

- [ ] Write failing tests for the exact score order and unreachable-frame treatment.
- [ ] Generate independent left/right XY/yaw, shared Z 0.79–1.00 m, and 0.40–0.80 m separations centred near 0.702 m.
- [ ] Make all screening consume the locked calibration and paired safe-candidate graph.
- [ ] Resume checkpoints by calibration/source fingerprint, never by task name alone.
- [ ] Select full finalists only after complete state and swept clearance audit.

### Task 6: Full acceptance and cached rendering

**Files:**
- Modify: `scripts/render_factory_dual_piperx_fixed_time.py`
- Create: `tests/factory_bimanual/test_piperx_acceptance.py`

**Interfaces:**
- Final summaries contain calibration fingerprint, state/swept collision counts, minimum clearances, synchronous coverage, longest failure interval, connectable ratio, and failure taxonomy.

- [ ] Run full Fold Box and Seal Bag joint solves without rendering and retain immutable trajectory caches.
- [ ] Assert exact source row/time alignment, zero state collision, zero swept collision, and no retiming.
- [ ] Compare coverage and longest failure run against the current baselines; do not publish an unsafe result.
- [ ] Render 1280×720 60 fps front videos from verified caches.
- [ ] Decode every video, inspect representative frames, and run the complete factory-bimanual test suite.

