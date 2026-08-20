# Mount / IK Fidelity Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an isolated xArm6/OpenArm pilot whose 4096-mount coarse stage retains a traceable diverse shortlist and whose strict stages use real layered IK candidates with an effective finite horizon, real timestamps, and state/swept collision checks.

**Architecture:** Add a pure candidate-funnel module for diversity and cross-fidelity statistics. Add an additive layered solver entry point that reuses `rolling_multibranch_ik.select_receding_horizon_path`; do not change the legacy solver used by historical reports. Add a pilot runner that owns stage budgets, atomic caches, A/B metrics, reports, and videos.

**Tech Stack:** Python 3, NumPy, SciPy, PyTorch/CUDA, MuJoCo, pytest, existing official-model registry and render pipeline.

## Global Constraints

- Use 4096 deterministic scrambled-Sobol X/Y/Z/Yaw candidates; Tilt=0 and Roll=0.
- Use the fixed installation bounds X `[-0.50, 0.50] m`, Y `[-0.55, 0.25] m`, Z `[0.02, 0.65] m`, Yaw `[-180, 180] deg`.
- Use `pick-right-left/teleop_20260805_154215` trimmed by 0.15 s at both ends.
- Pilot robots are `xarm6` and `openarm`; all artifacts go to a new pilot directory.
- Official URDF, physical TCP, table, pedestal, self-collision, state collision, and swept-edge collision remain authoritative.
- A complete episode succeeds only if every retained frame and edge is valid.
- The workspace is not a Git repository; each green test checkpoint replaces the unavailable commit step.

---

### Task 1: Candidate funnel and cross-fidelity metrics

**Files:**
- Create: `design_optimization/fidelity_funnel.py`
- Create: `tests/test_fidelity_funnel.py`

**Interfaces:**
- Produces `select_diverse_candidates(mounts, scores, *, lower, upper, count, pool_size) -> np.ndarray`.
- Produces `cross_fidelity_metrics(candidate_ids, coarse_scores, strict_candidate_ids, strict_scores, *, top_k) -> dict`.

- [ ] Write failing tests proving the selector retains the best candidate plus distant mount basins, is deterministic, and rejects incompatible shapes.
- [ ] Run `pytest -q tests/test_fidelity_funnel.py` and confirm imports/functions are missing.
- [ ] Implement normalized-distance farthest-point selection with score tie-breaking and stable candidate IDs.
- [ ] Run the tests and confirm they pass.
- [ ] Write failing tests for Spearman, strict top-K recall, optimistic pass rate, and candidate-ID traceability.
- [ ] Implement metrics using `scipy.stats.spearmanr`, treating constant ranks as undefined rather than zero.
- [ ] Run the focused tests and existing mount-search primitive tests.

### Task 2: Real per-frame IK candidate layers

**Files:**
- Modify: `scripts/strict_mujoco_ik.py`
- Modify: `tests/test_strict_mujoco_ik.py`

**Interfaces:**
- Produces `generate_pose_candidate_layers(model, site_name, joint_names, targets_xyz_m, targets_quaternion_wxyz, *, candidates_per_frame, iterations, global_seed_count, position_tolerance_m, orientation_tolerance_rad, candidate_collision_free, rng_seed) -> tuple[tuple[BranchCandidate, ...], ...]`.
- Produces `solve_pose_path_layered(..., horizon, beam_width, time_s, velocity_limit_rad_s, maximum_frame_jump_rad, candidate_collision_free, transition_collision_free) -> PosePathResult`.

- [ ] Add a failing planar MuJoCo test showing each reachable frame returns multiple deduplicated IK branches and a colliding state is marked unsafe.
- [ ] Run the exact test and confirm the new API is absent.
- [ ] Implement deterministic seed propagation: prior-layer solutions first, then midpoint and deterministic global seeds; solve each seed independently and deduplicate at 1 degree.
- [ ] Run the candidate-layer tests.
- [ ] Add a failing test where horizon 1 selects a dead-end branch and horizon 2 selects the future-feasible branch through the public layered path-selection helper.
- [ ] Implement `solve_pose_path_layered` by passing generated layers, real intervals, periodic mask, velocity/jump limits, and transition callback to `select_receding_horizon_path`.
- [ ] Re-evaluate the selected q path in MuJoCo and populate all `PosePathResult` fields without relabelling safe holds as collisions.
- [ ] Run strict IK and rolling planner test suites.

### Task 3: Strict mount-stage evaluator parity

**Files:**
- Modify: `scripts/search_strict_urdf_mount.py`
- Modify: `tests/test_best_first_search_integration.py`

**Interfaces:**
- Produces `_evaluate_layered_mount_candidates(...)` for contiguous strict windows and full episodes.
- Window records contain `scope="window"` and can never set `episode_success=True`.
- Full records contain candidate ID, state/edge failure counts, and chronological timeline evidence.

- [ ] Add failing tests that window records cannot claim episode success and all promoted candidate IDs survive merging.
- [ ] Add a failing test that a swept-collision edge makes a full candidate hard-infeasible.
- [ ] Implement the additive layered evaluator while leaving legacy and existing best-first entry points unchanged.
- [ ] Run focused search tests and strict collision contract tests.

### Task 4: Pilot orchestrator and atomic stage artifacts

**Files:**
- Create: `scripts/run_mount_ik_fidelity_pilot.py`
- Create: `tests/test_mount_ik_fidelity_pilot.py`

**Interfaces:**
- CLI supports `--robots`, `--coarse-candidates`, `--proxy-retain`, `--strict-window-retain`, `--full-retain`, `--final-retain`, `--workers`, and `--smoke`.
- Default stage counts are 4096 → 256 → 128 → 24 → 8–12.
- Writes `spec.json`, `coarse_candidates.json`, per-robot window/full/final JSON, `matrix.partial.json`, and `metrics.json` atomically.

- [ ] Add failing parser/default tests and an artifact-resume fingerprint test.
- [ ] Implement deterministic candidate generation and hard physical gate.
- [ ] Reuse the GPU proxy evaluator but persist candidate-level rows; proxy collision is a feature, not a hard gate.
- [ ] Select 256 diverse candidates, promote 128 to real contiguous-window evaluation, 24 to full chronological evaluation, and refine at least four distinct exact basins.
- [ ] Record per-stage elapsed time, frame solves, candidate IDs, reasons, Spearman, recall@K, and optimistic error rate.
- [ ] Run parser/unit tests without launching the pilot.

### Task 5: Pilot reports and videos

**Files:**
- Create: `scripts/build_mount_ik_fidelity_pilot_report.py`
- Modify: `tests/test_mount_ik_fidelity_pilot.py`

**Interfaces:**
- Produces `report.md`, `report.json`, comparison figures, and one success-or-failure MP4 per robot.

- [ ] Add failing completeness tests requiring both robot rows, stage metrics, failure causes, mount values, and video paths.
- [ ] Implement Markdown/JSON report and comparison figures from pilot artifacts.
- [ ] Reuse `render_strict_single_arm_task.py` for both pass and fail finalists.
- [ ] Run report tests.

### Task 6: Smoke, full pilot, and verification

**Files:**
- Generate only under `reports/single_arm/mount_ik_fidelity_pilot/`.

- [ ] Run `python scripts/run_mount_ik_fidelity_pilot.py --smoke --robots xarm6 openarm` and inspect all stage artifacts.
- [ ] Run all focused test suites and repair only pilot-scoped failures.
- [ ] Run the default 4096-candidate xArm6/OpenArm pilot with resumable atomic outputs.
- [ ] Render both final videos, including failures.
- [ ] Verify MP4 decode, cache fingerprints, state/swept collision arrays, candidate counts, report completeness, and absence of writes to formal ten-arm reports.
- [ ] Run the complete relevant regression suite and report exact pass counts plus pilot A/B metrics.
