# Handbook-Aligned Mount Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make yaw-only mount selection, continuous dense evaluation, failure diagnosis, evidence reporting, and validation/showcase video output conform to the approved handbook baseline.

**Architecture:** Keep cheap four-coordinate ranking separate from authoritative full chronological evaluation. Introduce small focused policy/report/video helpers, then make search, runner, cache, renderer, and reports consume the same contracts.

**Tech Stack:** Python 3.11, NumPy, MuJoCo, PyTorch/CUDA where available, pytest, ffmpeg/OpenCV.

## Global Constraints

- Active mount coordinates are exactly `[x, y, z, yaw]`; exported mounts are `[x, y, z, 0, yaw, 0]`.
- Position/orientation tolerances remain 1 mm and 1.5 degrees.
- Dense selection and final reporting use one full chronological rolling-multibranch evaluator.
- Validation video preserves source time; showcase dwell over 0.20 s compresses to 0.10 s and emits a mapping JSON.
- Old tilted results remain historical and cannot be mixed into new yaw-only reports.
- This directory is not a Git repository; create a dated `_codex_backup` snapshot before production edits.

---

### Task 1: Establish Baseline and Safe Snapshot

**Files:**
- Create: `docs/HANDBOOK_BASELINE.md`
- Create: `_codex_backup/handbook_alignment_20260811/`
- Test: `tests/test_handbook_baseline.py`

**Interfaces:**
- Produces: documented MUST/SHOULD/architecture-specific rules and a recoverable copy of every subsequently modified file.

- [ ] Write a failing test asserting the baseline document contains the mandatory selection/reporting equivalence, yaw-only, worst-render, aggregate, and video-timeline clauses.
- [ ] Run `python -m pytest tests/test_handbook_baseline.py -q`; expect failure because the document does not exist.
- [ ] Create the baseline document using the approved spec and copy target files to the dated backup before editing them.
- [ ] Re-run the baseline test; expect pass.

### Task 2: Enforce Four-Coordinate GPU and Strict Ranking Inputs

**Files:**
- Modify: `scripts/search_strict_urdf_mount.py`
- Modify: GPU coarse-search module discovered from the runner call graph.
- Test: `tests/test_mount_search_fixed_angles.py`
- Test: `tests/test_best_first_mount_search.py`

**Interfaces:**
- Produces: `compress_yaw_only_mounts(Nx6)->Nx4` and `expand_yaw_only_mounts(Nx4)->Nx6` with exact zero tilt/roll; all ranking stages consume Nx4.

- [ ] Add failing tests that reject non-Nx4 active candidates and verify GPU coarse candidates cannot contain tilt/roll.
- [ ] Run the focused tests and verify the new assertions fail for the existing six-dimensional GPU path.
- [ ] Convert GPU candidate bounds/generation/scoring to `[x,y,z,yaw]`; expand only at the MuJoCo/model boundary.
- [ ] Run focused tests and existing fixed-angle/search tests; expect pass.

### Task 3: Replace Disconnected Sequential Samples with Contiguous Windows

**Files:**
- Modify: `design_optimization/best_first_mount_search.py`
- Modify: `scripts/search_strict_urdf_mount.py`
- Test: `tests/test_best_first_mount_search.py`

**Interfaces:**
- Produces: `representative_frame_windows(positions, quaternions, requested_windows, window_length) -> tuple[np.ndarray, ...]`; each array is sorted and contiguous.

- [ ] Add failing tests for endpoint coverage, motion-peak coverage, contiguity, deterministic output, and no cross-window sequential jump calculation.
- [ ] Run the focused tests; expect missing-function/behavior failures.
- [ ] Implement deterministic 3–5 window selection and evaluate each window with reset branch state; aggregate ranking metrics without calling it full-episode success.
- [ ] Run focused and integration tests; expect pass.

### Task 4: Make Dense Promotion Authoritative and Chronological

**Files:**
- Modify: `scripts/search_strict_urdf_mount.py`
- Modify: `scripts/strict_mujoco_ik.py` only if an explicit evaluator wrapper is needed.
- Modify: `design_optimization/incremental_mount_evaluator.py` if its merge API is currently used for dense success.
- Test: `tests/test_best_first_search_integration.py`
- Test: `tests/test_incremental_mount_evaluator.py`
- Test: `tests/test_single_arm_runner_pipeline.py`

**Interfaces:**
- Produces: one authoritative `evaluate_full_episode_candidate(...)` result used directly by selection, cache, report, and render eligibility.

- [ ] Add failing tests proving merged independent subsets cannot set `episode_success=True`, dense evaluation receives all frames in chronological order, and runner does not perform a status-changing replay.
- [ ] Run focused tests; verify failures reproduce the Doosan/XArm status-flip class.
- [ ] Remove dense subset-result merging from success decisions; run the complete rolling multibranch path for promoted candidates and persist the winning result.
- [ ] Make final replay either the identical cached result or a labelled audit that cannot overwrite selection status.
- [ ] Run focused tests; expect pass.

### Task 5: Implement Bounded Post-Success Challenge Search

**Files:**
- Modify: `design_optimization/best_first_mount_search.py`
- Modify: `scripts/search_strict_urdf_mount.py`
- Test: `tests/test_best_first_mount_search.py`
- Test: `tests/test_mount_search_telemetry.py`

**Interfaces:**
- Produces: audit fields `first_success_dense_index`, `challenged_region_count`, `post_success_expansions`, `post_success_dense_candidates`, `tie_count`, and `stop_reason`.

- [ ] Add failing tests requiring three distinct regions where available, incumbent-neighbour refinement, two top challengers, deterministic ties, and a real stopping predicate.
- [ ] Run focused tests; expect failures showing unused/incomplete current policy fields.
- [ ] Implement bounded challenger selection and stop logic; do not dense-evaluate every local candidate.
- [ ] Run focused tests; expect pass and local population remains exactly 64 by default.

### Task 6: Add Falsifiable Failure Classification

**Files:**
- Modify: `design_optimization/episode_follow_metrics.py`
- Create: `scripts/strict_failure_diagnostics.py`
- Test: `tests/test_episode_follow_metrics.py`
- Create: `tests/test_strict_failure_diagnostics.py`

**Interfaces:**
- Produces: reason values `position_unreachable`, `pose_infeasible`, `collision`, `branch_lost`, `jump_violation`, `iteration_exhausted`, and `unknown` plus control evidence.

- [ ] Add failing unit tests for reason precedence and an ARX-shaped case where independent pose IK succeeds after rolling tracking fails.
- [ ] Run tests; expect classification failures/missing module.
- [ ] Implement classification from rolling result, position-only probes, independent multi-restart pose probes, collision state, and jump diagnostics.
- [ ] Run focused tests; expect pass.

### Task 7: Add Provenance and Report Contracts

**Files:**
- Create: `scripts/result_provenance.py`
- Modify: `scripts/search_strict_urdf_mount.py`
- Modify: relevant report builder used by `reports/single_arm/ten_arm_two_single_tasks/`.
- Create: `tests/test_result_provenance.py`
- Modify: `tests/test_mount_search_telemetry.py`

**Interfaces:**
- Produces: verification states and `build_result_summary(...)` with mean, threshold count, p10, tie count, solver budget/noise, unmodelled risks, and inspected-render record.

- [ ] Add failing tests for rejected silent UNVERIFIED promotion and all mandatory result fields.
- [ ] Run focused tests; expect failures.
- [ ] Implement machine-readable provenance and report aggregation; label claims as best-found under declared budget.
- [ ] Run focused tests; expect pass.

### Task 8: Split Validation and Showcase Timelines

**Files:**
- Create: `scripts/video_timeline.py`
- Modify: `scripts/render_strict_single_arm_task.py`
- Modify: `scripts/run_strict_render_batch.py`
- Create: `tests/test_video_timeline.py`
- Modify: `tests/test_strict_renderer_sampling.py`

**Interfaces:**
- Produces: `validation_timeline(source_time_s)` identity mapping and `showcase_timeline(source_time_s, dwell_threshold_s=0.20, compressed_dwell_s=0.10)` with JSON-serializable source/display mapping.

- [ ] Add failing tests for identity validation timing, unchanged short dwell, compressed long dwell, unchanged moving intervals, monotonic mapping, and visible showcase label metadata.
- [ ] Run focused tests; expect missing-function failures.
- [ ] Implement timeline helpers and renderer CLI mode; preserve solver/cache arrays and emit mapping JSON beside showcase mp4.
- [ ] Run focused renderer/video tests; expect pass.

### Task 9: Full Verification and Three-Task Acceptance

**Files:**
- Modify only defects exposed by tests, each behind a new failing regression test.
- Generate: new yaw-only cache, reports, validation/showcase videos, and inspection records under a new non-historical output namespace.

**Interfaces:**
- Consumes all prior task contracts.
- Produces final evidence for XArm6/open-box-2, Doosan/cap-left, and ARX-X5/cap-left.

- [ ] Run `python -m pytest -q`; expect all tests pass.
- [ ] Run the three searches with `python -u`, capturing stage time, candidate counts, strict frame solves, first success, challenger work, ties, and stop reason.
- [ ] Confirm every selected mount has zero tilt/roll and dense status equals reported status.
- [ ] Render validation videos for diagnostic timing and showcase videos with mapping JSON.
- [ ] Decode-check and personally inspect the worst result plus XArm dwell behaviour; record inspected paths and observations.
- [ ] Build the final handbook-aligned report with aggregate/noise/provenance/unmodelled-risk sections and compare runtime against the previous three-task logs.
