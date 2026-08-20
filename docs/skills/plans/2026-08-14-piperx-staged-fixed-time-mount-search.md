# PiperX Staged Fixed-Time Mount Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-search upright dual-PiperX mounts for Fold Box and Seal Bag, select each mount from full source-time evidence, and reuse the winning solve for rendering without a second IK run.

**Architecture:** Reuse the xArm6 bounded upright Mount methodology with the PiperX robot contract: scan left/right XY, relative spacing, yaw, and shared height; evaluate candidates with the same multiseed IK, rolling branch selection, and scoring order. Use a deterministic staged funnel only as a budget mechanism: 32 candidates receive a sparse xArm6-style evaluation, the best four receive denser targeted evaluation, and only the best two receive full fixed-time solves. Full solves are cached as NPZ/summary artifacts; the selected winner is rendered from that cache.

**Tech Stack:** Python, NumPy, SciPy, MuJoCo, OpenCV, pytest.

## Global Constraints

- Search both `fold_box` and `seal_bag` with upright bases, shared base Z, and at least `0.60 m` base separation.
- Use the official PiperX model, `3.0 rad/s` velocity limit, `35 deg` maximum joint step, `1 mm` position tolerance, and `1.5 deg` orientation tolerance.
- Fixed-time execution timestamps must equal relative source timestamps exactly; retiming is forbidden.
- Sparse and dense stages use the same xArm6-style per-arm solver and scoring semantics as the full stage; they may reject candidates but may not certify the winner.
- The selected mount must come from a complete source-row solve and final dual-arm state/swept-edge collision audit.
- Rank full finalists in the xArm6 order: synchronous strict coverage first, then collision count, pose error, continuity/failure duration, and separation/conditioning tie-breakers.
- Never repeat the winning full IK solve solely for video rendering.
- Preserve all historical mount, trajectory, and video artifacts under distinct names.

---

### Task 1: Staged-search contracts

**Files:**
- Create: `factory_bimanual/staged_mount_search.py`
- Create: `tests/factory_bimanual/test_staged_piperx_mount_search.py`

**Interfaces:**
- Produces: `rank_full_fixed_time_mount(record) -> tuple` using the xArm6 order: synchronous success, collision, pose error, continuity/failure duration, and separation/conditioning tie-breakers.
- Produces: `select_full_fixed_time_mount(records, expected_rows) -> dict` accepting only complete source-time records.
- Produces: `targeted_source_indices(frame_count, failure_mask, maximum) -> np.ndarray` covering endpoints, uniform rows, and old failure windows.

- [ ] Write tests that reject sampled, retimed, incomplete, and malformed full records; require deterministic targeted indices and collision-first ranking.
- [ ] Run `python -m pytest tests/factory_bimanual/test_staged_piperx_mount_search.py -q` and confirm import failure.
- [ ] Implement the three pure functions with no MuJoCo dependency.
- [ ] Re-run the focused test and require all cases to pass.

### Task 2: Cache-compatible fixed-time runner

**Files:**
- Modify: `scripts/render_factory_dual_piperx_fixed_time.py`
- Modify: `tests/factory_bimanual/test_piperx_fixed_time_runner.py`

**Interfaces:**
- Produces: `render_saved_run(task_name, mount, trajectory_path, output) -> dict`.
- Consumes: a trajectory NPZ produced by `run_task(..., render_video=False)` and verifies source timestamps, row count, qpos shape, and collision arrays before rendering.

- [ ] Write tests requiring cache validation and rejecting any cached run with changed source timestamps or retimed frames.
- [ ] Run the focused tests and confirm failure because `render_saved_run` is absent.
- [ ] Implement cache validation plus rendering from saved qpos and diagnostics without calling either IK solver.
- [ ] Re-run the runner tests and fixed-time contract tests.

### Task 3: Bounded two-task Mount search

**Files:**
- Create: `scripts/search_piperx_fixed_time_mounts.py`
- Modify: `tests/factory_bimanual/test_staged_piperx_mount_search.py`
- Produce: `reports/factory_bimanual/<task>_dual_piperx/<task>_piperx_staged_fixed_time_mount_search.json`

**Interfaces:**
- Consumes: xArm6-style bounded upright candidate generation, task-specific registered trajectories, `solve_multibranch_single_arm_method`, final dual-arm collision audit, and `run_task(render_video=False)`.
- Produces: checkpointed sparse, dense, and full records plus `selected_mount` and the winning trajectory path.

- [ ] Write tests requiring exactly 32 deterministic coarse candidates, four dense finalists, at most two full finalists, task-specific row counts, and resumable checkpoint selection.
- [ ] Run the focused tests and confirm failure for the new orchestration functions.
- [ ] Implement sparse screening with 20 targeted rows and the same xArm6-style IK/path scorer, dense screening with at most 96 targeted rows, and full evaluation of at most two finalists.
- [ ] Save a checkpoint after every candidate and reuse existing complete records on restart.
- [ ] Run both searches in parallel while leaving unrelated user processes untouched.

### Task 4: Select, render, and verify winners

**Files:**
- Produce: `reports/factory_bimanual/fold_box_dual_piperx/fold_box_piperx_remounted_fixed_time_front_720p.*`
- Produce: `reports/factory_bimanual/seal_bag_dual_piperx/seal_bag_piperx_remounted_fixed_time_front_720p.*`

**Interfaces:**
- Consumes: selected full-audited mount and cached winning trajectory from Task 3.
- Produces: MP4, summary JSON, provenance JSON, scene XML/JSON, trajectory NPZ, and preview PNG for each task.

- [ ] Render each winner directly from its cached full solve.
- [ ] Verify `retimed_frame_count == 0`, exact source/execution timestamp equality, complete row count, and video decode at `1280x720`, `60 fps`.
- [ ] Report collision counts and strict synchronous coverage without labeling failed frames as success.
- [ ] Run the focused test suite and the artifact validation command before claiming completion.
