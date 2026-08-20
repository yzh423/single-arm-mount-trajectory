# PiperX Synchronized Mount Fixed-Time Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-mount dual PiperX for Fold Box with a genuinely paired full-task search, then independently solve and render Fold Box and Seal Bag on immutable source timestamps.

**Architecture:** Keep historical retimed artifacts untouched. A focused PiperX fixed-time runner will reuse the existing task registration, target preparation, collision-hard paired planner, collision audit, and video renderer, but will always schedule output on the CSV source timeline. Fold Box receives a new synchronized paired mount selected only after full-timeline audit; Seal Bag uses its already validated zero-collision paired mount.

**Tech Stack:** Python, NumPy, SciPy, MuJoCo, OpenCV, pytest.

## Global Constraints

- PiperX uses the official six-joint contract, native joint limits, 3.0 rad/s controller limit, and official TCP.
- Both bases are upright and share one world-Z height.
- Fold Box base separation is at least 0.60 m.
- Mount selection is paired and ranks complete source-time synchronous coverage before residual metrics.
- State and swept-edge collisions are hard rejection criteria.
- Fixed-time output timestamps equal the registered CSV source timestamps exactly; no required interval may extend them.
- Historical retimed and playback-only no-retiming files are not overwritten.
- Videos are 1280x720 at 60 fps and retain explicit failure labels.

---

### Task 1: Fixed-time and synchronized-mount contracts

**Files:**
- Create: `factory_bimanual/fixed_time_run_contract.py`
- Create: `tests/factory_bimanual/test_piperx_fixed_time_run_contract.py`

**Interfaces:**
- Produces: `source_time_schedule(source_time_s) -> tuple[np.ndarray, np.ndarray]`.
- Produces: `validate_synchronized_mount(mount, minimum_separation_m=0.60) -> float`.

- [ ] Write tests requiring immutable relative source timestamps, source-derived intervals, shared upright mount height, and rejection below 0.60 m separation.
- [ ] Run the focused test and confirm failure because the module is absent.
- [ ] Implement the two small validation functions without solver or renderer dependencies.
- [ ] Re-run the focused test and require all cases to pass.

### Task 2: Independent fixed-time PiperX runner

**Files:**
- Create: `scripts/render_factory_dual_piperx_fixed_time.py`
- Create: `tests/factory_bimanual/test_piperx_fixed_time_runner.py`

**Interfaces:**
- Consumes: a task name, paired mount JSON, `solve_collision_safe_bimanual_method`, and `source_time_schedule`.
- Produces: `run_task(task_name, mount, output, render_video=True) -> dict`.
- Produces: task-specific MP4, provenance JSON, summary JSON, scene XML/JSON, and trajectory NPZ.

- [ ] Write tests asserting only `fold_box` and `seal_bag` are accepted, execution time equals source time, `retimed_frame_count` is zero, and output names are PiperX fixed-time-specific.
- [ ] Run the focused test and confirm failure because the runner is absent.
- [ ] Implement source loading, registration, paired target preparation, collision-hard fixed-time solve, collision audit, source-time rendering, and auditable artifact writing.
- [ ] Re-run focused runner and factory-bimanual contract tests.

### Task 3: Fold Box synchronized paired re-mount

**Files:**
- Create: `scripts/search_fold_box_piperx_fixed_time_mount.py`
- Create: `tests/factory_bimanual/test_fold_box_piperx_fixed_time_mount.py`
- Produce: `reports/factory_bimanual/fold_box_dual_piperx/fold_box_piperx_fixed_time_mount_search.json`

**Interfaces:**
- Produces: deterministic paired candidates with at least 0.60 m base separation.
- Consumes: sparse paired evaluation for screening and full-timeline paired evaluation for finalists.
- Produces: `selected_mount` only from a `valid_selection` full-timeline zero-collision record.

- [ ] Write tests for deterministic candidates, shared height, upright bases, minimum separation, and full-audit-only selection.
- [ ] Run the focused test and confirm failure because the fixed-time search module is absent.
- [ ] Implement a bounded candidate search around the Fold Box registered workspace with paired left/right variations and mirrored/facing yaw candidates.
- [ ] Run sparse screening, full-audit the best finalists over all 1964 rows, and reject any colliding finalist.
- [ ] Save the complete search record and selected mount.

### Task 4: Full fixed-time runs and rendered videos

**Files:**
- Produce under: `reports/factory_bimanual/fold_box_dual_piperx/`
- Produce under: `reports/factory_bimanual/seal_bag_dual_piperx/`

**Interfaces:**
- Produces: `fold_box_piperx_fixed_time_front_720p.*`.
- Produces: `seal_bag_piperx_fixed_time_front_720p.*`.

- [ ] Run Fold Box with the new full-audited synchronized mount.
- [ ] Run Seal Bag with the validated paired mount `left=(-0.35, 0.25)`, `right=(-0.30, -0.45)`, yaw `15/15`, shared Z `0.81`.
- [ ] Require `execution_duration_s == source_duration_s`, `retimed_frame_count == 0`, and collision count zero for both tasks.
- [ ] Decode every video frame and verify 1280x720 at 60 fps.
- [ ] Inspect beginning, middle, failure-window, and ending frames; report strict coverage and limitations without relabeling failed frames as success.

### Task 5: Final regression verification

**Files:**
- Test: `tests/factory_bimanual/`

**Interfaces:**
- Verifies: fixed-time additions do not alter xArm6 or historical PiperX retimed artifacts.

- [ ] Run the focused fixed-time and mount tests.
- [ ] Run all `tests/factory_bimanual` tests and report any unrelated pre-existing failures separately.
- [ ] Confirm historical retimed MP4/NPZ hashes or timestamps were not modified by the new runner.
