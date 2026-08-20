# Fix Yaw-Only Search Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make XYZ+yaw-only mount search valid and ensure its result is the only artifact rendered, with accurate failure labels.

**Architecture:** Generate candidates in a four-dimensional active space and expand them to the existing six-field mount contract at the search boundary. Treat search as the authoritative full solver, make the runner fail closed rather than reuse stale caches, and centralize interpolated failure-reason selection in the renderer.

**Tech Stack:** Python, NumPy, pytest, subprocess runner, MuJoCo renderer.

## Global Constraints

- Search only base XYZ and yaw.
- Emit six-field candidates as `[x, y, z, 0.0, yaw, 0.0]`.
- Never run solve or render after a failed search.
- Never overwrite a successful search cache with `optimization(...)` output.
- Never display `FAIL: none` when either interpolation endpoint failed.
- Do not launch production searches or video rendering during implementation.
- The workspace is not a Git repository; use `_codex_backup/fix_yaw_only_pipeline_20260811` as the recovery point.

---

### Task 1: Four-dimensional candidate adapter

**Files:**
- Modify: `scripts/search_strict_urdf_mount.py`
- Modify: `tests/test_mount_search_fixed_angles.py`

**Interfaces:**
- Produces `_compress_yaw_only_mounts(mounts)` and `_expand_yaw_only_mounts(mounts)` for six-to-four and four-to-six conversion.

- [ ] Add failing round-trip and candidate-generation tests proving finite four-dimensional bounds, unchanged XYZ/yaw, and zero tilt/roll after expansion.
- [ ] Run `python -m pytest tests/test_mount_search_fixed_angles.py -q` and observe the missing adapter failure.
- [ ] Implement the adapters and call legacy/best-first search with four-dimensional bounds through an evaluation wrapper that expands candidates before model evaluation.
- [ ] Run focused search-policy tests and expect all to pass.

### Task 2: Fail-closed runner and authoritative search cache

**Files:**
- Modify: `scripts/run_twelve_arm_two_single_tasks.py`
- Modify: `tests/test_formal_run_acceleration.py` or add `tests/test_single_arm_runner_pipeline.py`

**Interfaces:**
- Produces a task pipeline where search success permits rendering directly and search failure suppresses downstream stages.

- [ ] Add failing tests with a fake stage runner proving solve is never called, render runs only after search success with a current cache, and stale cache cannot bypass failed search.
- [ ] Run the focused runner tests and observe current unconditional solve/render behavior.
- [ ] Extract a testable stage-decision helper and remove the redundant solve invocation from `_run_task`.
- [ ] Run focused runner tests and expect all to pass.

### Task 3: Accurate interpolated failure labels

**Files:**
- Modify: `scripts/render_strict_single_arm_task.py`
- Modify: `tests/test_render_strict_single_arm_task.py`

**Interfaces:**
- Produces `interpolated_failure_reason(success, failure_reason, lower, upper)`.

- [ ] Add failing tests for success-to-failure, failure-to-success, both-failed, and both-success cases.
- [ ] Run the renderer test and observe `FAIL: none` behavior at the boundary.
- [ ] Implement the helper and use it in the video overlay.
- [ ] Run all renderer tests and expect all to pass.

### Task 4: Full verification

**Files:**
- Verify all changed files and tests.

**Interfaces:**
- Produces regression evidence without modifying experiment artifacts.

- [ ] Run `python -m pytest -q` and expect zero failures.
- [ ] Run a read-only candidate smoke check proving generated six-field candidates have zero columns 3 and 5.
- [ ] Inspect timestamps under the report/video directories to confirm no production artifacts were regenerated.

## Self-Review

- Coverage includes the zero-width crash, stale-cache continuation, search-cache overwrite, and `FAIL: none` display bug.
- No placeholders remain; every task has an exact test command and behavior.
- Four-dimensional adapters preserve the public six-dimensional cache/report contract.
