# Yaw-Only Mount Search and Realtime Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Search mount XYZ and yaw only while keeping tilt/pitch and roll exactly zero, and render strict-task videos at 1x speed.

**Architecture:** Preserve the six-field mount record for cache/report compatibility, but collapse the tilt/pitch and roll bounds to zero before any global or local candidate generation and normalize all warm starts to those fixed values. Change the renderer's single playback-speed constant to 1.0 so its existing duration and interpolation logic emits real-time video and matching metadata.

**Tech Stack:** Python, NumPy, pytest, MuJoCo renderer, ffmpeg.

## Global Constraints

- `tilt_pitch_deg` must always be `0.0` in generated and selected candidates.
- `roll_deg` must always be `0.0` in generated and selected candidates.
- `yaw_deg` and base XYZ remain searchable.
- Cache and report candidates remain six-dimensional in the existing field order.
- Rendered video playback speed and audit metadata must be `1x`.
- The workspace is not a Git repository; preserve changed originals under `_codex_backup` instead of committing.

---

### Task 1: Constrain the mount search dimensions

**Files:**
- Modify: `scripts/search_strict_urdf_mount.py`
- Test: `tests/test_mount_search_fixed_angles.py`

**Interfaces:**
- Consumes: six-element `lower`, `upper`, and warm-start mount arrays in `[x, y, z, tilt_pitch, yaw, roll]` order.
- Produces: `_yaw_only_mount_search_space(lower, upper, warm_starts) -> tuple[np.ndarray, np.ndarray, np.ndarray]` with columns 3 and 5 fixed to zero.

- [ ] **Step 1: Write a failing test** that passes nonzero angular bounds and warm starts, then asserts copied outputs have `lower[[3,5]] == upper[[3,5]] == 0`, every warm start has columns 3 and 5 zero, yaw is unchanged, and inputs are not mutated.
- [ ] **Step 2: Run `python -m pytest tests/test_mount_search_fixed_angles.py -q`** and verify import/behavior failure because the helper does not exist.
- [ ] **Step 3: Implement `_yaw_only_mount_search_space`** using defensive NumPy copies and call it in `main()` before warm-start validation and either search policy runs.
- [ ] **Step 4: Run `python -m pytest tests/test_mount_search_fixed_angles.py tests/test_best_first_search_integration.py tests/test_hierarchical_mount_search.py -q`** and expect all tests to pass.

### Task 2: Render videos at original speed

**Files:**
- Modify: `scripts/render_strict_single_arm_task.py`
- Test: `tests/test_render_strict_single_arm_task.py`

**Interfaces:**
- Consumes: source timestamps and the renderer module defaults.
- Produces: default `PLAYBACK_SPEED = 1.0`, frame count equal to source duration times FPS, and audit field `playback_speed = "1x"`.

- [ ] **Step 1: Write a failing test** asserting the default playback speed is 1.0 and a 10-second timestamp range yields 300 frames at 30 FPS.
- [ ] **Step 2: Run `python -m pytest tests/test_render_strict_single_arm_task.py -q`** and verify the new assertion fails against the current 2x default.
- [ ] **Step 3: Change `PLAYBACK_SPEED` to `1.0`, update the module description from 2x to 1x/original speed, and derive audit text as `f"{PLAYBACK_SPEED:g}x"`** so metadata cannot drift from the calculation.
- [ ] **Step 4: Run `python -m pytest tests/test_render_strict_single_arm_task.py -q`** and expect all tests to pass.

### Task 3: Regression verification

**Files:**
- Verify only: all Python tests.

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: evidence that search compatibility and renderer helpers remain valid.

- [ ] **Step 1: Run `python -m pytest -q`** and expect the complete suite to pass (with only pre-existing skips).
- [ ] **Step 2: Inspect changed files and backup contents** to confirm no reports, caches, or videos were regenerated and no unrelated files changed.

## Self-Review

- Spec coverage: fixed tilt/pitch, fixed roll, searchable yaw/XYZ, six-field compatibility, and 1x video are each covered.
- Placeholder scan: no TBD/TODO or unspecified implementation steps remain.
- Type consistency: the helper consumes and returns NumPy arrays; the renderer continues to use its existing numeric speed API.
