# PiperX Zero-Collision Follow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve 100% calibrated TCP follow at 1 mm / 0.5 degrees for Fold_Box and Seal_Bag while reducing both MuJoCo state and swept-edge collision counts to zero.

**Architecture:** Extend the fixed source-hand-to-TCP mapping from rotation-only to a task-level SE(3) transform. Fold_Box additionally uses one time-bounded right-wrist orientation adaptation during the physically infeasible opening segment, and the complete-follow runner rejects colliding candidate pairs instead of merely ranking them last. Published evidence distinguishes source hand traces, calibrated TCP targets, and the bounded Fold_Box adaptation.

**Tech Stack:** Python 3.11, NumPy, SciPy Rotation, MuJoCo, OpenCV, ReportLab, pytest.

## Global Constraints

- IK acceptance remains 1 mm position error and 0.5 degree orientation error against the calibrated TCP target.
- Fold_Box and Seal_Bag must each retain 100% source-frame coverage.
- State collision count and swept incoming-edge collision count must both be zero.
- Joint velocity and acceleration limits remain 1 rad/s and 4 rad/s^2.
- Fixed tool translations are task-level constants in the registered source-hand frame; no per-frame separation clamp is allowed.
- Fold_Box right-wrist adaptation is limited to -12.5 degrees, held through 0.3666666667 s and linearly returned to zero by 1.0 s.

---

### Task 1: Fixed SE(3) tool mapping and wrist schedule

**Files:**
- Modify: `factory_bimanual/tool_frame_calibration.py`
- Modify: `factory_bimanual/piperx_recommended.py`
- Modify: `configs/piperx_recommended_v31.json`
- Modify: `scripts/run_piperx_recommended_v31.py`
- Test: `tests/factory_bimanual/test_piperx_recommended_config.py`
- Test: `tests/factory_bimanual/test_run_piperx_recommended_v31.py`

**Interfaces:**
- Produces: `apply_fixed_tool_translation(position_m, source_quaternion_wxyz, translation_m) -> np.ndarray`.
- Produces: `apply_bounded_wrist_adaptation(quaternion_wxyz, time_s, spec) -> tuple[np.ndarray, np.ndarray]`.
- Consumes: per-task translation vectors and optional wrist adaptation from `RecommendedMountSpec`.

- [ ] **Step 1: Write failing tests** for quaternion-rotated fixed translation, zero translation identity, schedule hold/ramp/end behavior, configuration parsing, and evidence fields.
- [ ] **Step 2: Run the focused tests** and verify they fail because the translation and wrist APIs do not exist.
- [ ] **Step 3: Implement minimal typed configuration and mapping functions**, applying translation before replacing source quaternions with mapped TCP quaternions.
- [ ] **Step 4: Add the experimentally verified constants**: Fold_Box translations `[-0.01399595, 0.00244380, -0.02057040]` and `[0.00861553, 0.01174536, 0.02031795]` m plus the bounded right wrist schedule; Seal_Bag translations `[-0.00347928, 0.00355632, 0.00867452]` and `[0.00188397, -0.00019278, -0.00981904]` m.
- [ ] **Step 5: Run focused tests** and commit the mapping slice.

### Task 2: Make collision-free selection a hard invariant

**Files:**
- Modify: `factory_bimanual/complete_follow.py`
- Test: `tests/factory_bimanual/test_complete_follow.py`

**Interfaces:**
- Consumes: `MuJoCoPairedCollisionChecker.state` and `.transition`.
- Produces: `_choose_pair(...) -> None` when every strict pair collides; `CompleteFollowInfeasibleError` after global rescue also has no safe pair.

- [ ] **Step 1: Write a failing regression test** where all local and global pairs collide and assert an infeasible error instead of a colliding result.
- [ ] **Step 2: Run the regression test** and observe the current colliding selection.
- [ ] **Step 3: Filter colliding state or transition pairs out of `_choose_pair`** and pass collision predicates into lossless retiming.
- [ ] **Step 4: Run complete-follow and collision-adapter tests**, then commit the invariant slice.

### Task 3: Regenerate and validate zero-collision evidence

**Files:**
- Modify: `scripts/validate_piperx_two_task_bundle.py`
- Modify: `tests/factory_bimanual/test_validate_piperx_two_task_bundle.py`
- Regenerate: `reports/piperx_two_task_complete_follow/fold_box/*`
- Regenerate: `reports/piperx_two_task_complete_follow/seal_bag/*`
- Regenerate: `reports/piperx_two_task_complete_follow/two_task_manifest.json`
- Regenerate: `reports/piperx_two_task_complete_follow/two_task_summary.csv`
- Modify: `reports/piperx_two_task_complete_follow/two_task_experiment_log.json`

**Interfaces:**
- Validator requires calibrated target basis, 100% coverage, zero state collisions, zero incoming-edge collisions, valid hashes, and complete MP4 decode.

- [ ] **Step 1: Write failing validator tests** for nonzero collision counts and missing tool-mapping evidence.
- [ ] **Step 2: Implement the stricter validation contract** and verify the tests pass.
- [ ] **Step 3: Run both tasks with 16 candidates and render 1280x720 30 fps MP4s** into the published bundle.
- [ ] **Step 4: Run full video decode and manifest regeneration**; inspect start, adapted segment, middle, and end frames.
- [ ] **Step 5: Commit the machine-readable evidence and videos.**

### Task 4: Update report and reader guidance

**Files:**
- Modify: `scripts/build_piperx_two_task_report.py`
- Modify: `tests/factory_bimanual/test_piperx_two_task_report.py`
- Regenerate: `reports/piperx_two_task_complete_follow/PiperX双任务严格完全跟随实验报告.pdf`
- Modify: `README.md`
- Modify: `reports/README.md`

**Interfaces:**
- Report consumes the new manifest and experiment log and explicitly separates raw hand trace, fixed SE(3) TCP calibration, bounded Fold_Box wrist adaptation, strict IK error, and simulation-only collision evidence.

- [ ] **Step 1: Write failing report assertions** for zero collisions and explicit adaptation disclosure.
- [ ] **Step 2: Update report narrative and tables**, regenerate the PDF, render all pages, and inspect for clipping or stale collision claims.
- [ ] **Step 3: Update README commands and metrics**, preserving historical report/video links and the hardware-safety caution.
- [ ] **Step 4: Run README structural/link checks and commit documentation.**

### Task 5: Final verification and submission

**Files:**
- Verify all changed files and published artifacts.

**Interfaces:**
- Produces: a clean branch whose HEAD is byte-for-byte equal to remote `main` after push.

- [ ] **Step 1: Run targeted tests, full pytest with the known SciPy test isolated to `arm-design`, all CLI help commands, bundle validation, PDF page/text checks, and README link checks.**
- [ ] **Step 2: Perform five-axis code review, diff/secret checks, and confirm the workspace is clean.**
- [ ] **Step 3: Push `HEAD:main` and verify `git ls-remote origin refs/heads/main` equals local HEAD.**
