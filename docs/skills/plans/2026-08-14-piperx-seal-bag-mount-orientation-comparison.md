# Piper X Seal Bag Mount Orientation Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare optimized upright-table, horizontal-wall, and inverted Piper X bimanual mounts on the complete Seal Bag trajectory, then produce reproducible metrics, three videos, and a Chinese PDF.

**Architecture:** Extend the isolated bimanual scene builder with an explicit base-orientation contract, then build a comparison runner on the existing paired IK/collision/full-audit pipeline. Persist per-frame arrays first; derive tables, plots, videos, and PDF only from those immutable artifacts.

**Tech Stack:** Python, MuJoCo, NumPy, pytest, Matplotlib, OpenCV, ReportLab, Poppler.

## Global Constraints

- Use the complete 2439-row Seal Bag source trajectory.
- Both arms share exactly the same world-coordinate mount z within a candidate.
- The three modes use identical XY/z/yaw bounds and identical solver budgets.
- Only MuJoCo state or swept contact is a hard collision rejection; projected crossing is diagnostic.
- Do not modify protected single-arm code or result files.
- Render all three modes from the same front camera on the original source timeline.

---

### Task 1: Orientation-aware Piper X scene construction

**Files:**
- Modify: `factory_bimanual/scene_builder.py`
- Create: `factory_bimanual/mount_orientation.py`
- Test: `tests/factory_bimanual/test_mount_orientation.py`

**Interfaces:**
- Produce `MountOrientation` and `mount_quaternion(mode, xy, target_center, yaw_deg)`.
- Add optional `mount_quaternion_wxyz` and `mount_adapter_axis` inputs to `build_same_model_scene` while preserving upright defaults.

- [ ] Write tests that compile upright, wall, and inverted official Piper X scenes; assert equal left/right base z and expected base-axis world direction.
- [ ] Run the tests and observe missing-interface failures.
- [ ] Implement normalized quaternion composition and orientation-aware adapter geometry.
- [ ] Re-run focused scene tests and the existing factory scene suite.

### Task 2: Fair deterministic orientation search

**Files:**
- Create: `factory_bimanual/orientation_mount_search.py`
- Create: `scripts/run_piperx_seal_bag_orientation_comparison.py`
- Test: `tests/factory_bimanual/test_orientation_mount_search.py`

**Interfaces:**
- Produce `OrientationSearchConfig`, `generate_mounts(mode, task, config)`, `evaluate_sparse`, `refine_finalists`, and `audit_full_timeline`.
- Save resumable stage manifests under `reports/factory_bimanual/piperx_seal_bag_mount_orientation_comparison/results/`.

- [ ] Write tests proving identical candidate budgets/bounds, exact shared z, deterministic fingerprints, resumability, and collision-only hard gating.
- [ ] Run tests and observe missing-module failures.
- [ ] Implement the common expanded grid and per-mode quaternion mapping using the existing paired Piper X solver.
- [ ] Implement coarse, dense, local-refinement, and full-timeline stages with atomic result writes.
- [ ] Run focused tests and a short-prefix real MuJoCo smoke for all three modes.

### Task 3: Reproducible metric derivation

**Files:**
- Create: `factory_bimanual/orientation_comparison_metrics.py`
- Test: `tests/factory_bimanual/test_orientation_comparison_metrics.py`

**Interfaces:**
- Produce `derive_orientation_metrics(npz_path, summary_path)` and `build_comparison_table(records, baseline='upright_table')`.

- [ ] Write tests for coverage, longest failure, TCP mean/P50/P95/max, angular error, per-joint margin, singularity margin, velocity, total variation, collision counts, and baseline deltas.
- [ ] Run tests and observe missing-module failures.
- [ ] Implement metrics exclusively from persisted source-row-aligned arrays.
- [ ] Verify tables reject missing/incomplete fingerprints and unequal source-row counts.

### Task 4: Same-camera original-time videos

**Files:**
- Create: `scripts/render_piperx_seal_bag_orientation_comparison.py`
- Test: `tests/factory_bimanual/test_orientation_comparison_video.py`

**Interfaces:**
- Produce one MP4 plus provenance JSON for each mode using a fixed front camera contract.

- [ ] Write tests for causal source-time sampling, common camera parameters, diagnostic overlay inputs, and decode validation.
- [ ] Run tests and observe missing-renderer failures.
- [ ] Implement rendering from final XML/NPZ artifacts without rerunning IK.
- [ ] Decode-check frame count, duration, resolution, and first/last frames.

### Task 5: Chinese comparison PDF and end-to-end experiment

**Files:**
- Create: `reports/factory_bimanual/piperx_seal_bag_mount_orientation_comparison/generate_report.py`
- Create: `reports/factory_bimanual/piperx_seal_bag_mount_orientation_comparison/test_generate_report.py`
- Produce: `reports/factory_bimanual/piperx_seal_bag_mount_orientation_comparison/piperx_seal_bag_mount_orientation_comparison_report.pdf`

**Interfaces:**
- Consume only final JSON/NPZ/video provenance artifacts.

- [ ] Write report tests that recompute every displayed value and require all three decoded videos.
- [ ] Run the complete search and full 2439-row audits for all three modes.
- [ ] Derive raw CSV/JSON tables and relative deltas against upright-table.
- [ ] Generate trajectory, error, failure-window, joint-margin, velocity, and keyframe figures.
- [ ] Mark the PDF create operation, generate the report, render every page to PNG, inspect the contact sheet, and fix all layout defects.
- [ ] Run the full `tests/factory_bimanual` suite and isolation tests; record exact commands, durations, and limitations.

