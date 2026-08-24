# PiperX Multitask Fixed-Time Mount Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, run, visualize, validate, and publish a 27-trajectory × 4-mount PiperX fixed-time IK comparison with a multi-task PDF report.

**Architecture:** A dataset contract discovers the exact valid dual-hand CSV inventory. A resumable experiment runner searches each non-baseline mount once on the configured representative dual-hand take of each task family, then reuses that family mount while running strict IK/collision validation at native timestamps for every trajectory. It produces one evidence shard per trajectory and mount. Separate aggregation, synchronized 2×2 rendering, and report modules consume only validated shards.

**Tech Stack:** Python 3.11, NumPy, MuJoCo, OpenCV, Matplotlib, ReportLab, pytest, JSON/NPZ/XML/MP4.

## Global Constraints

- Input is every dual-hand CSV under `data/factory/` except paths containing `_rejected_short`.
- The current inventory must resolve to 27 trajectories and 12 task families.
- Mount modes are `baseline`, `upright_table`, `horizontal_wall`, and `inverted`.
- Fixed-time means exact original timestamps, no retiming, no inserted frames, and no added duration.
- Strict calibrated TCP gates are 0.001 m and 0.5 degrees.
- Failed and unsafe results remain visible in shards, aggregation, videos, and report.
- Existing `reports/piperx_two_task_fixed_time/` artifacts are not overwritten.

---

### Task 1: Dataset and experiment contracts

**Files:**
- Create: `factory_bimanual/multitask_fixed_time_study.py`
- Test: `tests/factory_bimanual/test_multitask_fixed_time_study.py`

**Interfaces:**
- Produces: `discover_dual_hand_trajectories(root: Path) -> tuple[TrajectorySpec, ...]`, `StudyConfig`, `StudyShard`, `validate_shard(payload, spec, mode)`.
- Consumes: `TaskFamily`, `source_time_schedule`, `MOUNT_MODES`.

- [ ] **Step 1: Write failing inventory and shard tests**

```python
def test_repository_inventory_is_all_nonrejected_dual_hand_data():
    specs = discover_dual_hand_trajectories(ROOT / "data/factory")
    assert len(specs) == 27
    assert len({item.family.key for item in specs}) == 12
    assert all("_rejected_short" not in item.path.parts for item in specs)

def test_fixed_time_shard_rejects_substituted_timestamps():
    payload = valid_payload()
    payload["fixed_time_s"][2] += 0.01
    with pytest.raises(ValueError, match="source timestamps"):
        validate_shard(payload, SPEC, "upright_table")
```

- [ ] **Step 2: Verify RED**

Run: `E:\Anaconda\python.exe -m pytest -q tests\factory_bimanual\test_multitask_fixed_time_study.py`
Expected: FAIL because `factory_bimanual.multitask_fixed_time_study` does not exist.

- [ ] **Step 3: Implement immutable discovery and evidence validation**

Implement `TrajectorySpec` with `family`, `take`, `path`, and SHA-256; require dual-hand pose columns, strictly increasing timestamps, the four exact modes, exact timestamp identity, `retiming_applied is False`, matching frame arrays, and explicit collision/dynamics fields.

- [ ] **Step 4: Verify GREEN and commit**

Run the Task 1 test and commit `feat: add multitask fixed-time study contract`.

### Task 2: Resumable equal-budget mount search and full IK

**Files:**
- Create: `scripts/run_piperx_multitask_fixed_time_mount_study.py`
- Modify: `factory_bimanual/orientation_mount_search.py`
- Test: `tests/factory_bimanual/test_run_piperx_multitask_fixed_time_mount_study.py`

**Interfaces:**
- Consumes: `discover_dual_hand_trajectories`, `OrientationSearchConfig`, `generate_mounts`, `mount_quaternions`, strict paired IK and collision adapters.
- Produces: `plan_jobs(specs, modes)`, `rank_mount_result(record)`, `run_job(job, output_root)`, and resumable `study_manifest.json`.

- [ ] **Step 1: Write failing job-matrix and ranking tests**

```python
def test_job_matrix_contains_every_trajectory_mount_pair():
    jobs = plan_jobs(SPECS, STUDY_MODES)
    assert len(jobs) == 108
    assert {(j.spec.take, j.mode) for j in jobs} == expected_pairs

def test_collision_free_full_coverage_outranks_failed_mount():
    assert rank_mount_result(safe_full) < rank_mount_result(colliding_full)
    assert rank_mount_result(colliding_full) < rank_mount_result(partial)
```

- [ ] **Step 2: Verify RED**

Run the two new tests and require missing-symbol failures.

- [ ] **Step 3: Implement baseline replay and three equal-budget searches**

Reuse the task registration and calibrated TCP mapping from `scripts.run_piperx_recommended_v31`. For each task family and non-baseline mode, use the configured representative dual-hand take to run deterministic coarse geometry, first-frame anchor, sparse warm-start probe, and finalist screening. Reuse the selected family mount for the remaining takes, but solve and audit strict fixed-time IK independently on all 27 trajectories and all four modes. Preserve the best record even when no safe/full candidate exists; mark it `infeasible` with reason counts.

- [ ] **Step 4: Implement atomic checkpoints and resume hashes**

Checkpoint after every evaluated candidate. Refuse resume when source CSV, code/config fingerprint, calibration, robot model, or mount grid hashes differ.

- [ ] **Step 5: Verify GREEN and commit**

Run new tests plus existing orientation/mount/IK tests and commit `feat: run resumable multitask fixed-time mount search`.

### Task 3: Evidence bundle and validation

**Files:**
- Create: `scripts/build_piperx_multitask_fixed_time_bundle.py`
- Test: `tests/factory_bimanual/test_piperx_multitask_fixed_time_bundle.py`

**Interfaces:**
- Consumes: completed per-job checkpoints and `validate_shard`.
- Produces: per-mode summary JSON, trajectory NPZ, scene XML, provenance JSON, artifact hashes, and aggregate CSV/JSON.

- [ ] **Step 1: Write failing bundle tests**

```python
def test_bundle_requires_all_108_shards():
    with pytest.raises(ValueError, match="108"):
        validate_manifest(manifest_with_107_shards)

def test_aggregate_recomputes_accept_and_collision_counts():
    row = aggregate_shard(load_fixture_npz())
    assert row["both_accept_frames"] == int(np.count_nonzero(BOTH_ACCEPT))
    assert row["collision_frames"] == int(np.count_nonzero(COLLISION))
```

- [ ] **Step 2: Verify RED**

Run the Task 3 tests and require missing-module failure.

- [ ] **Step 3: Implement bundle builder and validator**

Use relative repository paths in published manifests, SHA-256 every artifact, and recompute all summary metrics from NPZ rather than trusting checkpoint totals. Add `--validate-only` and optional full MP4 decode.

- [ ] **Step 4: Verify GREEN and commit**

Run Task 1–3 tests and commit `feat: build validated multitask mount evidence bundle`.

### Task 4: Synchronized comparison videos and figures

**Files:**
- Create: `factory_bimanual/mount_comparison_visuals.py`
- Create: `scripts/render_piperx_multitask_mount_comparisons.py`
- Create: `scripts/plot_piperx_multitask_mount_results.py`
- Test: `tests/factory_bimanual/test_mount_comparison_visuals.py`

**Interfaces:**
- Consumes: validated 4-mode shards for one trajectory and aggregate CSV.
- Produces: one 2×2 synchronized MP4 per trajectory plus heatmap, winner-count, failure-window, collision, singularity, margin, and dynamics PNGs.

- [ ] **Step 1: Write failing timeline/layout/color tests**

```python
def test_four_panel_timeline_is_exact_source_domain():
    timeline = comparison_timeline([SOURCE_TIME] * 4, fps=30)
    assert timeline[0] == 0.0
    assert timeline[-1] <= SOURCE_TIME[-1]
    assert timeline[-1] + 1 / 30 > SOURCE_TIME[-1]

def test_mount_visual_semantics_are_stable():
    assert MOUNT_COLORS["upright_table"] == "#2F80ED"
    assert MOUNT_COLORS["horizontal_wall"] == "#F2994A"
    assert MOUNT_COLORS["inverted"] == "#9B51E0"
```

- [ ] **Step 2: Verify RED**

Run the visual test and require missing-module failure.

- [ ] **Step 3: Implement renderer and charts**

Render 1280×720 or 1920×1080 videos with equal world scale and view target. Every panel displays mount label/axis, source time, left/right/both ACCEPT, HOLD reason, maximum error, collision state, and target/actual trails. Charts use fixed ordering and denominators across tasks.

- [ ] **Step 4: Verify GREEN, decode representative videos, and commit**

Decode every frame of at least one short and one long comparison; inspect start/middle/end frames. Commit `feat: visualize four-mount fixed-time comparisons`.

### Task 5: Multi-task PDF, README, full run, and publication

**Files:**
- Create: `scripts/build_piperx_multitask_mount_report.py`
- Modify: `README.md`
- Modify: `.gitignore`
- Test: `tests/factory_bimanual/test_piperx_multitask_mount_report.py`

**Interfaces:**
- Consumes: validated study manifest, aggregate CSV, figures, and representative video frames.
- Produces: `reports/piperx_multitask_fixed_time_mount_study/PiperX多任务Fixed-Time四构型对比报告.pdf`.

- [ ] **Step 1: Write failing report-evidence tests**

```python
def test_report_rejects_missing_trajectory_or_mount():
    with pytest.raises(ValueError, match="complete 27 x 4 matrix"):
        validate_report_manifest(incomplete_manifest)

def test_report_numbers_equal_aggregate_rows():
    claims = build_report_claims(manifest)
    assert claims["total_trajectories"] == 27
    assert sum(claims["winner_counts"].values()) == 27
```

- [ ] **Step 2: Verify RED, then implement evidence-gated ReportLab report**

Generate overview pages, report-style heatmaps and winner distributions, family summaries, collision/dynamics audit, representative case studies, and one compact result page per trajectory. Every numeric claim comes from validated aggregates.

- [ ] **Step 3: Run the full 108-job experiment and render 27 comparison videos**

Use resume checkpoints continuously. Rerun failed jobs with the same published budget only when failure is an infrastructure error; do not silently increase budget for poor scientific results.

- [ ] **Step 4: Validate and visually inspect all deliverables**

Run full tests, `--validate-only`, decode all comparison MP4s, render every PDF page with Poppler, inspect overview and representative detail pages, and run `git diff --check`.

- [ ] **Step 5: Update README, commit, and push**

Document exact reproduction/validation commands, fixed-time meaning, dataset scope, headline metrics, limitations, PDF, videos, figures, manifests, and model provenance. Commit `feat: publish PiperX multitask fixed-time mount study` and push to `origin/main` only after local and remote commit hashes agree.
