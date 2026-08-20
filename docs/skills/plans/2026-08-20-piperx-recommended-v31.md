# PiperX recommended_v3_1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 PiperX 双臂增加严格 1 mm/0.5°、PDF 推荐 mount、warm-start/HOLD、受控翻腕和 rescue_v3.1 的可复现 MuJoCo 执行链，并生成真实视频、PDF 报告和可发布仓库。

**Architecture:** 纯 NumPy 模块负责协议和调度，MuJoCo 适配层负责真实 IK/碰撞/渲染，JSON 配置负责 12 个任务族共享底座。新入口与历史求解器隔离，旧结果只读保留。

**Tech Stack:** Python 3.11、NumPy、SciPy、MuJoCo 3.2+、pytest、ReportLab、Matplotlib、ffmpeg/ffprobe、GitHub CLI。

## Global Constraints

- ACCEPT: position error <= 0.001 m, orientation error <= 0.5 degree, per-joint shortest delta <= 0.30 rad.
- PiperX execution model: omega=1 rad/s, acceleration=4 rad/s^2, settle=0.1 s, decision=0.015 s.
- Frame 0 uses 40 deterministic restarts; later frames warm-start from the last commanded q.
- A failed frame holds the last command and never terminates the remaining source trajectory.
- FOLLOW cannot change wrist branch; only an audited rescue event may do so.
- Existing reports, selected visualizations, and selected videos remain available.
- Final Git history contains no secret and no file larger than 100 MB.

---

### Task 1: Recommended configuration and task-family identity

**Files:**
- Create: `configs/piperx_recommended_v31.json`
- Create: `factory_bimanual/task_family.py`
- Create: `factory_bimanual/piperx_recommended.py`
- Create: `tests/factory_bimanual/test_piperx_recommended_config.py`

**Interfaces:**
- Produces: `TaskFamily(date: str, task: str)`, `family_from_path(path, root)`, `load_recommended_config(path)`, `world_mount_for_family(config, family, registration_R, registration_t)`.
- Consumes: existing `RigidTaskRegistration` convention and mount-orientation names.

- [ ] **Step 1: Write the failing tests**

```python
def test_family_keeps_same_task_on_different_dates_separate(factory_root):
    a = family_from_path(factory_root / "8-11/Fold_Box/a.csv", factory_root)
    b = family_from_path(factory_root / "8-12/Fold_Box/b.csv", factory_root)
    assert a.key == "8-11/Fold_Box"
    assert b.key == "8-12/Fold_Box"
    assert a != b

def test_fold_box_pdf_base_is_registered_into_world(config):
    mount = world_mount_for_family(
        config, TaskFamily("8-11", "Fold_Box"), np.eye(3),
        np.array([-0.3995700068, 0.0200368514, 1.444496408]))
    assert mount.mode == "upright_table"
    assert np.allclose(mount.left_xyz_m, [-0.2495700068, 0.3200368514, 0.761496408])
    assert np.allclose(mount.right_xyz_m, [-0.3495700068, -0.2799631486, 0.761496408])

def test_strict_thresholds_are_not_relaxed(config):
    assert config.accept.position_tolerance_m == 0.001
    assert np.isclose(config.accept.orientation_tolerance_rad, np.deg2rad(0.5))
    assert config.accept.branch_guard_rad == 0.30
    assert config.anchor_restarts == 40
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/factory_bimanual/test_piperx_recommended_config.py -q`

Expected: collection/import failure because `task_family` and `piperx_recommended` do not exist.

- [ ] **Step 3: Implement the minimal contracts and full PDF mount table**

```python
@dataclass(frozen=True, order=True)
class TaskFamily:
    date: str
    task: str
    @property
    def key(self) -> str:
        return f"{self.date}/{self.task}"

def family_from_path(path: Path, root: Path) -> TaskFamily:
    relative = Path(path).resolve().relative_to(Path(root).resolve())
    if len(relative.parts) < 3:
        raise ValueError("factory path must be <date>/<task>/<file>")
    return TaskFamily(relative.parts[0], relative.parts[1])

def transform_base(source_xyz, rotation, translation):
    return np.asarray(rotation, float) @ np.asarray(source_xyz, float) + np.asarray(translation, float)
```

The JSON contains all 12 PiperX rows from PDF page 7, the four morphology names, funnel counts 150/12/3, probe stride 60, and the exact global constraints.

- [ ] **Step 4: Verify GREEN and regressions**

Run: `python -m pytest tests/factory_bimanual/test_piperx_recommended_config.py tests/factory_bimanual/test_factory_task_catalog.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add configs/piperx_recommended_v31.json factory_bimanual/task_family.py factory_bimanual/piperx_recommended.py tests/factory_bimanual/test_piperx_recommended_config.py
git commit -m "feat: add PDF-aligned PiperX mount protocol"
```

### Task 2: Strict gate, wrist guard, and execution timing

**Files:**
- Create: `factory_bimanual/rescue_v31.py`
- Create: `tests/factory_bimanual/test_rescue_v31.py`

**Interfaces:**
- Produces: `StrictGate.accepts(...)`, `shortest_joint_delta(...)`, `wrist_branch_signature(...)`, `trapezoidal_transition_time(...)`, `minimum_jerk_transition(...)`.
- Consumes: strict values from `PiperXRecommendedConfig`.

- [ ] **Step 1: Write exact-boundary failing tests**

```python
@pytest.mark.parametrize("pe,oe_deg,dq,expected", [
    (0.001000, 0.500, 0.300, True),
    (0.001001, 0.500, 0.300, False),
    (0.001000, 0.501, 0.300, False),
    (0.001000, 0.500, 0.301, False),
])
def test_strict_gate_boundaries(pe, oe_deg, dq, expected):
    gate = StrictGate(0.001, np.deg2rad(0.5), 0.30)
    assert gate.accepts(pe, np.deg2rad(oe_deg), np.array([dq, 0, 0, 0, 0, 0])) is expected

def test_minimum_jerk_hits_recovery_branch_with_zero_seam():
    q0 = np.zeros(6); q1 = np.array([.8, -.2, .1, 1.2, -1.1, .3])
    path = minimum_jerk_transition(q0, q1, 19)
    assert np.array_equal(path[-1], q1)
    assert np.max(np.abs(path[-1] - q1)) <= 1e-12
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/factory_bimanual/test_rescue_v31.py -q`

Expected: import failure for `rescue_v31`.

- [ ] **Step 3: Implement strict math**

```python
def shortest_joint_delta(new, old, periodic):
    delta = np.asarray(new, float) - np.asarray(old, float)
    delta = delta.copy()
    delta[np.asarray(periodic, bool)] = (delta[np.asarray(periodic, bool)] + np.pi) % (2*np.pi) - np.pi
    return delta

def trapezoidal_transition_time(delta_rad, velocity_rad_s=1.0, acceleration_rad_s2=4.0,
                                settle_s=0.1, decision_s=0.015):
    distance = np.max(np.abs(np.asarray(delta_rad, float)), initial=0.0)
    switch_distance = velocity_rad_s**2 / acceleration_rad_s2
    motion = (2*np.sqrt(distance/acceleration_rad_s2) if distance <= switch_distance
              else 2*velocity_rad_s/acceleration_rad_s2 + (distance-switch_distance)/velocity_rad_s)
    return motion + settle_s + decision_s

def minimum_jerk_transition(q0, q1, frames):
    u = np.linspace(0.0, 1.0, frames)
    s = 10*u**3 - 15*u**4 + 6*u**5
    result = np.asarray(q0)[None] + s[:, None]*(np.asarray(q1)-np.asarray(q0))[None]
    result[-1] = q1
    return result
```

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/factory_bimanual/test_rescue_v31.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add factory_bimanual/rescue_v31.py tests/factory_bimanual/test_rescue_v31.py
git commit -m "feat: add strict branch and wrist rescue primitives"
```

### Task 3: rescue_v3.1 state machine

**Files:**
- Modify: `factory_bimanual/rescue_v31.py`
- Modify: `tests/factory_bimanual/test_rescue_v31.py`

**Interfaces:**
- Produces: `CandidateFrame`, `RescueEvent`, `FollowSchedule`, `schedule_rescue_v31(layers, source_time_s, periodic, config, state_valid, transition_valid)`.
- Consumes: Task 2 strict math.

- [ ] **Step 1: Write failing protocol tests**

```python
def test_nonterminal_hold_resumes_from_last_command():
    layers = [layer(q=0.0), (), layer(q=0.1)]
    out = schedule_rescue_v31(layers, np.arange(3)/60, PERIODIC, cfg(dwell_frames=99))
    assert out.state.tolist() == ["FOLLOW", "HOLD", "FOLLOW"]
    assert np.allclose(out.command_q[1], out.command_q[0])

def test_mode_a_intercepts_future_target_and_counts_drops():
    layers = branch_loss_then_future_alternative()
    out = schedule_rescue_v31(layers, np.arange(len(layers))/60, PERIODIC, cfg())
    event = out.events[0]
    assert event.mode == "A"
    assert np.array_equal(out.command_q[event.end_frame], event.recovery_q)
    assert event.seam_error_rad <= 1e-12
    assert out.dropped_source_frames == event.transition_frames

def test_mode_b_preserves_source_frames_and_adds_cycle_delay():
    out = schedule_rescue_v31(mode_b_fixture(), np.arange(8)/60, PERIODIC, cfg())
    assert out.events[0].mode == "B"
    assert set(out.source_index) == set(range(8))
    assert out.cycle_delay_s > 0
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/factory_bimanual/test_rescue_v31.py -q`

Expected: missing scheduler symbols.

- [ ] **Step 3: Implement scheduler**

Implement deterministic candidate ordering by future strict run, transition time, wrist risk, joint margin, then branch index. FOLLOW uses only candidates inside the 0.30 rad guard and unchanged wrist signature. Rescue alone can cross signature, and every transition endpoint is overwritten with the exact recovery candidate.

- [ ] **Step 4: Verify GREEN and repeatability**

Run twice: `python -m pytest tests/factory_bimanual/test_rescue_v31.py -q`

Expected: identical passing output both times.

- [ ] **Step 5: Commit**

```bash
git add factory_bimanual/rescue_v31.py tests/factory_bimanual/test_rescue_v31.py
git commit -m "feat: implement rescue v3.1 branch scheduling"
```

### Task 4: MuJoCo warm-start DLS and bimanual safety adapter

**Files:**
- Create: `factory_bimanual/recommended_follow.py`
- Create: `tests/factory_bimanual/test_recommended_follow.py`
- Modify: `factory_bimanual/mujoco_candidate_generator.py`

**Interfaces:**
- Produces: `RecommendedFollowRunner(model, task, mapped_quaternions, config).run() -> RecommendedFollowResult`.
- Consumes: `MuJoCoCandidateGenerator`, `MuJoCoPairedCollisionChecker`, Task 3 scheduler, official prefixed joint names/TCP sites.

- [ ] **Step 1: Write failing adapter tests**

```python
def test_failed_frame_holds_and_later_frame_is_still_attempted(fake_adapter):
    result = fake_adapter.run(reference_outcomes=[q(0), None, q(.1)])
    assert result.state.tolist() == ["FOLLOW", "HOLD", "FOLLOW"]
    assert fake_adapter.attempted_rows == [0, 1, 2]

def test_every_reported_accept_meets_1mm_half_degree(result):
    accepted = result.accepted
    assert np.all(result.position_error_m[accepted] <= 0.001 + 1e-12)
    assert np.all(result.orientation_error_rad[accepted] <= np.deg2rad(0.5) + 1e-12)

def test_collision_rejected_candidate_becomes_hold(fake_adapter):
    result = fake_adapter.run(reference_outcomes=[q(0), colliding_q(.1)])
    assert result.state[1] == "HOLD"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/factory_bimanual/test_recommended_follow.py -q`

Expected: import failure for `recommended_follow`.

- [ ] **Step 3: Implement adapter and explicit 0.5° candidate config**

```python
candidate_config = CandidateGeneratorConfig(
    position_tolerance_m=config.accept.position_tolerance_m,
    orientation_tolerance_rad=config.accept.orientation_tolerance_rad,
    damping=config.dls.damping, max_iterations=config.dls.max_iterations,
    global_seed_count=config.anchor_restarts,
    stratified_seed_enabled=True, wrist_risk_enabled=True)
```

Frame 0 uses all 40 restarts. Later frames first call a warm-start-only DLS solve from the last command. Global/stratified candidates are generated only for rescue lookahead. Paired state and swept-edge collision callbacks gate every issued transition.

- [ ] **Step 4: Verify GREEN and MuJoCo smoke**

Run: `python -m pytest tests/factory_bimanual/test_recommended_follow.py tests/factory_bimanual/test_mujoco_candidate_generator.py tests/factory_bimanual/test_bimanual_collision.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add factory_bimanual/recommended_follow.py factory_bimanual/mujoco_candidate_generator.py tests/factory_bimanual/test_recommended_follow.py
git commit -m "feat: connect recommended follow to MuJoCo IK"
```

### Task 5: Reproducible CLI, artifacts, and real render

**Files:**
- Create: `scripts/run_piperx_recommended_v31.py`
- Create: `tests/factory_bimanual/test_run_piperx_recommended_v31.py`
- Modify: `factory_bimanual/video.py`

**Interfaces:**
- Produces: CLI arguments `--family`, `--source`, `--output-dir`, `--max-source-frames`, `--no-video`; artifacts `.summary.json`, `.trajectory.npz`, `.scene.xml`, `.mp4`, QA PNGs.
- Consumes: Tasks 1-4 and existing calibration/scene/video APIs.

- [ ] **Step 1: Write failing CLI/artifact tests**

```python
def test_cli_defaults_to_pdf_fold_box_validation(parse_args):
    args = parse_args([])
    assert args.family == "8-11/Fold_Box"
    assert args.output_dir.as_posix().endswith("reports/piperx_recommended_v31")

def test_summary_declares_strict_limits(summary):
    assert summary["acceptance"]["position_tolerance_mm"] == 1.0
    assert summary["acceptance"]["orientation_tolerance_deg"] == 0.5
    assert summary["acceptance"]["branch_guard_rad"] == 0.30
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/factory_bimanual/test_run_piperx_recommended_v31.py -q`

Expected: runner module is missing.

- [ ] **Step 3: Implement CLI and atomic artifacts**

The runner resolves the PDF source take, loads/registers the task, converts the PDF base into world coordinates, builds the official PiperX scene, applies locked TCP calibration, runs recommended follow, audits all states/edges, writes atomically, and renders with visible FOLLOW/HOLD/RESCUE state.

- [ ] **Step 4: Verify short integration then full real render**

Run: `python scripts/run_piperx_recommended_v31.py --max-source-frames 180 --output-dir tmp/piperx_v31_smoke`

Expected: exit 0, decoded MP4, no collision frames, every accepted frame within both strict thresholds.

Run: `python scripts/run_piperx_recommended_v31.py --output-dir reports/piperx_recommended_v31`

Expected: complete real task artifacts and decoded 1280x720 MP4.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_piperx_recommended_v31.py factory_bimanual/video.py tests/factory_bimanual/test_run_piperx_recommended_v31.py reports/piperx_recommended_v31
git commit -m "feat: render PiperX recommended v3.1 validation"
```

### Task 6: PDF report and visual QA

**Files:**
- Create: `scripts/build_piperx_recommended_v31_report.py`
- Create: `tests/test_build_piperx_recommended_v31_report.py`
- Create: `reports/piperx_recommended_v31/piperx_recommended_v31_optimization_report.pdf`

**Interfaces:**
- Produces: stable final PDF and figures under `reports/piperx_recommended_v31/figures/`.
- Consumes: source PDF, final summary/trajectory, QA frames.

- [ ] **Step 1: Write failing report tests**

```python
def test_report_inputs_reject_unverified_summary(tmp_path):
    with pytest.raises(ValueError, match="0.5"):
        validate_summary({"acceptance": {"orientation_tolerance_deg": 1.5}})

def test_report_pdf_has_expected_sections(pdf_path):
    text = "\n".join(page.extract_text() or "" for page in pdfplumber.open(pdf_path).pages)
    for heading in ("执行摘要", "安装方案", "IK 与翻腕", "rescue_v3.1", "真实渲染验证", "限制"):
        assert heading in text
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_build_piperx_recommended_v31_report.py -q`

Expected: report builder missing.

- [ ] **Step 3: Mark PDF creation and implement report builder**

Run exactly once before authoring: `node container_tools/mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 1 --output-format pdf`

Use ReportLab with a discovered CJK font, Matplotlib figures, page numbers, provenance table, strict-threshold table, event timeline, error percentiles, selected mount drawing and representative real-render frames.

- [ ] **Step 4: Generate, render every PDF page, and inspect**

Run: `python scripts/build_piperx_recommended_v31_report.py`

Run: `pdftoppm -png reports/piperx_recommended_v31/piperx_recommended_v31_optimization_report.pdf tmp/pdfs/piperx_v31/page`

Expected: extraction test passes and every rendered page has readable Chinese, no overlap/cutoff/black squares.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_piperx_recommended_v31_report.py tests/test_build_piperx_recommended_v31_report.py reports/piperx_recommended_v31
git commit -m "docs: add PiperX optimization report"
```

### Task 7: README, repository curation, verification, and GitHub publish

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Create: `docs/ARTIFACT_MANIFEST.md`
- Create: `LICENSE` only if repository publication policy requires it; otherwise omit.

**Interfaces:**
- Consumes: completed implementation and final artifacts.
- Produces: concise use-case-first README, tracked-file manifest, clean Git repository and GitHub remote.

- [ ] **Step 1: Run the README skill pipeline**

Run source analysis, CLEANUP/WRITE decision, draft/edit, critique, and verify against the completed code. The README opens with the recommended reproduction command, explains thresholds and evidence, then indexes legacy assets without catalog bloat.

- [ ] **Step 2: Curate Git tracking without deleting retained local evidence**

`.gitignore` excludes `data/**`, `third_party/**`, bulk `reports/**`, bulk `videos/**`, `.tmp/`, `tmp/`, `_codex_backup/`, caches, PID/log files, then re-includes the exact official PiperX model, validation source CSV, final report/video/summary/trajectory and selected QA images. `docs/ARTIFACT_MANIFEST.md` records what remains local-only and why.

- [ ] **Step 3: Run fresh verification**

Run: `python -m pytest -q`

Run: `python -m compileall factory_bimanual scripts tests`

Run: `ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate -show_entries format=duration -of json reports/piperx_recommended_v31/fold_box_8-11_v31.mp4`

Run: `git diff --check && git status --short && git ls-files -z | python scripts/audit_git_payload.py`

Expected: zero test failures, compile exit 0, video decodes as H.264 1280x720, no whitespace errors, no secret, no tracked file over 100 MB.

- [ ] **Step 4: Commit curated release**

```bash
git add README.md .gitignore docs/ARTIFACT_MANIFEST.md
git commit -m "docs: curate PiperX reproduction repository"
```

- [ ] **Step 5: Publish**

```bash
gh repo create yzh423/single-arm-mount-trajectory --public --source . --remote origin --push
```

If the repository already exists, add/verify `origin` and run `git push -u origin main`. Verify the remote branch and final commit URL with `gh repo view` and `git ls-remote origin refs/heads/main`.

## Plan self-review

- Spec coverage: all mount, IK, strict gate, wrist, rescue, video, PDF, README, preservation and GitHub requirements map to Tasks 1-7.
- Placeholder scan: no TBD/TODO/deferred implementation markers.
- Type consistency: the config feeds the pure scheduler and MuJoCo adapter; the runner is the sole artifact producer; the report and README consume verified artifacts.
- Execution choice: user explicitly requested uninterrupted direct execution, so use inline `executing-plans` with TDD checkpoints.
