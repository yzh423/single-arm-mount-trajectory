# Official Piper X Factory-Bimanual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the isolated factory-bimanual experiments with the pinned, native-scale AgileX Piper X model and recover strict IK solutions that the existing damped least-squares search misses near tolerance boundaries.

**Architecture:** Keep the authoritative Piper X URDF immutable under `third_party/`, load its audited native meshes in memory for MuJoCo, and point only the factory-bimanual contract at this source. Extend candidate generation with a deterministic constrained fallback that runs only after ordinary DLS returns no strict candidates, then invalidate scaled-model results and perform a complete native-scale mount selection before solving the trajectory.

**Tech Stack:** Python 3.11, MuJoCo 3.3, NumPy, SciPy SLSQP, pytest, URDF/XML, JSON/NPZ experiment artifacts.

## Global Constraints

- Official repository: `https://github.com/agilexrobotics/agx_arm_urdf`.
- Pinned revision: `f6642ce0d7872c686f29c99e9e10cd23d1d49313`.
- Robot subtree: `piper_x/`; the older `piper/` model is forbidden.
- Native model scale is exactly `1.0`; no reach normalization is allowed.
- TCP is official `ee_frame`; additional TCP offset is exactly `(0, 0, 0)`.
- Both bases are upright, fixed to the same tabletop, and have equal base height.
- Strict tracking remains `1 mm / 1.5 deg`; fallback must not relax it.
- Single-arm source, assets, reports, and solver behavior must remain unchanged.
- New outputs must not overwrite scaled legacy outputs.
- The current directory has no Git metadata; commit steps are blocked until a valid repository root is supplied or restored.

---

### Task 1: Audit and pin the existing official Piper X bridge

**Files:**
- Modify: `third_party/official_robot_models/piperx/SOURCE.md`
- Modify: `scripts/official_model_manifest.py`
- Modify: `tests/test_piperx_model_identity.py`
- Test: `tests/factory_bimanual/test_robot_contracts.py`
- Test: `tests/factory_bimanual/test_scene_builder.py`

**Interfaces:**
- Consumes: `factory_bimanual.robot_contracts.ROBOT_CONTRACTS["piperx"]` and `factory_bimanual.scene_builder.build_same_model_scene(...)`.
- Produces: a pinned native-scale model identity whose source path is `third_party/official_robot_models/piperx/PiperX.urdf`, TCP parent is `ee_frame`, TCP offset is zero, and upstream revision is machine-verifiable.

- [ ] **Step 1: Add failing provenance assertions**

```python
def test_piperx_provenance_pins_official_agx_arm_urdf_revision() -> None:
    manifest = OFFICIAL_MODELS["piperx"]
    assert manifest["repository"] == "https://github.com/agilexrobotics/agx_arm_urdf"
    assert manifest["revision"] == "f6642ce0d7872c686f29c99e9e10cd23d1d49313"
    assert manifest["source_subtree"] == "piper_x"
    assert manifest["normalization_scale"] == 1.0
```

- [ ] **Step 2: Run the identity test and verify RED**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -q tests/test_piperx_model_identity.py::test_piperx_provenance_pins_official_agx_arm_urdf_revision
```

Expected: FAIL because the current manifest has a workspace snapshot and no official repository, subtree, or native-scale fields.

- [ ] **Step 3: Pin official provenance without altering geometry**

Set the Piper X manifest entry to:

```python
"piperx": {
    "repository": "https://github.com/agilexrobotics/agx_arm_urdf",
    "revision": "f6642ce0d7872c686f29c99e9e10cd23d1d49313",
    "source_subtree": "piper_x",
    "variant": "PiPER-X native 6-DoF model + parallel gripper",
    "normalization_scale": 1.0,
    "source_artifact_sha256": "9d5b0490df5d3469fa08fae355d5dbda3761af5dace5624d49eda56896b72ecb",
},
```

Update `SOURCE.md` with the repository, revision, subtree, vendored and source hashes, native scale, and official `ee_frame` transform.

- [ ] **Step 4: Verify the bridge and real MuJoCo scene are GREEN**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -q tests/test_piperx_model_identity.py tests/factory_bimanual/test_robot_contracts.py tests/factory_bimanual/test_scene_builder.py
```

Expected: 29 or more tests PASS, including two native-scale Piper X arms, official joint ranges, equal-height upright bases, official TCP, and named collision geometry.

- [ ] **Step 5: Record the unavailable Git checkpoint**

Run `git rev-parse --show-toplevel`. Expected in the current environment: failure with `not a git repository`. Do not initialize a repository or commit unrelated user files.

---

### Task 2: Add deterministic acceptance-aware constrained IK fallback

**Files:**
- Modify: `factory_bimanual/mujoco_candidate_generator.py`
- Modify: `tests/factory_bimanual/test_mujoco_candidate_generator.py`

**Interfaces:**
- Consumes: `CandidateGeneratorConfig`, `_world_rotation_error(...)`, the existing deterministic seed bank, and `IKCandidate`.
- Produces: `MuJoCoCandidateGenerator._constrained_refine(side, seed, target_p, target_q) -> tuple | None`, invoked only when ordinary DLS yields zero strict candidates.

- [ ] **Step 1: Add a real regression reproducing the tolerance-boundary false negative**

Build a temporary scene from the retained scaled legacy Piper X asset with the historical mount `left=(-0.35, 0.25)`, `right=(-0.30, -0.45)`, yaw `15 deg`, base height `0.81 m`, and target:

```python
position = np.array([0.06312342461090609, -0.24703320367035672, 1.227163873])
quaternion = np.array([
    0.47757521388755036, 0.17889655332932233,
    0.7437187524778983, -0.43220406696359104,
])
```

Assert that `constrained_fallback_enabled=False` produces no candidate and `True` produces at least one candidate with position error `<= .001` and orientation error `<= deg2rad(1.5)`. Run twice and assert identical joint vectors.

- [ ] **Step 2: Run the regression and verify RED**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -q tests/factory_bimanual/test_mujoco_candidate_generator.py -k constrained_fallback
```

Expected: FAIL because `CandidateGeneratorConfig` has no constrained fallback and ordinary DLS returns zero candidates at this target.

- [ ] **Step 3: Extend the configuration minimally**

Add:

```python
constrained_fallback_enabled: bool = True
constrained_fallback_seed_count: int = 4
constrained_fallback_max_iterations: int = 180
```

Reject negative seed counts and iteration counts below one.

- [ ] **Step 4: Preserve near-miss endpoints from DLS**

Refactor the internal iteration so each seed returns its terminal joint vector and measured position/orientation residual even when it misses strict acceptance. Keep `_solve(...)` behavior unchanged for existing callers and tests.

- [ ] **Step 5: Implement the constrained refinement**

Use `scipy.optimize.minimize(..., method="SLSQP")` with native joint bounds. Minimize position error while enforcing:

```python
orientation_tolerance_rad - orientation_error(q) >= 0
```

Then accept only after independently recomputing both strict inequalities. Start from the four deterministic near-miss endpoints with the smallest normalized residual. Return standard `IKCandidate` metrics, including joint-limit margin and Jacobian smallest singular value. Never mutate the live `MjData`, collision masks, task, or tolerance configuration.

- [ ] **Step 6: Integrate fallback only after ordinary DLS returns no candidate**

The call order must be:

```python
solved, near_misses = self._ordinary_candidates(...)
if not solved and self.config.constrained_fallback_enabled:
    solved = self._fallback_candidates(near_misses, ...)
```

Apply existing deduplication and deterministic sorting to fallback results. Collision filtering remains outside candidate generation.

- [ ] **Step 7: Verify the fallback regression and existing generator tests**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -q tests/factory_bimanual/test_mujoco_candidate_generator.py
```

Expected: all tests PASS; the regression recovers the strict target without relaxed tolerances and remains deterministic.

- [ ] **Step 8: Verify the complete factory-bimanual suite**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -q tests/factory_bimanual
```

Expected: all 167 or more collected tests PASS.

---

### Task 3: Protect single-arm state and invalidate scaled factory results

**Files:**
- Create: `reports/factory_bimanual/seal_bag_dual_piperx/official_native_scale_preflight.json` (generated artifact)
- Modify: `tests/factory_bimanual/test_seal_bag_right_mount_search.py`
- Inspect only: `reports/single_arm/`

**Interfaces:**
- Consumes: source-URDF hash, scene manifest, solver configuration, and mount-search fingerprinting.
- Produces: a native-scale preflight record whose fingerprint cannot resume a scaled-model job.

- [ ] **Step 1: Add a failing fingerprint test**

Assert two otherwise identical jobs produce different fingerprints when their model source hash, TCP parent, TCP offset, or normalization scale differs. The native record must contain:

```json
{
  "model_identity": "agilex_piper_x",
  "upstream_revision": "f6642ce0d7872c686f29c99e9e10cd23d1d49313",
  "normalization_scale": 1.0,
  "tcp_frame": "ee_frame",
  "tcp_offset_m": [0.0, 0.0, 0.0]
}
```

- [ ] **Step 2: Run and verify RED if any identity field is absent from the fingerprint**

Run the single test with `python -m pytest -q ...`; expected failure must identify the missing fingerprint distinction, not an import error.

- [ ] **Step 3: Add only the missing identity fields to the existing fingerprint payload**

Do not rename or delete legacy output files. Use a new output stem containing `official_native_scale`.

- [ ] **Step 4: Capture and compare protected single-arm hashes**

Hash all files under `reports/single_arm/` and protected single-arm source paths before and after the factory work. Expected: byte-for-byte equality; any difference blocks the experiment.

- [ ] **Step 5: Run the preflight**

Compile a real official Piper X two-arm scene and write the preflight JSON under `reports/factory_bimanual/`. Assert no initial penetration, equal base height, upright base axes, official model identity, and source fingerprint.

---

### Task 4: Recompute complete official Piper X mount selection and trajectory

**Files:**
- Generate: `reports/factory_bimanual/seal_bag_dual_piperx/official_native_scale_mount_search.json`
- Generate: `reports/factory_bimanual/seal_bag_dual_piperx/*official_native_scale*.scene.xml`
- Generate: `reports/factory_bimanual/seal_bag_dual_piperx/*official_native_scale*.trajectory.npz`
- Generate: `reports/factory_bimanual/seal_bag_dual_piperx/*official_native_scale*.summary.json`

**Interfaces:**
- Consumes: official native-scale contract, both complete factory trajectories, strict candidate generator plus fallback, paired collision checker, and source-time velocity gate.
- Produces: a whole-trajectory-ranked upright equal-height mount and a complete source-row-aligned seal-bag result.

- [ ] **Step 1: Run the complete upright tabletop mount enumeration**

Search all permitted right and left table positions and yaw values using the existing coarse-grid and local-refinement policy. Do not search tilt, roll, or pitch. Evaluate both task CSVs on identical candidate sets.

- [ ] **Step 2: Rank by the established lexicographic safety objective**

Rank by minimum task coverage, longest contiguous failure, cross-arm collisions, table/base collisions, TCP error, singularity margin, joint-limit margin, and then smaller mount displacement. Do not optimize only the `1.4-2.0 s` seal-bag interval.

- [ ] **Step 3: Validate the selected mount on every source row**

Require both arms to be attached at the same height and collision-free at initialization. Record every raw candidate count, constrained-fallback count, collision-free count, velocity violation, recovery mode, and failure reason.

- [ ] **Step 4: Solve the complete seal-bag trajectory**

Use strict `1 mm / 1.5 deg` acceptance and the existing `3 rad/s` controller limit. Preserve source timestamps. Report source-time strict tracking separately from any explicit retiming or bounded recovery.

- [ ] **Step 5: Audit the opening interval**

For `1.4-2.0 s`, report which frames are statically infeasible, DLS-recovered by the constrained fallback, source-time edge infeasible, collision rejected, or successfully tracked. Confirm that `CANNOT FOLLOW` is never emitted merely because ordinary DLS missed a tolerance-boundary solution.

- [ ] **Step 6: Compare only against labeled scaled legacy evidence**

The final comparison must state that the old run used scale `0.9489617327844065`, `gripper_base + 0.13 m`, and a different source fingerprint. Do not claim an algorithmic improvement from a geometry change alone.

- [ ] **Step 7: Run final verification**

Run the complete factory-bimanual pytest suite, compile the selected scene, decode/load all JSON and NPZ outputs, verify row counts and timestamps, and recheck protected single-arm hashes. Any failure blocks completion.

## Plan self-review

- Spec coverage: official identity, pinned revision, native scale, official TCP, isolation, equal-height tabletop mounting, fallback IK, complete mount selection, reporting, and single-arm protection are each mapped to a task.
- Placeholder scan: no `TBD`, `TODO`, or deferred implementation language remains.
- Type consistency: `CandidateGeneratorConfig`, `MuJoCoCandidateGenerator`, `IKCandidate`, `ROBOT_CONTRACTS`, and `build_same_model_scene` match current project interfaces.
- Repository limitation: all code and tests can be executed, but no commit can be truthfully created in the current non-Git directory.
