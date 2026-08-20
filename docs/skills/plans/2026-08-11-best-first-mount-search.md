# Best-first Mount Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed 218-candidate strict mount funnel with a deterministic best-first search that ranks cheaply, evaluates exactly 64 local candidates, reuses IK work across candidates and fidelity levels, and performs bounded post-success exploration before final strict replay.

**Architecture:** Add a focused `best_first_mount_search` module containing budgets, candidate state, representative-frame selection, region generation, frontier ordering, and stop decisions. Keep MuJoCo/IK execution in `search_strict_urdf_mount.py`, exposed through an incremental evaluator that caches frame results and accepts parent seeds. Integrate behind an explicit search-policy CLI switch, retain the legacy path for paired regression, and split search/solve/render fingerprints so unrelated changes do not invalidate expensive search state.

**Tech Stack:** Python 3, NumPy, SciPy Sobol sampling, MuJoCo Python bindings, pytest, JSON/NPZ checkpoints.

## Global Constraints

- Preserve the existing task success thresholds, collision rules, joint limits, and final formal replay protocol.
- Default Global budget is 128 candidates.
- Default Local budget is exactly 64 candidates: 8 regions × 8 candidates.
- Normal Dense promotion budget is at most 3 candidates; fallback permits at most 5 additional Dense candidates.
- After the first full success, expand at most 8 additional frontier nodes and spend at most 25% of pre-success shard time.
- Final formal replay evaluates 1-2 candidates over every trajectory frame with the existing 16-restart budget.
- Never claim mathematical global optimality; report only budget-bounded near-global evidence.
- Preserve a `legacy` search policy until paired ablation passes.
- Use fixed seeds and deterministic tie-breaking for all tests and formal runs.
- Do not add new runtime dependencies.
- The current checkout has no `.git` directory; execute every verification checkpoint, but skip commit commands unless the implementation is moved into a Git worktree.

---

## File Structure

- Create `design_optimization/best_first_mount_search.py`: pure search policy, state types, frame selection, region selection, frontier, promotion and stopping decisions.
- Create `design_optimization/incremental_mount_evaluator.py`: frame-result cache, seed selection and incremental fidelity evaluation interfaces.
- Create `scripts/mount_search_telemetry.py`: aggregation-only stage timings and IK counters.
- Modify `scripts/search_strict_urdf_mount.py`: connect MuJoCo/IK execution to the new policy while preserving legacy mode.
- Modify `scripts/run_twelve_arm_two_single_tasks.py`: policy CLI, split fingerprints, resumable search-state paths and result metadata.
- Modify `scripts/formal_run_acceleration.py`: typed fingerprint helpers and atomic JSON checkpoint utilities.
- Create `tests/test_best_first_mount_search.py`: deterministic pure-policy unit tests.
- Create `tests/test_incremental_mount_evaluator.py`: no-recomputation and parent-seed tests with a fake solver.
- Create `tests/test_mount_search_telemetry.py`: counter and elapsed-time aggregation tests.
- Create `tests/test_best_first_search_integration.py`: small-budget real MuJoCo integration and legacy comparison.
- Modify `tests/test_formal_run_acceleration.py`: independent fingerprint invalidation tests.

---

### Task 1: Establish Stage-Level Performance Telemetry

**Files:**
- Create: `scripts/mount_search_telemetry.py`
- Modify: `scripts/search_strict_urdf_mount.py`
- Test: `tests/test_mount_search_telemetry.py`

**Interfaces:**
- Produces: `SearchTelemetry`, `SearchTelemetry.stage(name)`, `record_candidate(...)`, `record_ik(...)`, and `to_dict()`.
- Consumed by: Tasks 5 and 8 for result JSON and performance gates.

- [ ] **Step 1: Write failing aggregation tests**

```python
from scripts.mount_search_telemetry import SearchTelemetry


def test_telemetry_aggregates_candidates_frames_restarts_and_iterations():
    telemetry = SearchTelemetry()
    telemetry.record_candidate("rank", outcome="evaluated", new_frames=16, reused_frames=0)
    telemetry.record_candidate("medium", outcome="promoted", new_frames=48, reused_frames=16)
    telemetry.record_ik("medium", first_frame=True, restarts=2, iterations=73, success=True)

    payload = telemetry.to_dict()
    assert payload["stages"]["rank"]["evaluated_candidates"] == 1
    assert payload["stages"]["medium"]["promoted_candidates"] == 1
    assert payload["stages"]["medium"]["new_frames"] == 48
    assert payload["stages"]["medium"]["reused_frames"] == 16
    assert payload["stages"]["medium"]["first_frame_restart_histogram"]["1-2"] == 1
    assert payload["stages"]["medium"]["dls_iterations"] == 73


def test_stage_context_records_monotonic_elapsed_time(monkeypatch):
    ticks = iter((10.0, 12.5))
    monkeypatch.setattr("scripts.mount_search_telemetry.time.perf_counter", lambda: next(ticks))
    telemetry = SearchTelemetry()
    with telemetry.stage("selection"):
        pass
    assert telemetry.to_dict()["stages"]["selection"]["elapsed_s"] == 2.5
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `python -m pytest tests/test_mount_search_telemetry.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'scripts.mount_search_telemetry'`.

- [ ] **Step 3: Implement the aggregation-only telemetry module**

Implement dataclasses with these exact signatures:

```python
@dataclass
class StageTelemetry:
    elapsed_s: float = 0.0
    evaluated_candidates: int = 0
    promoted_candidates: int = 0
    rejected_candidates: int = 0
    new_frames: int = 0
    reused_frames: int = 0
    solve_pose_calls: int = 0
    dls_iterations: int = 0
    first_frame_restart_histogram: dict[str, int] = field(default_factory=lambda: {
        "0": 0, "1-2": 0, "3-5": 0, "6-10": 0,
        "11-20": 0, "21-40": 0, "failed": 0,
    })


class SearchTelemetry:
    def stage(self, name: str) -> ContextManager[None]: ...
    def record_candidate(
        self, stage: str, *, outcome: Literal["evaluated", "promoted", "rejected"],
        new_frames: int, reused_frames: int,
    ) -> None: ...
    def record_ik(
        self, stage: str, *, first_frame: bool, restarts: int,
        iterations: int, success: bool,
    ) -> None: ...
    def to_dict(self) -> dict[str, object]: ...
```

Reject negative counters with `ValueError`. The context manager must use `time.perf_counter()` and accumulate repeated entries under the same stage.

- [ ] **Step 4: Run telemetry tests**

Run: `python -m pytest tests/test_mount_search_telemetry.py -v`

Expected: all tests pass.

- [ ] **Step 5: Add legacy-path instrumentation without changing decisions**

Wrap the current global/local/medium/strict/micro/final calls in named telemetry stages. Record candidate/frame counts already available at each boundary. Add `"telemetry": telemetry.to_dict()` to the output audit JSON. Do not change stage budgets or IK calls in this step.

- [ ] **Step 6: Verify the original compact smoke still produces telemetry**

Run:

```powershell
python scripts/search_strict_urdf_mount.py --domain local --robot xarm6 --task cap-left --candidates 2 --screen-frames 2 --medium-frames 2 --output tmp/telemetry_smoke.npz
```

Expected: command may return 0 or 2 according to task success, but `tmp/telemetry_smoke.json` exists and contains non-negative elapsed/candidate/frame counters for every executed stage.

- [ ] **Step 7: Checkpoint**

Run: `python -m pytest tests/test_mount_search_telemetry.py tests/test_hierarchical_mount_search.py -q`

Expected: pass. Record the test output in the implementation log; no Git commit is possible in this checkout.

---

### Task 2: Implement Deterministic Candidate State and Frontier Ordering

**Files:**
- Create: `design_optimization/best_first_mount_search.py`
- Test: `tests/test_best_first_mount_search.py`

**Interfaces:**
- Produces: `SearchBudget`, `Fidelity`, `CandidateState`, `Frontier`, `promotion_target()`, and `should_stop_after_success()`.
- Consumed by: Tasks 3, 5 and 6.

- [ ] **Step 1: Write failing state and frontier tests**

```python
import numpy as np
from design_optimization.best_first_mount_search import (
    CandidateState, Fidelity, Frontier, SearchBudget,
    promotion_target, should_stop_after_success,
)


def candidate(identifier: int, optimistic: tuple, region: int = 0) -> CandidateState:
    return CandidateState(
        candidate_id=identifier,
        mount=np.zeros(6),
        parent_id=None,
        region_id=region,
        fidelity=Fidelity.RANK,
        score=(0.0,),
        optimistic_score=optimistic,
    )


def test_frontier_uses_optimistic_score_then_candidate_id():
    frontier = Frontier()
    frontier.push(candidate(2, (0.8,)))
    frontier.push(candidate(1, (0.8,)))
    frontier.push(candidate(3, (0.9,)))
    assert [frontier.pop().candidate_id for _ in range(3)] == [3, 1, 2]


def test_fidelity_promotion_is_monotonic():
    assert promotion_target(Fidelity.RANK) is Fidelity.MEDIUM
    assert promotion_target(Fidelity.MEDIUM) is Fidelity.DENSE
    assert promotion_target(Fidelity.DENSE) is Fidelity.FINAL
    assert promotion_target(Fidelity.FINAL) is None


def test_post_success_hard_budget_stops_at_eight_expansions():
    budget = SearchBudget()
    assert should_stop_after_success(
        budget=budget, expansions=8, pre_success_elapsed_s=100.0,
        post_success_elapsed_s=10.0, verified_regions=2,
        non_improving_expansions=1, frontier_can_improve=True,
    ) == "expansion_budget"
```

- [ ] **Step 2: Run the focused tests**

Run: `python -m pytest tests/test_best_first_mount_search.py -v`

Expected: missing-module failure.

- [ ] **Step 3: Implement immutable budgets and mutable candidate state**

Use these exact definitions:

```python
class Fidelity(enum.IntEnum):
    GATE = 0
    RANK = 1
    MEDIUM = 2
    DENSE = 3
    FINAL = 4


@dataclass(frozen=True)
class SearchBudget:
    global_candidates: int = 128
    retained_regions: int = 8
    local_per_region: int = 8
    rank_frames: int = 24
    medium_frames: int = 64
    dense_before_success: int = 3
    dense_fallback: int = 5
    post_success_expansions: int = 8
    post_success_time_fraction: float = 0.25
    post_success_regions: int = 3
    non_improving_limit: int = 4
    final_candidates: int = 2

    def __post_init__(self) -> None:
        if self.retained_regions * self.local_per_region != 64:
            raise ValueError("default local population must contain exactly 64 candidates")


@dataclass
class CandidateState:
    candidate_id: int
    mount: np.ndarray
    parent_id: int | None
    region_id: int
    fidelity: Fidelity
    score: tuple[float, ...]
    optimistic_score: tuple[float, ...]
    closed: bool = False
```

`Frontier` must wrap `heapq`, negate numeric tuple values for max-priority behavior, and append `candidate_id` as an ascending deterministic tie-break.

- [ ] **Step 4: Implement explicit stop reasons**

`should_stop_after_success(...) -> str | None` must return one of:

```text
expansion_budget
time_budget
frontier_empty
frontier_dominated
stable_multi_region
```

The hard limits take precedence. `stable_multi_region` requires at least 3 regions, 4 non-improving expansions, and `frontier_can_improve=False`.

- [ ] **Step 5: Run policy tests**

Run: `python -m pytest tests/test_best_first_mount_search.py -v`

Expected: pass.

- [ ] **Step 6: Checkpoint**

Run: `python -m pytest tests/test_best_first_mount_search.py tests/test_search_policy.py -q`

Expected: pass. Save output; skip Git commit in the non-Git checkout.

---

### Task 3: Select Representative Frames and Generate Exactly 64 Local Candidates

**Files:**
- Modify: `design_optimization/best_first_mount_search.py`
- Test: `tests/test_best_first_mount_search.py`

**Interfaces:**
- Produces: `representative_frame_indices(...)`, `select_diverse_regions(...)`, and `generate_local_population(...)`.
- Consumes: `SearchBudget` and existing `local_mount_candidates`/installation bounds.
- Consumed by: Task 5.

- [ ] **Step 1: Add failing representative-frame tests**

```python
def test_representative_frames_include_ends_and_motion_peaks():
    positions = np.c_[np.arange(100), np.zeros(100), np.zeros(100)].astype(float)
    positions[50, 1] = 10.0
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (100, 1))
    indices = representative_frame_indices(positions, quaternions, requested=24)
    assert len(indices) == 24
    assert indices[0] == 0
    assert indices[-1] == 99
    assert 50 in indices
    assert np.all(np.diff(indices) > 0)


def test_local_population_is_eight_regions_times_eight_candidates():
    centers = np.arange(48, dtype=float).reshape(8, 6)
    lower = np.full(6, -100.0)
    upper = np.full(6, 100.0)
    population, region_ids, parent_ids = generate_local_population(
        centers, lower, upper, per_region=8, seed=751,
    )
    assert population.shape == (64, 6)
    assert np.bincount(region_ids).tolist() == [8] * 8
    assert np.array_equal(population[::8], centers)
    assert np.array_equal(parent_ids[::8], np.arange(8))
```

- [ ] **Step 2: Run focused tests and observe missing symbols**

Run: `python -m pytest tests/test_best_first_mount_search.py -v`

Expected: import or name failures for the new functions.

- [ ] **Step 3: Implement representative frames**

Implement:

```python
def representative_frame_indices(
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    *,
    requested: int,
) -> np.ndarray:
```

Requirements:

- Validate equal non-zero lengths and `requested >= 2`.
- Always include first and last frame.
- Compute translation arc-length increments.
- Compute quaternion angular increments using `2*arccos(abs(dot(q[i-1], q[i])))`.
- Reserve up to one quarter of the budget for largest combined motion-change peaks.
- Fill remaining slots with arc-length quantiles; use uniform indices only when total arc length is zero.
- Deduplicate and deterministically fill any shortfall with lowest unused uniform indices.

- [ ] **Step 4: Implement diverse regions and local population**

Implement:

```python
def select_diverse_regions(
    mounts: np.ndarray,
    scores: Sequence[tuple[float, ...]],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    count: int = 8,
    shortlist: int = 24,
) -> np.ndarray:

def generate_local_population(
    centers: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    per_region: int = 8,
    seed: int = 751,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
```

Normalize all six mount dimensions by `(upper-lower)`. Select the highest-ranked center first, then greedily maximize minimum normalized distance with rank as tie-break. Preserve each center as the first candidate of its 8-item region; generate exactly seven deterministic Sobol perturbations for the rest.

- [ ] **Step 5: Run policy tests**

Run: `python -m pytest tests/test_best_first_mount_search.py -v`

Expected: pass, including exact 64-candidate assertions.

- [ ] **Step 6: Checkpoint**

Run: `python -m pytest tests/test_best_first_mount_search.py tests/test_hierarchical_mount_search.py -q`

Expected: pass.

---

### Task 4: Implement Incremental Frame Evaluation and Parent-Seed Reuse

**Files:**
- Create: `design_optimization/incremental_mount_evaluator.py`
- Test: `tests/test_incremental_mount_evaluator.py`

**Interfaces:**
- Produces: `FrameResult`, `CandidateEvaluation`, `IncrementalMountEvaluator.evaluate(...)`, `nearest_seed(...)`.
- Consumes: a solver callback supplied by `search_strict_urdf_mount.py`; no MuJoCo import in this module.
- Consumed by: Task 5.

- [ ] **Step 1: Write failing tests with a counting fake solver**

```python
import numpy as np
from design_optimization.incremental_mount_evaluator import IncrementalMountEvaluator


class CountingSolver:
    def __init__(self):
        self.calls = []

    def __call__(self, frame_index, seed_q, restart_limit):
        self.calls.append((frame_index, None if seed_q is None else tuple(seed_q), restart_limit))
        return {
            "q": np.array([float(frame_index)]),
            "success": True,
            "position_error_m": 0.0,
            "orientation_error_rad": 0.0,
            "table_collision": False,
            "self_collision": False,
            "restarts": 0,
            "iterations": 1,
        }


def test_promotion_only_solves_new_frames():
    solver = CountingSolver()
    evaluator = IncrementalMountEvaluator(frame_count=8)
    evaluator.evaluate(candidate_id=1, frame_indices=np.array([0, 4, 7]), solver=solver, restart_limit=2)
    evaluator.evaluate(candidate_id=1, frame_indices=np.arange(8), solver=solver, restart_limit=3)
    assert [call[0] for call in solver.calls] == [0, 4, 7, 1, 2, 3, 5, 6]


def test_child_uses_parent_frame_solution_as_first_seed():
    solver = CountingSolver()
    evaluator = IncrementalMountEvaluator(frame_count=4)
    evaluator.evaluate(candidate_id=1, frame_indices=np.array([0, 2]), solver=solver, restart_limit=2)
    evaluator.evaluate(
        candidate_id=2, parent_id=1, frame_indices=np.array([0, 2]),
        solver=solver, restart_limit=2,
    )
    child_calls = solver.calls[-2:]
    assert child_calls[0][1] == (0.0,)
    assert child_calls[1][1] == (2.0,)
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `python -m pytest tests/test_incremental_mount_evaluator.py -v`

Expected: missing-module failure.

- [ ] **Step 3: Implement typed frame results and candidate cache**

```python
@dataclass(frozen=True)
class FrameResult:
    q: np.ndarray
    success: bool
    position_error_m: float
    orientation_error_rad: float
    table_collision: bool
    self_collision: bool
    restarts: int
    iterations: int


@dataclass
class CandidateEvaluation:
    candidate_id: int
    frames: dict[int, FrameResult] = field(default_factory=dict)


Solver = Callable[[int, np.ndarray | None, int], Mapping[str, object]]
```

`IncrementalMountEvaluator` owns `dict[int, CandidateEvaluation]`. Repeated requests for the same candidate/frame return the cached result and never call the solver.

- [ ] **Step 4: Implement deterministic seed precedence**

For every new frame, choose seeds in this order:

1. Same-frame parent result if successful.
2. Closest successful frame already solved for the current candidate; break equal distances toward the lower frame index.
3. Closest successful parent frame.
4. `None`, allowing the solver adapter to use its formal default.

The evaluator must not copy parent metrics to the child. Every requested child frame calls the solver and receives fresh FK/error/collision results.

- [ ] **Step 5: Implement interval-friendly evaluation order**

When adding frames during `medium → dense`, solve missing indices by increasing distance from an already successful index. If no success exists, solve ascending indices. This lets successful solutions warm-start neighboring frames without changing the requested frame set.

- [ ] **Step 6: Run incremental-evaluator tests**

Run: `python -m pytest tests/test_incremental_mount_evaluator.py -v`

Expected: pass with exact fake-solver call counts.

- [ ] **Step 7: Checkpoint**

Run: `python -m pytest tests/test_incremental_mount_evaluator.py tests/test_best_first_mount_search.py -q`

Expected: pass.

---

### Task 5: Integrate Best-first Search Behind an Explicit Policy Switch

**Files:**
- Modify: `scripts/search_strict_urdf_mount.py`
- Modify: `design_optimization/best_first_mount_search.py`
- Modify: `design_optimization/incremental_mount_evaluator.py`
- Test: `tests/test_best_first_search_integration.py`

**Interfaces:**
- Consumes: Tasks 1-4 interfaces.
- Produces: CLI `--search-policy {legacy,best-first}`, best-first audit fields, and a small-budget callable `run_best_first_search(...)`.
- Consumed by: Tasks 6-8 and the formal runner.

- [ ] **Step 1: Add failing parser and small-budget integration tests**

```python
from scripts.search_strict_urdf_mount import build_parser, resolved_best_first_budget


def test_best_first_cli_defaults():
    args = build_parser().parse_args([
        "--domain", "local", "--robot", "xarm6", "--task", "cap-left",
        "--search-policy", "best-first",
    ])
    assert args.search_policy == "best-first"
    budget = resolved_best_first_budget(args)
    assert budget.global_candidates == 128
    assert budget.retained_regions * budget.local_per_region == 64
    assert budget.post_success_expansions == 8


def test_legacy_policy_remains_selectable():
    args = build_parser().parse_args([
        "--domain", "local", "--robot", "xarm6", "--task", "cap-left",
        "--search-policy", "legacy",
    ])
    assert args.search_policy == "legacy"


def test_legacy_remains_default_until_ablation_passes():
    args = build_parser().parse_args([
        "--domain", "local", "--robot", "xarm6", "--task", "cap-left",
    ])
    assert args.search_policy == "legacy"
```

- [ ] **Step 2: Run focused tests and verify parser failure**

Run: `python -m pytest tests/test_best_first_search_integration.py -v`

Expected: failure because `--search-policy` and `resolved_best_first_budget` do not exist.

- [ ] **Step 3: Add policy and budget CLI arguments**

Add:

```python
parser.add_argument("--search-policy", choices=("legacy", "best-first"), default="legacy")
parser.add_argument("--global-candidates", type=int, default=128)
parser.add_argument("--rank-frames", type=int, default=24)
parser.add_argument("--medium-frames", type=int, default=64)
parser.add_argument("--regions", type=int, default=8)
parser.add_argument("--local-per-region", type=int, default=8)
parser.add_argument("--dense-before-success", type=int, default=3)
parser.add_argument("--dense-fallback", type=int, default=5)
parser.add_argument("--post-success-expansions", type=int, default=8)
parser.add_argument("--final-candidates", type=int, default=2)
```

Keep deprecated `--candidates` behavior only for legacy/smoke compatibility. Reject a best-first budget unless `regions * local_per_region == 64`.

- [ ] **Step 4: Extract the current implementation into `run_legacy_search(...)`**

Move the current fixed-stage body without semantic changes into:

```python
def run_legacy_search(args: argparse.Namespace, context: SearchContext) -> SearchOutcome:
```

Define `SearchContext` to carry targets, quaternions, time, model entry, bounds, warm starts and output paths. Define `SearchOutcome` to carry selected mount, q-path, metrics, audit fields and telemetry. `main()` should only parse/build context, dispatch policy and serialize outcome.

- [ ] **Step 5: Implement the real solver adapter**

For one candidate, build or obtain its model once. The per-frame callback passed to `IncrementalMountEvaluator` must:

- accept the chosen seed and stage restart limit;
- solve the requested target pose;
- execute FK/error and collision checks for the child candidate;
- return the exact `FrameResult` fields;
- update `SearchTelemetry` with actual restart and iteration counters exposed by the IK result.

If existing IK results do not expose total iterations/restarts, extend their result dataclass with default-zero counters and update the solver loops to accumulate them; add assertions to existing strict IK unit tests.

- [ ] **Step 6: Implement `run_best_first_search(...)`**

The implementation must follow this exact control flow:

```text
Global 128 → geometry gate → rank representative frames
→ retain 8 diverse regions → generate/evaluate 64 Local
→ unified priority frontier
→ promote best node one fidelity at a time
→ Dense up to 3 nodes before first success
→ if none succeeds, Dense fallback up to 5 more
→ after first success, expand up to 8 frontier nodes under time/region/stability rules
→ add legacy incumbent to finalist set
→ Final formal replay for best 1-2 unique mounts
```

Every promotion must update `score`, `optimistic_score`, telemetry and incumbent history. The legacy incumbent must always be eligible for Final even when it was not selected by the new frontier.

- [ ] **Step 7: Add explicit audit fields**

Include:

```json
{
  "search_policy": "best-first",
  "near_global_claim": "budget_bounded_no_better_candidate_observed",
  "stop_reason": "...",
  "first_success_elapsed_s": 0.0,
  "post_success_elapsed_s": 0.0,
  "post_success_expansions": 0,
  "verified_region_count": 0,
  "frontier_best_optimistic_score": [],
  "incumbent_history": [],
  "search_budget": {},
  "new_frame_solves": 0,
  "reused_frame_results": 0,
  "dense_candidate_count": 0,
  "fallback_used": false,
  "telemetry": {}
}
```

- [ ] **Step 8: Run parser/pure integration tests**

Run: `python -m pytest tests/test_best_first_search_integration.py tests/test_best_first_mount_search.py tests/test_incremental_mount_evaluator.py -v`

Expected: pass.

- [ ] **Step 9: Run a minimal real-MuJoCo smoke**

Run:

```powershell
python scripts/search_strict_urdf_mount.py --domain local --robot xarm6 --task cap-left --search-policy best-first --global-candidates 8 --regions 2 --local-per-region 32 --rank-frames 4 --medium-frames 8 --dense-before-success 1 --dense-fallback 1 --post-success-expansions 1 --final-candidates 1 --output tmp/best_first_smoke.npz
```

Expected: audit records 8 Global and exactly 64 Local candidates; every promotion reports `reused_frame_results`; Final evaluates all task frames. Exit may be 0 or 2 based on formal success.

- [ ] **Step 10: Checkpoint**

Run: `python -m pytest tests/test_best_first_search_integration.py tests/test_strict_mujoco_ik.py tests/test_collision_path_stabilization.py -q`

Expected: pass.

---

### Task 6: Split Fingerprints and Add Atomic Search-State Resume

**Files:**
- Modify: `scripts/formal_run_acceleration.py`
- Modify: `scripts/run_twelve_arm_two_single_tasks.py`
- Modify: `scripts/search_strict_urdf_mount.py`
- Modify: `tests/test_formal_run_acceleration.py`
- Test: `tests/test_best_first_search_integration.py`

**Interfaces:**
- Produces: `search_fingerprint(...)`, `solve_fingerprint(...)`, `render_fingerprint(...)`, `atomic_write_json(...)`, `search_state_is_current(...)`.
- Consumes: best-first policy/budget metadata from Task 5.
- Consumed by: formal runner and interrupted-search recovery.

- [ ] **Step 1: Add failing independence tests**

```python
def test_render_input_change_does_not_invalidate_search_fingerprint(tmp_path):
    model = tmp_path / "robot.urdf"
    trajectory = tmp_path / "trajectory.npz"
    renderer = tmp_path / "renderer.py"
    for path, value in ((model, b"m"), (trajectory, b"t"), (renderer, b"v1")):
        path.write_bytes(value)
    before = search_fingerprint([model, trajectory], {"policy": "best-first"})
    renderer.write_bytes(b"v2")
    after = search_fingerprint([model, trajectory], {"policy": "best-first"})
    assert before == after


def test_render_fingerprint_changes_when_renderer_changes(tmp_path):
    renderer = tmp_path / "renderer.py"
    renderer.write_bytes(b"v1")
    before = render_fingerprint([renderer], {"width": 960})
    renderer.write_bytes(b"v2")
    assert before != render_fingerprint([renderer], {"width": 960})
```

- [ ] **Step 2: Run acceleration tests and observe missing helpers**

Run: `python -m pytest tests/test_formal_run_acceleration.py -v`

Expected: import failures for the new helpers.

- [ ] **Step 3: Implement namespaced fingerprint helpers**

Each helper calls the existing `input_fingerprint` with a fixed namespace:

```python
def search_fingerprint(paths, parameters):
    return input_fingerprint(paths, {"namespace": "strict-search-v2", **parameters})

def solve_fingerprint(paths, parameters):
    return input_fingerprint(paths, {"namespace": "strict-solve-v2", **parameters})

def render_fingerprint(paths, parameters):
    return input_fingerprint(paths, {"namespace": "strict-render-v2", **parameters})
```

Search inputs include model/collision assets, validation trajectory, actual incumbent row content, search policy and budget. They exclude renderer files, video settings and the raw `gpu_candidates` value. Solve inputs include selected mount, test trajectory, strict solver and thresholds. Render inputs include strict solution fingerprint, renderer/scene files and output settings.

- [ ] **Step 4: Implement atomic JSON checkpoints**

```python
def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)
```

Serialize frontier nodes, candidate states, evaluated frame indices, frame metrics, q arrays as list data, incumbent history, telemetry and RNG seeds after every completed promotion. Do not checkpoint inside an IK solve.

- [ ] **Step 5: Update formal runner resume decisions**

Use separate paths:

```text
videos/single_arm/search_state/local/<robot>/<task>.json
videos/single_arm/strict_cache/local/<robot>/<task>.npz/.json
reports/.../videos/<robot>/<task>.mp4/.json
```

Decision order:

1. Current strict solution + current video: resume all.
2. Current strict solution + missing/stale video: render only.
3. Current search state + stale/missing strict solution: resume/finish search, then solve/render.
4. No current state: start search.

- [ ] **Step 6: Add interrupted-state tests**

Create a fixture with two closed nodes and one frontier node, write it atomically, reload it, and assert the next popped candidate matches uninterrupted execution. Corrupt the `.tmp` file and assert the valid main checkpoint remains usable.

- [ ] **Step 7: Run fingerprint and resume tests**

Run: `python -m pytest tests/test_formal_run_acceleration.py tests/test_best_first_search_integration.py -v`

Expected: pass.

- [ ] **Step 8: Verify current caches are no longer invalidated by render-only edits in a temporary fixture**

Run a fixture-level script that computes all three fingerprints, modifies a copied renderer, and prints:

```text
search_changed=False
solve_changed=False
render_changed=True
```

- [ ] **Step 9: Checkpoint**

Run: `python -m pytest tests/test_formal_run_acceleration.py tests/test_all_single_task_formal_contract.py -q`

Expected: pass.

---

### Task 7: Reuse Compiled MuJoCo Model Topology Without Changing Collision Semantics

**Files:**
- Modify: `scripts/solve_strict_urdf_task_cache.py`
- Modify: `scripts/search_strict_urdf_mount.py`
- Create: `tests/test_mount_model_reuse.py`

**Interfaces:**
- Produces: `MountModelTemplate`, `build_mount_model_template(robot)`, `apply_mount(model, mount)`.
- Consumed by: best-first solver adapter from Task 5.

- [ ] **Step 1: Write numerical-equivalence tests before optimization**

For xarm6, openarm and willow, build the same three mount poses by the legacy `build_model` and by the proposed template path. Assert:

```python
np.testing.assert_allclose(reused.body_pos, legacy.body_pos, atol=1e-12)
np.testing.assert_allclose(reused.body_quat, legacy.body_quat, atol=1e-12)
np.testing.assert_allclose(reused.geom_pos, legacy.geom_pos, atol=1e-12)
np.testing.assert_allclose(reused.geom_quat, legacy.geom_quat, atol=1e-12)
```

Run identical q samples through both models and require TCP position/orientation errors and table/self-collision flags to match exactly.

- [ ] **Step 2: Run equivalence tests and verify missing API failure**

Run: `python -m pytest tests/test_mount_model_reuse.py -v`

Expected: import failures for `MountModelTemplate` APIs.

- [ ] **Step 3: Implement a fixed-topology template**

Compile the robot, TCP site, table, pedestal and adapter once. Store root body id, root source transform, mount depth, pedestal/adapter geom ids and immutable baseline arrays. `apply_mount` must restore baselines before applying each candidate, update body/geom transforms, call `mujoco.mj_forward`, and return a fresh/reset `MjData`.

If pedestal/adapter `fromto` geometry cannot be reproduced by direct model-array updates with the equivalence tolerances, stop this task and retain per-candidate compile. Record `model_reuse_supported=false` in telemetry; do not weaken geometry or collision checks.

- [ ] **Step 4: Run equivalence tests**

Run: `python -m pytest tests/test_mount_model_reuse.py -v`

Expected: pass for every supported robot; an explicit skip with a technical reason is acceptable only for an unsupported MuJoCo geometry update, never for numerical mismatch.

- [ ] **Step 5: Benchmark compile/apply cost**

Run 20 mounts per robot after one warm-up. Report median legacy build time and median template apply/reset time. Require at least 5× speedup for openarm and willow before enabling reuse by default.

- [ ] **Step 6: Integrate template lifecycle per robot subprocess**

Create one template before candidate evaluation. Do not share mutable `MjModel` or `MjData` across concurrent subprocesses. Preserve legacy `build_model` for final renderer and equivalence tests.

- [ ] **Step 7: Checkpoint**

Run: `python -m pytest tests/test_mount_model_reuse.py tests/test_thirteen_arm_contract.py tests/test_strict_mujoco_ik.py -q`

Expected: pass.

---

### Task 8: Paired Ablation, Formal Acceptance and Documentation

**Files:**
- Create: `scripts/benchmark_best_first_mount_search.py`
- Create: `reports/single_arm/best_first_search_ablation.json` at execution time
- Modify: `docs/skills/specs/2026-08-11-best-first-mount-search-design.md` only if measured constraints require an approved design correction
- Test: all affected tests

**Interfaces:**
- Consumes: legacy and best-first CLI policies plus telemetry.
- Produces: paired machine-readable benchmark and a go/no-go result for making best-first the formal default.

- [ ] **Step 1: Implement the paired benchmark driver**

Use these fixed samples:

```text
xarm6/cap-left              easy, known pass
openarm/open-box-2          expensive mesh, known fail
willow/open-box-2           difficult IK long tail, known fail
```

For each sample run four configurations with identical inputs and seeds:

```text
legacy
cheap_restart_only
best_first_without_reuse
best_first_full
```

Collect formal status/rank, selected mount, worker-seconds, first-success time, candidate counts, new/reused frames, Dense candidates, IK calls/iterations, model compile time and stop reason.

- [ ] **Step 2: Add benchmark validation before running expensive jobs**

The driver must reject a result unless every configuration contains the same input fingerprint, target frame count, thresholds and seed. It must compute paired deltas rather than compare unrelated historical timestamps.

- [ ] **Step 3: Run fast test suite**

Run:

```powershell
python -m pytest tests/test_mount_search_telemetry.py tests/test_best_first_mount_search.py tests/test_incremental_mount_evaluator.py tests/test_best_first_search_integration.py tests/test_formal_run_acceleration.py tests/test_mount_model_reuse.py -q
```

Expected: pass.

- [ ] **Step 4: Run the paired ablation**

Run:

```powershell
python scripts/benchmark_best_first_mount_search.py --output reports/single_arm/best_first_search_ablation.json
```

Expected: valid JSON containing 3 samples × 4 configurations and paired summaries.

- [ ] **Step 5: Apply formal go/no-go gates**

Best-first becomes the default only if all are true:

```text
xarm6/cap-left formal pass is preserved
legacy incumbent is present in every Final finalist set
best-first formal rank is not worse than the legacy incumbent rank
Local candidate count is exactly 64
normal Dense candidate count is <= 3; fallback total is <= 8
promotion duplicate-frame IK count is 0
full-trajectory frame opportunities fall by >= 70%
three-sample cumulative worker time falls by >= 50%
fixed-seed reruns select identical winners and stop reasons
```

If a gate fails, keep `legacy` as runner default and save the ablation evidence; do not silently loosen success thresholds.

If every gate passes, change the parser default from `legacy` to `best-first`, update `test_legacy_remains_default_until_ablation_passes` into `test_best_first_is_default_after_ablation`, and rerun the focused and full regression suites before accepting the switch.

- [ ] **Step 6: Run full regression suite**

Run: `python -m pytest -q`

Expected: all tests pass. Report total count and duration.

- [ ] **Step 7: Run a 24-combination dry-run resume audit**

Compute, without launching expensive search, which of the 24 combinations would resume search, solve or render under the split fingerprints. Expected output lists exactly one action per combination and proves render-only changes do not schedule search.

- [ ] **Step 8: Final checkpoint**

Save test output, benchmark JSON, go/no-go decision and any unsupported-model-reuse explanation. No Git commit can be created in the current non-Git checkout; if moved to a Git worktree, commit implementation and benchmark metadata in reviewable task-sized commits.
