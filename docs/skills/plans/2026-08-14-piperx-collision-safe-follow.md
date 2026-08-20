# Piper X Collision-Safe Follow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a full-source-time Piper X bimanual Seal Bag path with cross-arm collision as a hard constraint and lexicographically minimal pose relaxation when exact SE(3) following is physically incompatible.

**Architecture:** Extend native MuJoCo candidate generation with weighted orientation solving, then add a tiered target generator that emits exact candidates first and bounded collision-separation/pose-relaxed candidates afterward. Feed both arms into one paired beam/path selector that filters state and swept collision before selection; the renderer consumes only this paired result.

**Tech Stack:** Python, NumPy, SciPy, MuJoCo, pytest.

## Global Constraints

- Both bases remain upright, on the same table, at identical height; no roll, pitch, or tilt search.
- Official Piper X joint limits, native scale, `ee_frame`, table, base, self, cross-arm state, and swept collision remain enforced.
- Cross-arm collision is infeasible, never a score penalty.
- Preserve all 2439 source rows and source timestamps; failed rows use a verified collision-free hold.
- Exact 1 mm / 1.5 degree candidates dominate every relaxed candidate.
- Do not modify protected single-arm code or artifacts.

---

### Task 1: Weighted pose candidate generation

**Files:**
- Modify: `factory_bimanual/mujoco_candidate_generator.py`
- Test: `tests/factory_bimanual/test_mujoco_candidate_generator.py`

**Interfaces:**
- Produces: `CandidateGeneratorConfig.orientation_weight` and `MuJoCoCandidateGenerator.generate_target(side, position, quaternion)`.

- [ ] Write a real-scene failing test showing position-only weight reaches a target while reporting the realized orientation error.
- [ ] Run the focused test and confirm it fails because the configuration/interface is absent.
- [ ] Apply the orientation weight consistently to the DLS error/Jacobian and expose target-level generation.
- [ ] Run candidate-generator tests and keep strict default behavior unchanged.

### Task 2: Tiered collision-safe paired planner

**Files:**
- Create: `factory_bimanual/collision_safe_follow.py`
- Create: `tests/factory_bimanual/test_collision_safe_follow.py`

**Interfaces:**
- Produces: `PoseToleranceTier`, `CollisionSafeFollowConfig`, `CollisionSafeFollowResult`, and `solve_collision_safe_follow(...)`.
- Consumes: `MuJoCoCandidateGenerator.generate_target`, `MuJoCoPairedCollisionChecker`, source timestamps and paired target poses.

- [ ] Write a failing synthetic test where the lowest-cost exact pair collides but a relaxed pair is safe.
- [ ] Confirm the test fails because the paired planner is absent.
- [ ] Implement tier generation, deterministic branch encoding, collision-free state-pair layers, swept transition checks, beam-width 64 selection, and collision-safe holds.
- [ ] Add tests that exact candidates dominate relaxed candidates, collision-only layers hold safely, exact 5 degree boundaries stay deterministic, and outputs remain source-row aligned.
- [ ] Run the focused planner tests.

### Task 3: Seal Bag production wiring

**Files:**
- Modify: `scripts/render_factory_dual_xarm6_se3_follow.py`
- Modify: `tests/factory_bimanual/test_se3_follow_contract.py`

**Interfaces:**
- Consumes: `solve_collision_safe_follow`.
- Produces: trajectory arrays with per-side tier, desired/realized pose errors, paired branch ID, state/edge collision diagnostics, and verified holds.

- [ ] Write a failing contract test proving Piper X rendering invokes the paired path rather than two independent selectors.
- [ ] Add a Piper-X collision-safe mode while leaving xArm6 historical behavior unchanged.
- [ ] Ensure summary ranking reports zero collision, exact/relaxed tier counts, longest failure window, maximum and mean position/orientation errors.
- [ ] Run focused render-contract tests.

### Task 4: Full validation and artifact

**Files:**
- Output only under: `reports/factory_bimanual/seal_bag_dual_piperx/`

- [ ] Run all `tests/factory_bimanual` tests.
- [ ] Run the selected official Piper X mount over all 2439 rows without video.
- [ ] Reject the result if state or swept collision count is nonzero.
- [ ] Compare synchronous coverage, longest failure window, pose error, and tier counts against `seal_bag_piperx_official_native_consistent_targets_v2`.
- [ ] Render a front 720p video only after the zero-collision gate passes.
- [ ] Verify the protected single-arm snapshot is unchanged.
