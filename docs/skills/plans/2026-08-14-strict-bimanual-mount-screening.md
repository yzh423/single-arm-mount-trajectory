# Strict Bimanual Mount Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select Piper X tabletop mounts only after paired IK, collision, swept-edge and large structural-crossing hard gates, while allowing limited contact-free gripper overlap.

**Architecture:** Add a focused topology evaluator beside the existing MuJoCo collision adapter. Strengthen the paired mount evaluator so connectivity never resets, expand the deterministic coarse grid, and require a full-timeline audit record before selection. Dense work remains bounded to hard-gate survivors.

**Tech Stack:** Python, NumPy, MuJoCo, pytest.

## Global Constraints

- Modify only `factory_bimanual`, paired-mount scripts, their tests and documentation.
- Both Piper X bases remain upright, on one table and at exactly one shared height; do not search roll, pitch or tilt.
- MuJoCo state and swept collision counts must be zero.
- Structural link-order reversal over 0.030 m is rejected.
- Contact-free TCP/gripper order reversal up to 0.080 m is allowed.
- Full timeline means all 2439 Seal Bag source rows.
- Do not modify protected single-arm code or artifacts.

---

### Task 1: Structural topology gate

**Files:**
- Create: `factory_bimanual/mount_topology.py`
- Create: `tests/factory_bimanual/test_mount_topology.py`

**Interfaces:**
- Produces: `MountTopologyConfig(structural_crossing_limit_m=.030, gripper_overlap_limit_m=.080, transition_steps=5)`.
- Produces: `MuJoCoMountTopologyChecker(model, data, name_map, config).state(left_q, right_q)` and `.transition(previous, current)` returning a report with structural crossing depth/count, gripper overlap depth/count and `valid`.

- [ ] Write tests proving a 31 mm structural reversal is invalid, an 80 mm contact-free TCP overlap is valid, an 81 mm overlap is invalid, and an interpolated midpoint crossing is detected.
- [ ] Run `python -m pytest -q tests/factory_bimanual/test_mount_topology.py` and verify RED because the module is missing.
- [ ] Implement base-axis projections for `left/right_link3`, `left/right_link5`, and `left/right_tcp`; validate required MuJoCo names and interpolate joint pairs deterministically.
- [ ] Run the focused test and verify PASS.

### Task 2: Strict sparse paired evaluator

**Files:**
- Modify: `scripts/search_fold_box_piperx_paired_mount.py`
- Modify: `tests/factory_bimanual/test_fold_box_piperx_mount_search.py`

**Interfaces:**
- Consumes: `MuJoCoMountTopologyChecker`.
- Produces: `evaluate_pair(...)` records containing `continuous_pair_coverage`, `disconnected_rows`, `structural_crossing_frames`, `structural_edge_crossing_frames`, `maximum_structural_crossing_m`, `gripper_overlap_frames`, and `rejection_reason`.

- [ ] Add a failing evaluator test showing that a gap followed by safe candidates cannot reset connectivity, and selection rejects any structural state/edge crossing.
- [ ] Run the new tests and verify the expected RED assertion.
- [ ] Keep the last connected pair layer across missing rows, reject transitions from the last connected source row using the actual elapsed source interval, integrate topology state/edge gates, and emit explicit rejection evidence.
- [ ] Run the focused paired-search tests and verify PASS.

### Task 3: Expanded deterministic coarse search and joint refinement

**Files:**
- Modify: `scripts/search_fold_box_piperx_paired_mount.py`
- Modify: `scripts/search_fold_box_piperx_mount.py`
- Modify: `tests/factory_bimanual/test_fold_box_piperx_mount_search.py`

**Interfaces:**
- Produces: `deterministic_pair_mounts(task, maximum=72)` using radii `(0.26, 0.34, 0.42)`, 45-degree table angles, yaw offsets `(-30, 0, 30)` and explicit farthest-point deterministic selection.
- Produces: joint local neighbours that perturb one or both bases while retaining exact shared height.

- [ ] Add failing tests for deterministic repeatability, inclusive search boundaries, shared-height preservation and expanded spatial/yaw coverage.
- [ ] Run focused tests and verify RED against the old 36-record stride thinning.
- [ ] Implement the expanded grid and deterministic farthest-point reduction; remove enumeration-stride thinning; keep local evaluation paired.
- [ ] Run focused tests and verify PASS.

### Task 4: Full-timeline selection evidence

**Files:**
- Modify: `scripts/search_fold_box_piperx_mount.py`
- Modify: `scripts/search_fold_box_piperx_paired_mount.py`
- Modify: `scripts/search_seal_bag_piperx_paired_mount.py`
- Modify: `tests/factory_bimanual/test_fold_box_piperx_mount_search.py`
- Modify: `tests/factory_bimanual/test_search_seal_bag_piperx_paired_mount.py`

**Interfaces:**
- Produces: `select_paired_mount(records)` accepting only `status == "valid_selection"`, `audit_scope == "full_timeline"`, `audited_source_rows == 2439`, and zero collision/crossing counts.
- Produces: a bounded finalist phase using the existing collision-safe paired planner and a SHA-256 audit fingerprint.

- [ ] Add failing tests proving sparse-only, incomplete-row, collision, edge-collision and structural-crossing records cannot be selected; small contact-free gripper overlap may be selected.
- [ ] Run focused tests and verify RED.
- [ ] Implement full-audit validation, fingerprint generation and lexicographic ranking: coverage, longest HOLD, relaxed-tier count, TCP error, margins, spacing.
- [ ] Run focused tests and verify PASS.

### Task 5: Verification and isolation

**Files:**
- Verify all files above; do not create production video.

- [ ] Run `python -m py_compile factory_bimanual/mount_topology.py scripts/search_fold_box_piperx_paired_mount.py scripts/search_seal_bag_piperx_paired_mount.py`.
- [ ] Run `python -m pytest -q tests/factory_bimanual` and require zero failures.
- [ ] Run the protected-path isolation test and inspect modified paths to confirm no single-arm files/artifacts changed.
- [ ] Run a short real MuJoCo Seal Bag mount-screen smoke test and confirm its selected/surviving records have zero state, swept and structural crossing counts.
- [ ] Record the remaining cost and do not claim a full 2439-row mount selection until the full finalist audit actually finishes.
