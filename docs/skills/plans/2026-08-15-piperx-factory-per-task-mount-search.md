# PiperX Factory Per-Task Mount Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a resumable batch that chooses one representative episode per factory task, searches three physical PiperX mount modes independently, and emits one fully collision-audited layout per task.

**Architecture:** A dataset catalog freezes the 11 representative inputs and hashes. A task registration adapter feeds the existing paired MuJoCo evaluator. A batch orchestrator runs deterministic coarse/dense/local/final stages with task-scoped fingerprints and atomic checkpoints, then a reporting layer selects only complete-track zero-collision layouts.

**Tech Stack:** Python 3.11, NumPy, MuJoCo, pytest, existing `factory_bimanual` and `scripts` modules.

## Global Constraints

- Do not modify files or results under `reports/single_arm`.
- Exclude the cleaning manifest and `_rejected_short` inputs.
- Select the longest source-row episode per task with normalized-path tie-break.
- Search `upright_table`, `horizontal_forward`, and `inverted` independently.
- Keep both bases at identical Z and at least 0.60 m apart.
- State and swept collision counts must both be zero for final selection.
- Persist every candidate atomically and resume only matching fingerprints.
- Store new artifacts only under `reports/factory_bimanual/piperx_factory_per_task_mount_search`.

---

### Task 1: Deterministic Factory Dataset Catalog

**Files:**
- Create: `factory_bimanual/factory_task_catalog.py`
- Create: `tests/factory_bimanual/test_factory_task_catalog.py`

**Interfaces:**
- Produces: `FactoryEpisode`, `TaskRepresentative`, `build_factory_task_catalog(root: Path)`, and `write_dataset_manifest(...)`.
- Consumes: only CSV paths and bytes below the supplied factory root.

- [ ] **Step 1: Write failing tests** for manifest/rejected exclusions, longest-row selection, deterministic path tie-breaking, SHA-256 hashes, and the 11-category workspace result.
- [ ] **Step 2: Run** `pytest -q tests/factory_bimanual/test_factory_task_catalog.py` and verify the module/API failures.
- [ ] **Step 3: Implement** immutable episode/representative records, streaming row counts and hashes, deterministic grouping, and atomic manifest output.
- [ ] **Step 4: Run the focused tests** and verify all catalog assertions pass.

### Task 2: Physical Horizontal-Forward Mount Mode

**Files:**
- Modify: `factory_bimanual/mount_orientation.py`
- Modify: `factory_bimanual/orientation_mount_search.py`
- Modify: `factory_bimanual/scene_builder.py`
- Modify: `tests/factory_bimanual/test_mount_orientation.py`
- Modify: `tests/factory_bimanual/test_orientation_mount_search.py`

**Interfaces:**
- Produces: `horizontal_forward` mode quaternions and deterministic task-specific candidate grids.
- Consumes: registered task trajectory, base XY positions, common mode height, and per-base yaw.

- [ ] **Step 1: Write failing tests** asserting parallel horizontal base axes point from the base midpoint toward the task center, bases have identical midpoint-height Z, physical vertical poles compile, upright is 0.81 m, inverted is 1.41 m, and all candidates satisfy 0.60 m separation.
- [ ] **Step 2: Run the two focused test files** and verify failures mention the missing mode/axis behavior.
- [ ] **Step 3: Implement** the new mode without changing the legacy `horizontal_wall` interpretation used by existing archived comparisons.
- [ ] **Step 4: Run focused and scene smoke tests** and verify all three new batch modes compile.

### Task 3: Generic Task Registration and Search Configuration

**Files:**
- Create: `factory_bimanual/per_task_mount_search.py`
- Create: `tests/factory_bimanual/test_per_task_mount_search.py`

**Interfaces:**
- Produces: `load_registered_representative(...)`, `PerTaskSearchConfig`, `candidate_fingerprint(...)`, `rank_full_finalist(...)`, and `select_safe_layout(...)`.
- Consumes: `TaskRepresentative`, the lossless factory loader, shared translation registration, and existing paired search metrics.

- [ ] **Step 1: Write failing tests** for full-row loading, deterministic shared registration, fingerprint changes across CSV/mode/rules/settings, 36/6/12/4 budgets, lexicographic ranking, and rejection when every finalist collides.
- [ ] **Step 2: Run the focused tests** and verify the desired API is absent.
- [ ] **Step 3: Implement** the generic registration/configuration/ranking module using existing source loader and paired search primitives.
- [ ] **Step 4: Run focused tests** and confirm selection cannot promote a diagnostic collision result.

### Task 4: Atomic Resumable Batch Orchestrator

**Files:**
- Create: `scripts/run_piperx_factory_per_task_mount_search.py`
- Create: `tests/factory_bimanual/test_factory_per_task_search_runner.py`

**Interfaces:**
- Produces: dataset manifest, per-task/per-mode checkpoints, execution artifacts, and `selected_layouts.json`.
- Consumes: Tasks 1–3 plus existing `evaluate_pair`, full fixed-time runner, and atomic artifact helpers.

- [ ] **Step 1: Write failing tests** using injected evaluators for exact 11×3 scheduling, candidate-level resume, stale-fingerprint rejection, single-task failure isolation, stage budgets, and final hard gates.
- [ ] **Step 2: Run the focused test** and verify orchestration APIs fail before implementation.
- [ ] **Step 3: Implement** a CLI with `--dry-run`, `--short-prefix`, `--task`, and `--full`; default to one worker and write a heartbeat/status file after every candidate.
- [ ] **Step 4: Run focused tests** and a dry-run against the real 11-task catalog.

### Task 5: Metrics and Aggregate Report Inputs

**Files:**
- Create: `factory_bimanual/per_task_mount_report.py`
- Create: `tests/factory_bimanual/test_per_task_mount_report.py`

**Interfaces:**
- Produces: `comparison_metrics.csv`, report model JSON, and completeness validation.
- Consumes: checkpoints, complete finalist summaries, and selected layouts.

- [ ] **Step 1: Write failing tests** for 33 task-mode rows, 11 final rows, collision fields, tracking/error/margin metrics, explicit `no_safe_layout`, and rejection of partial/stale evidence.
- [ ] **Step 2: Run the focused report tests** and verify the module is absent.
- [ ] **Step 3: Implement** evidence loading, completeness gates, CSV output, and report-model JSON without overstating failed modes.
- [ ] **Step 4: Run focused tests** and verify deterministic row ordering.

### Task 6: Verification and Batch Launch

**Files:**
- Modify only files from Tasks 1–5 if verification exposes defects.
- Create runtime artifacts under `reports/factory_bimanual/piperx_factory_per_task_mount_search/`.

**Interfaces:**
- Consumes: completed implementation and real factory data.
- Produces: a running resumable batch with preflight evidence.

- [ ] **Step 1: Run focused tests** for Tasks 1–5.
- [ ] **Step 2: Run** `pytest -q tests/factory_bimanual` with an extended timeout and resolve regressions without weakening safety gates.
- [ ] **Step 3: Run the real dry-run** and verify exactly 11 representatives, 33 mode jobs, output isolation, hashes, and physical scene compilation.
- [ ] **Step 4: Run a short-prefix integration** across all 11 tasks and three modes; require zero orchestration crashes and explicit per-mode outcomes.
- [ ] **Step 5: Launch the full resumable batch** in a hidden process, persist PID/log paths/status, and verify the first checkpoint is advancing.

## Plan Self-Review

- Every design requirement maps to Tasks 1–6.
- New APIs and output locations are defined before consumption.
- The batch never treats collision minimization as collision freedom.
- The legacy outward-facing comparison remains reproducible; the new forward mode has a distinct name and fingerprint schema.
- The plan contains no deferred implementation placeholders.
