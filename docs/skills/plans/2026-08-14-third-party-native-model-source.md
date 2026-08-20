# Third-party Native Model Source Migration Implementation Plan

> **For Codex:** Execute this plan continuously; do not restart a still-running experiment whose resolved models are already native `third_party` inputs.

**Goal:** Remove 750 mm normalization from every active experiment path and make `third_party` the sole robot-model authority for IK, collision checking, MuJoCo rendering, and reports.

**Architecture:** Centralize all 13 robot contracts in the strict model registry. Every consumer resolves URDF/MJCF, meshes, joint names, flange/TCP, and base link through that registry. Model dimensions remain vendor-native; no forward or inverse uniform scale is permitted. Add source-boundary tests so an `Assets` model or normalization branch cannot silently return.

**Tech Stack:** Python, pytest, MuJoCo, URDF/XML, YAML, PowerShell only for exact binary mesh copying with hash verification.

---

### Task 1: Establish failing source-boundary tests

**Files:**
- Create: `tests/test_third_party_native_model_boundary.py`
- Inspect: `scripts/strict_urdf_model_audit.py`
- Inspect: `configs/robot_registry_13.yaml`

1. Assert all 13 configured model paths resolve below `third_party`.
2. Assert each resolved model and every referenced mesh exists and compiles.
3. Assert active runtime modules contain no `Assets` model paths, `apply_uniform_scale`, or normalization/inverse-normalization execution.
4. Run the test and record the expected failures before implementation.

### Task 2: Complete the third-party model inventory

**Files:**
- Modify: `third_party/official_robot_models/piperx/PiperX.urdf`
- Create/modify: `third_party/official_robot_models/{piperx,big_yam,nero,willow}/`
- Create: per-model `SOURCE.md`

1. Copy exact required binary meshes into each model's third-party directory and verify source/destination SHA-256 equality.
2. Vendor missing robot descriptions under `third_party`, preserving native geometry and joint limits.
3. Record upstream/workspace snapshot provenance without claiming unverified models are official.
4. Verify all model-relative mesh URIs resolve without `Assets` search paths.

### Task 3: Make the strict registry authoritative for all 13 arms

**Files:**
- Modify: `scripts/strict_urdf_model_audit.py`
- Modify: `configs/robot_registry_13.yaml`
- Modify: `scripts/official_model_manifest.py`
- Modify: `SINGLE_ARM_PROJECT_MANIFEST.json`

1. Add Big YAM, Nero, and Willow to the strict model registry.
2. Point all 13 registry entries and mesh roots exclusively at `third_party`.
3. Preserve native dimensions and model-defined TCPs; use the documented 130 mm fallback only where no TCP exists.
4. Remove active legacy/canonical model registries and normalization provenance branches.

### Task 4: Remove normalization from IK, collision, rendering, and reports

**Files:**
- Modify: `scripts/solve_strict_urdf_task_cache.py`
- Modify: `scripts/render_model_assembly_qa.py`
- Modify: `scripts/run_twelve_arm_two_single_tasks.py`
- Modify: report builders consuming normalization provenance
- Modify: relevant `design_optimization/` runtime modules

1. Delete forward/inverse 750 mm scaling calls and scale-derived reach calculations.
2. Read native reach and geometry directly from the strict audit.
3. Remove `Assets/canonical/provenance.json` from cache fingerprints and execution inputs.
4. Preserve scientific outputs while relabeling dimensions as native rather than normalized.

### Task 5: Unify factory/bimanual model loading

**Files:**
- Modify: `factory_bimanual/robot_contracts.py`
- Modify: `factory_bimanual/scene_builder.py`
- Modify: associated tests

1. Derive model paths, joints, base, and TCP from the same strict registry.
2. Load robot specs with model-local third-party asset resolution.
3. Remove special-case `Assets` mesh lookup and legacy duplicate contracts.

### Task 6: Regenerate audits and verify end to end

**Files:**
- Regenerate: `reports/single_arm/strict_urdf_model_audit.json`
- Regenerate: model/provenance/collision audit artifacts

1. Run all focused boundary, authority, topology, collision, IK, and factory tests.
2. Compile every robot model in MuJoCo and perform finite FK/TCP checks.
3. Search active source/config trees for forbidden 750 mm and `Assets` model dependencies.
4. Inspect the still-running formal experiment and confirm no restart is needed.
5. Run the broader relevant pytest suite and report any historical-only remnants separately from active execution paths.

