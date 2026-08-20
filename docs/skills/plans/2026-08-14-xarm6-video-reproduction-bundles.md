# xArm6 Video Reproduction Bundles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and transfer two independent, video-free bundles that can re-render the selected Fold Box and Seal Bag xArm6 results on `rg-4090`.

**Architecture:** Preserve the repository-relative imports used by the original renderers, copy only task-relevant code/data/assets, and rewrite copied scene mesh paths to be bundle-relative. Each bundle exposes the original Python renderer and documents exact Linux commands.

**Tech Stack:** Python 3, NumPy, SciPy, MuJoCo Python bindings, OpenCV, Bash, SSH/SCP.

## Global Constraints

- Destinations are `/home/rg/Code/fold_box_dual_xarm6` and `/home/rg/Code/seal_bag_dual_xarm6`.
- Copy no `.mp4`, contact sheet, screenshot, cache, or unrelated report files.
- The primary supported operation is re-rendering from the saved final trajectory.
- Bundles must not depend on `E:\YZH123123\single-arm-mount`.
- Verify source compilation, imports, trajectory/scene compatibility, and one in-memory MuJoCo frame.

---

### Task 1: Assemble Fold Box bundle

**Files:**
- Copy: `factory_bimanual/` to `work/bundles/fold_box_dual_xarm6/factory_bimanual/`
- Copy: `scripts/render_factory_dual_xarm6_fold_box.py`, `scripts/rolling_multibranch_ik.py`, and `scripts/strict_mujoco_ik.py`
- Copy: `Assets/canonical/xarm6/`
- Copy: `data/factory/8-11/Fold_Box/handheld_20260811_160754.csv`
- Copy: final Fold Box `.trajectory.npz`, `.scene.xml`, `.scene.json`, `.summary.json`, and `.provenance.json`
- Create: `work/bundles/fold_box_dual_xarm6/README.md`
- Create: `work/bundles/fold_box_dual_xarm6/requirements.txt`

**Interfaces:**
- Consumes: final Fold Box trajectory stem `fold_box_full_se3_acceleration_smoothed_front_720p`
- Produces: a repository-shaped bundle runnable with `python scripts/render_factory_dual_xarm6_fold_box.py`

- [ ] Copy the listed files into the staging bundle while excluding bytecode and videos.
- [ ] Rewrite the copied scene XML `meshdir` to `../../../Assets/canonical/xarm6/`.
- [ ] Add exact environment, re-render, and smoke-test commands to the README.
- [ ] Assert recursively that the bundle contains zero `.mp4` files.

### Task 2: Assemble Seal Bag bundle

**Files:**
- Copy: `factory_bimanual/` to `work/bundles/seal_bag_dual_xarm6/factory_bimanual/`
- Copy: `scripts/render_seal_bag_front_view.py` and trajectory-generation lineage scripts
- Copy: `Assets/canonical/xarm6/`
- Copy: final source trajectory/scene/summary plus final-render metadata and provenance
- Create: `work/bundles/seal_bag_dual_xarm6/README.md`
- Create: `work/bundles/seal_bag_dual_xarm6/requirements.txt`

**Interfaces:**
- Consumes: `seal_bag_global_retimed_no_visual_flip.trajectory.npz`
- Produces: `python scripts/render_seal_bag_front_view.py --source-stem seal_bag_global_retimed_no_visual_flip --output-stem seal_bag_v6_global_retimed_no_flip_front_720p`

- [ ] Copy the listed files into the staging bundle while excluding videos and images.
- [ ] Rewrite the copied scene XML `meshdir` to `../../../Assets/canonical/xarm6/`.
- [ ] Add exact environment, re-render, and smoke-test commands to the README.
- [ ] Assert recursively that the bundle contains zero `.mp4` files.

### Task 3: Transfer atomically to rg-4090

**Files:**
- Replace contents: `/home/rg/Code/fold_box_dual_xarm6/`
- Replace contents: `/home/rg/Code/seal_bag_dual_xarm6/`

**Interfaces:**
- Consumes: the two verified staging directories
- Produces: two complete remote task directories

- [ ] Upload each staging directory to a uniquely named temporary directory beneath `/home/rg/Code/`.
- [ ] Verify uploaded file counts and SHA-256 manifests against local manifests.
- [ ] Move existing empty targets aside, promote the temporary directories, and remove only the now-obsolete empty backups.

### Task 4: Verify remote reproducibility

**Files:**
- Read: both remote bundle trees
- Create transiently: no deliverable video; one rendered RGB frame remains in memory only

**Interfaces:**
- Consumes: remote bundles and installed Python environment
- Produces: command output proving bundle integrity and runtime readiness

- [ ] Run `python -m compileall -q factory_bimanual scripts` in each bundle.
- [ ] Import NumPy, SciPy, MuJoCo, OpenCV, and the task renderer.
- [ ] Load each `.trajectory.npz` and corresponding `.scene.xml`; assert `qpos.shape[1] == model.nq`.
- [ ] With `MUJOCO_GL=egl`, render one frame in memory and assert shape `(120, 160, 3)`.
- [ ] Confirm `find . -type f -iname '*.mp4'` prints nothing.
