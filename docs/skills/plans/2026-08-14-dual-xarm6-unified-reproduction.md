# Unified dual-xArm6 Reproduction Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the duplicated Fold Box and Seal Bag bundles on `rg-4090` with one configuration-driven, video-free reproduction suite.

**Architecture:** Store shared Python modules and xArm6 assets once. Keep task-specific source data, trajectories, scene files, mount metadata, and solver policy in separate task paths selected through JSON configurations and one `run_render.sh` interface.

**Tech Stack:** Python 3, JSON, Bash, NumPy, SciPy, MuJoCo, headless OpenCV, SSH, tar, SHA-256.

## Global Constraints

- Final path: `/home/rg/Code/dual_xarm6_reproduction`.
- Remove the old `/home/rg/Code/fold_box_dual_xarm6` and `/home/rg/Code/seal_bag_dual_xarm6` only after the unified suite passes remote verification.
- Include no MP4, JPG, JPEG, or PNG files.
- Preserve different Fold Box and Seal Bag mount and IK-retiming policies.
- Keep one `factory_bimanual/`, one xArm6 asset tree, and one `requirements.txt`.
- Full frame rendering is not required while MuJoCo, SciPy, and OpenCV are absent remotely.

---

### Task 1: Build the configuration-driven local suite

**Files:**
- Create: `work/dual-xarm6-unified-ready/dual_xarm6_reproduction/configs/fold_box.json`
- Create: `work/dual-xarm6-unified-ready/dual_xarm6_reproduction/configs/seal_bag.json`
- Create: `work/dual-xarm6-unified-ready/dual_xarm6_reproduction/scripts/render_task.py`
- Create: `work/dual-xarm6-unified-ready/dual_xarm6_reproduction/run_render.sh`
- Create: `work/dual-xarm6-unified-ready/dual_xarm6_reproduction/README.md`
- Create: `work/dual-xarm6-unified-ready/dual_xarm6_reproduction/requirements.txt`
- Copy once: `factory_bimanual/` and `Assets/canonical/xarm6/`
- Copy task-specific: generator scripts, source CSVs, final trajectories, scenes, summaries, and provenance metadata

**Interfaces:**
- Consumes: `configs/{fold_box,seal_bag}.json`
- Produces: `bash run_render.sh fold_box` and `bash run_render.sh seal_bag`

- [ ] Copy shared and task-specific files without caches, videos, or images.
- [ ] Write both JSON configurations with relative paths, mount coordinates, solver policy, camera, and interpolation settings.
- [ ] Implement `render_task.py CONFIG` to validate configuration, load NPZ data, verify MuJoCo `nq`, and call the shared renderer.
- [ ] Implement `run_render.sh` with an explicit two-value case statement and a nonzero error for any other argument.
- [ ] Document setup, both render commands, and the non-identical mount/IK policies.

### Task 2: Verify the local suite

**Files:**
- Read: the complete local suite

**Interfaces:**
- Consumes: Task 1 output
- Produces: evidence that the suite is portable and complete

- [ ] Compile every Python file using encoding-aware source loading.
- [ ] Load both JSON configurations and assert every referenced input exists.
- [ ] Assert trajectory shapes `(1964, 12)` and `(2439, 12)`.
- [ ] Parse both scene XML files and assert all 12 referenced meshes exist.
- [ ] Assert no video, image, cache, or bytecode file exists.
- [ ] Assert invalid `run_render.sh` input exits nonzero before importing unavailable renderer dependencies.

### Task 3: Upload, verify, and replace

**Files:**
- Create: `/home/rg/Code/dual_xarm6_reproduction`
- Remove after successful replacement: `/home/rg/Code/fold_box_dual_xarm6`
- Remove after successful replacement: `/home/rg/Code/seal_bag_dual_xarm6`

**Interfaces:**
- Consumes: a SHA-256-verified tar archive of the local suite
- Produces: one remote unified suite and no old duplicated bundle directories

- [ ] Archive the suite, upload it to a uniquely named temporary remote directory, and compare local and remote SHA-256.
- [ ] Extract and run the full non-rendering verification against the temporary directory.
- [ ] Move any existing unified destination aside, promote the verified temporary suite, and move both old task directories to backups.
- [ ] Re-run verification against the promoted suite.
- [ ] Remove only the two verified obsolete backups, any previous unified backup, and temporary transfer files.
