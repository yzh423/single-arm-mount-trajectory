# Unified dual-xArm6 reproduction suite

## Goal

Replace the two duplicated remote bundles with one suite at
`/home/rg/Code/dual_xarm6_reproduction`. The suite must reproduce the saved
Fold Box and Seal Bag renders without storing any video or diagnostic image.

## Architecture

The suite contains one shared `factory_bimanual/` package, one shared xArm6
asset tree, one dependency file, and one saved-trajectory renderer. Task-level
differences remain explicit under `configs/`, `data/`, and `reports/`.

```text
dual_xarm6_reproduction/
  factory_bimanual/
  Assets/canonical/xarm6/
  scripts/
  configs/fold_box.json
  configs/seal_bag.json
  data/factory/8-11/Fold_Box/
  data/factory/8-11/Seal_Bag/
  reports/factory_bimanual/fold_box_dual_xarm6/
  reports/factory_bimanual/seal_bag_dual_xarm6/
  run_render.sh
  requirements.txt
  README.md
```

`run_render.sh fold_box` and `run_render.sh seal_bag` select a JSON task
configuration and call the same renderer. The configuration records the
trajectory and scene paths, output stem, camera, interpolation mode, and the
final mount parameters used by that task.

## Preserved task differences

Both tasks use the same xArm6 model, full-SE(3) candidate generator, 3.14
rad/s velocity limit, 35-degree maximum joint step, 1 mm position threshold,
and 1.5-degree orientation threshold. They do not share one mount:

- Fold Box uses base spacing 0.4630826309545253 m and task-specific XY
  positions, followed by paired global retiming and 60 rad/s^2 acceleration
  retiming.
- Seal Bag uses base spacing 0.6097157330035753 m and different XY positions,
  with the no-visual-flip minimum-retiming path.
- Both use upright bases at z=0.87 m with yaw 315 degrees left and 45 degrees
  right.

The original task generators remain separate scripts because their solver and
retiming policies differ. Shared support modules are stored only once.

## Replacement and safety

Upload and validate the unified suite in a temporary remote directory first.
After validation, atomically promote it to
`/home/rg/Code/dual_xarm6_reproduction`, move the old
`fold_box_dual_xarm6` and `seal_bag_dual_xarm6` directories to temporary
backups, verify the promoted suite again, and only then remove those backups.
Temporary transfer archives are also removed after successful promotion.

## Verification

Verification must prove:

1. no MP4 or diagnostic image exists in the suite;
2. all Python files compile;
3. both JSON task configurations resolve to existing files;
4. Fold Box trajectory shape is `(1964, 12)` and Seal Bag is `(2439, 12)`;
5. both scene XML files resolve all 12 mesh assets through portable paths;
6. `run_render.sh` accepts exactly `fold_box` or `seal_bag`;
7. `requirements.txt` contains NumPy, SciPy, MuJoCo, and headless OpenCV.

The current remote environment lacks MuJoCo, SciPy, and OpenCV. Runtime setup
remains documented through `python -m pip install -r requirements.txt`; full
frame rendering is not required during replacement unless those dependencies
are already installed.
