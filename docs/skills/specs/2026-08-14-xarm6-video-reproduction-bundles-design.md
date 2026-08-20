# xArm6 final-video reproduction bundles

## Goal

Create two independent, Linux-portable bundles for `fold_box_dual_xarm6` and
`seal_bag_dual_xarm6`, then place them under `/home/rg/Code/` on `rg-4090`.
Each bundle must contain the code, final trajectory data, scene description,
robot assets, and environment instructions needed to reproduce the selected
final render. MP4 files and diagnostic images are excluded.

## Selected final results

- Fold Box: `fold_box_full_se3_acceleration_smoothed_front_720p`
- Seal Bag: `seal_bag_v6_global_retimed_no_flip_front_720p`, rendered from
  `seal_bag_global_retimed_no_visual_flip.trajectory.npz`

## Bundle layout

Each task directory preserves a small repository-like layout:

- `factory_bimanual/`: imported production modules required by the renderer
- `scripts/`: the task renderer and its direct script dependencies
- `Assets/canonical/xarm6/`: URDF, meshes, and related MuJoCo assets
- `reports/factory_bimanual/<task>/`: final trajectory, scene XML/JSON,
  summary, provenance, and relevant non-video intermediate metadata
- `data/`: original task CSV when needed by the full trajectory generator
- `README.md`, `requirements.txt`, and a Linux entry point

The entry point resolves paths relative to its own bundle. It must not depend
on the original `E:\YZH123123\single-arm-mount` path.

## Scope and exclusions

The original generation scripts are retained so the optimization lineage is
auditable. The primary supported operation is deterministic re-rendering from
the saved final trajectory. Full MP4 files, contact sheets, screenshots,
temporary files, caches, unrelated robot assets, and unrelated experiment
reports are excluded.

## Verification

After transfer, verify on `rg-4090` that:

1. expected files exist and no video files were copied;
2. Python source compiles and required imports resolve;
3. the MuJoCo scene loads with the saved trajectory;
4. a short smoke render succeeds without producing a full deliverable video.

If the remote Python environment lacks a dependency, record the exact install
command in the bundle README rather than silently omitting validation.
