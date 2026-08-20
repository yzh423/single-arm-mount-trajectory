# Official Piper X for Factory Bimanual Experiments

Date: 2026-08-14

## Objective

Use the manufacturer-supplied AgileX Piper X geometry at its native scale for
the isolated factory-bimanual experiments. Do not modify or invalidate the
single-arm implementation or its assets.

## Authoritative source

- Repository: `https://github.com/agilexrobotics/agx_arm_urdf`
- Pinned revision: `f6642ce0d7872c686f29c99e9e10cd23d1d49313`
- Robot subtree: `piper_x/`
- Arm model: `piper_x/urdf/piper_x_description.urdf`
- Gripper composition: `piper_x/urdf/piper_x_with_gripper_description.xacro`
- Meshes: `piper_x/meshes/*.stl`

The repository distinguishes `piper_x/` from `piper/`. Only `piper_x/` is in
scope. Every imported source must retain its upstream path, pinned revision,
SHA-256 digest, and license/provenance record.

## Current defect

The factory-bimanual `piperx` contract currently loads
`Assets/canonical/piperx/robot.urdf`. That derivative has a recorded
normalization scale of `0.9489617327844065`, so its kinematics are not at the
official physical dimensions. It also defines the task TCP as
`gripper_base + 0.13 m`, whereas the official Piper X model defines `ee_frame`
at `gripper_base + 0.115 m`.

Consequently, all existing factory-bimanual Piper X mount choices, IK coverage,
failure windows, trajectories, and videos are legacy evidence for a scaled
model. They must not be presented as results for the official Piper X.

## Asset isolation

Create a factory-bimanual-only, MuJoCo-loadable derivative from the pinned
official files. It must:

- preserve all native joint origins, axes, limits, inertials, and mesh scale;
- use `normalization_scale = 1.0`;
- resolve meshes without ROS package lookup or hidden runtime rewriting;
- preserve the official `ee_frame` fixed transform;
- live outside `Assets/canonical/piperx/` so single-arm consumers are unchanged;
- contain a machine-readable provenance manifest with upstream hashes.

The factory-bimanual robot contract will point only to this isolated asset.

## Kinematic contract

- Model identity: AgileX Piper X (`piper_x`), six revolute arm joints.
- Controlled joints: `joint1` through `joint6`.
- Base frame: `base_link`.
- TCP frame: official `ee_frame`.
- Additional TCP offset: `(0, 0, 0)`.
- Model scale: exactly `1.0`.
- Gripper geometry: retained for visualization and collision checking; gripper
  joints are not part of the six-DoF arm controller state.
- Both bases remain upright, attached to the same tabletop, at equal height.

## Data and execution flow

1. Verify/download the pinned official Piper X source and meshes.
2. Produce the isolated MuJoCo-loadable asset without geometric normalization.
3. Load it through the existing factory-bimanual scene builder.
4. Compile a two-arm MuJoCo scene and verify frame names, dimensions, joint
   limits, TCP transforms, contacts, and fixed tabletop mounts.
5. Invalidate reuse of scaled-model mount and trajectory fingerprints.
6. Re-run the complete left/right mount search using both target trajectories.
7. Select mounts by whole-trajectory synchronized strict SE(3) coverage,
   collision counts, longest failure run, tracking error, singularity margin,
   and joint-limit margin.
8. Re-run the seal-bag trajectory and specifically audit the opening
   `1.4-2.0 s` interval.

## Solver requirement exposed by the diagnosis

The official-model rerun must not rely solely on the current damped least
squares candidate generator. The opening failure audit demonstrated that DLS
can miss poses satisfying the declared `1 mm / 1.5 deg` inequalities near a
singular boundary. Add an acceptance-aware constrained fallback that activates
only when ordinary DLS returns no strict candidate. It must optimize within the
declared tolerances, preserve deterministic seeds, report singularity and joint
margin, and remain subject to the existing state/edge collision and source-time
velocity gates.

This fallback must not silently relax task tolerances. Any retiming or bounded
recovery must be explicitly reported rather than counted as source-time strict
tracking.

## Tests and acceptance criteria

Tests must be written and observed failing before production changes.

The change is accepted only when all of the following hold:

1. Asset identity tests match the pinned official Piper X joint origins, axes,
   limits, fixed frames, mesh hashes, and upstream revision.
2. The factory-bimanual contract uses the isolated asset, `ee_frame`, zero TCP
   offset, and scale `1.0`.
3. Protected single-arm files and their recorded hashes are unchanged.
4. A real two-arm MuJoCo scene compiles and both bases are fixed to the same
   tabletop at equal height without initial penetration.
5. Forward kinematics at deterministic joint samples agrees with the pinned
   official model to numerical precision.
6. A regression test reproduces a DLS false negative near a tolerance boundary
   and the constrained fallback recovers it without relaxing tolerances.
7. Collision, velocity, discontinuity, and row-alignment tests remain green.
8. Mount selection is based on the complete trajectory, not only the opening
   failure interval.
9. The final report labels old outputs as scaled legacy and new outputs as
   official native-scale Piper X.

## Output isolation

New results stay under `reports/factory_bimanual/` with a distinct official
Piper X run identity and source fingerprint. Existing reports are retained for
comparison and are never overwritten or relabeled in place.

## Out of scope

- Modifying single-arm assets, solvers, reports, or videos.
- Using the older Piper model.
- Relaxing the `1 mm / 1.5 deg` strict metric to inflate coverage.
- Reusing the scaled-model mount without a complete native-scale validation.

## Repository note

At design time, `E:/YZH123123/single-arm-mount` is not inside a Git worktree.
The design file therefore cannot be committed until repository metadata is
restored or a valid repository root is supplied.
