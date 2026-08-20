# PiperX Factory Per-Task Mount Search Design

## Objective

Select one representative trajectory for each formal task category under
`data/factory`, independently search three physical PiperX bimanual mounting
modes, and retain exactly one fully audited mount layout per task category.
The experiment must not modify or consume the protected single-arm results.

## Dataset Scope

- Include the 27 formal handheld CSV trajectories under `data/factory`.
- Exclude `cleaning_manifest_20260811.csv` because it is metadata, not a
  trajectory.
- Exclude both files under `_rejected_short` because they were explicitly
  rejected upstream.
- Group by the immediate task directory name, producing 11 task categories.
- Select the trajectory with the largest source-row count in each category.
  Break an exact row-count tie by normalized relative path in ascending order.
- Persist the chosen files, row counts, SHA-256 hashes, and exclusion reasons
  before mount evaluation begins.
- Preserve the complete source timeline. Non-finite pose values may be
  interpolated only on rows whose corresponding source validity flag is false;
  the flag remains false. Non-finite values on valid rows remain a hard error.
- Record but permit finite valid-row translation jumps up to 0.20 m so the two
  single-episode categories with measured 0.152 m and 0.165 m jumps remain in
  scope; these jumps are not smoothed away and may reduce tracking coverage.

## Physical Mount Modes

Every task is evaluated independently in these three modes:

1. `upright_table`: both bases are mounted on the workbench and their local
   mounting axes point vertically upward. The common base-origin height is
   0.81 m in the existing 0.75 m workbench scene.
2. `horizontal_forward`: both bases are mounted at the same height on physical
   vertical poles. Their common height is the midpoint of the combined left
   and right trajectory vertical range. Both mounting axes are horizontal and
   parallel. Their shared forward direction is the normalized XY vector from
   the midpoint of the two base origins to the combined registered trajectory
   center; candidates with a degenerate vector are invalid.
3. `inverted`: both bases are mounted to the overhead support at a common
   base-origin height of 1.41 m and their local mounting axes point vertically
   downward.

For every mode, the two base origins share the same world Z coordinate. Their
XY positions and yaw values are searched independently within the existing
workbench-safe XY bounds.

## Safety Invariants

- Base-origin separation is at least 0.60 m for every generated, refined, and
  executed mount.
- The two bases, adapters, poles, workbench, and overhead support are physical
  MuJoCo collision geometry where applicable.
- State collisions and swept collisions between adjacent audited states are
  hard failures.
- Structural arm crossing is a diagnostic, not a collision by itself. A small
  gripper projection overlap is allowed only when MuJoCo reports no contact.
- No candidate with a collision in the complete source timeline can be the
  final layout.
- If no fully collision-free finalist exists, the task is reported as
  `no_safe_layout`; a minimum-collision diagnostic must never be promoted to a
  selected layout.

## Search Procedure

For each of the 11 selected trajectories and each of the three modes:

1. Generate 36 deterministic coarse candidates spanning task-relative XY,
   yaw, and the mode-specific installation height.
2. Run sparse synchronized paired IK with real MuJoCo state and swept collision
   checks.
3. Rank hard-gate survivors and run dense evaluation on the best 6 mounts.
4. Generate 12 deterministic local refinements around the best safe dense
   mount and evaluate them with the dense solver settings.
5. Run complete source-time execution and collision audit for the best 4
   finalists.

The search is resumable at the individual candidate level. Candidate identity
includes the source CSV hash, task registration, mount-mode schema, collision
rules, solver settings, and mount pose, so stale results cannot cross modes or
tasks.

## Selection and Metrics

A task's final selection is made only among finalists with zero complete-track
state and swept collision frames. The lexicographic ranking is:

1. Higher synchronous strict coverage.
2. Shorter longest contiguous failure segment.
3. Lower mean and high-percentile TCP position/orientation error.
4. Larger minimum joint-limit margin.
5. Larger low-percentile singularity margin.
6. Larger base separation as a final clearance preference.

Reports retain per-arm coverage and errors, recovery/hold counts, collision
classes, joint margins, singularity margins, mount XYZ/yaw/quaternion, source
timing, solver configuration, and wall time.

## Outputs

All new files remain below
`reports/factory_bimanual/piperx_factory_per_task_mount_search/`:

- `dataset_manifest.json`: the 11 representatives and excluded inputs.
- `checkpoints/<task>/<mode>.json`: atomic resumable search state.
- `runs/<task>/<mode>/<fingerprint>.*`: complete finalist artifacts.
- `selected_layouts.json`: one selected layout or explicit safe-failure status
  per task.
- `comparison_metrics.csv`: task-by-mode and final-selection metrics.
- `piperx_factory_per_task_mount_search_report.pdf`: aggregate methodology,
  results, limitations, and per-task tables.

Video generation is outside this batch's required output. Existing videos and
the earlier Seal Bag comparison remain unchanged.

## Failure Handling and Resource Policy

- A malformed selected CSV blocks only its task and records the loader error.
- A failed candidate records its reason and does not abort its mode.
- A failed mode does not abort the remaining modes or tasks.
- Atomic JSON checkpoints are updated after every candidate.
- The default runner uses one MuJoCo worker because each worker has previously
  consumed about 1 GB of memory. Controlled concurrency can be enabled later,
  but deterministic results and checkpoint isolation must remain identical.

## Verification

- Unit tests cover representative selection, exclusions, deterministic ties,
  mode-specific height/orientation, 0.60 m separation at all search stages,
  task-scoped fingerprints, hard collision rejection, ranking, and resume.
- Scene smoke tests compile all three modes for PiperX and verify base axes,
  equal Z, physical poles/supports, and collision vocabulary.
- A short-prefix integration test runs all 11 tasks through all three modes
  without executing the full dataset.
- The full batch starts only after the focused and existing factory-bimanual
  regression suites pass.
