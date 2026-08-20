# Rolling Multi-Branch IK Design

## Objective

Replace the current frame-greedy pose IK selection with a bounded rolling-horizon
multi-branch planner. The planner must reduce branch dead ends and joint-continuity
failures without weakening pose tolerances, collision checks, candidate-search size,
or the whole-episode success definition.

## Scope

This change affects strict real-URDF trajectory solving only. GPU mount screening,
mount candidate counts, canonical robot models, TCP transforms, trajectory sources,
and rendering quality remain unchanged. The formal 12-arm run stays stopped until
the representative validation gate passes.

## Constraints and success criteria

- Position tolerance remains 1 mm.
- Orientation tolerance remains 1.5 degrees.
- The real URDF/MJCF model remains the source of Jacobians and collision geometry.
- Table, pedestal, and self-collision checks remain mandatory.
- An episode succeeds only when every evaluated frame succeeds.
- Joint transitions use source timestamps. A configured velocity limit determines
  the allowed transition, capped at 25 degrees per source frame for safety.
- The new planner must not reduce successful-frame count relative to the current
  continuity-recovery solver on any representative validation episode.
- It must reduce joint-discontinuity frames on at least two of the three selected
  difficult representative episodes without adding collision frames.

## Architecture

### Candidate generation

For each target pose, generate up to eight unique IK candidates from the previous
frontier states, the joint-range midpoint, and deterministic seeded restarts. Each
candidate records joint values, position and orientation error, distance to joint
limits, and a Jacobian singularity score. Periodic joints are compared using their
shortest equivalent angular displacement.

Candidate generation remains independent from path selection so it can be tested
with small synthetic MuJoCo chains.

### Rolling branch planner

Use a 12-frame rolling horizon. At each planning step, construct a layered graph
from the candidate sets and run dynamic programming over valid transitions. Keep a
bounded beam of the best paths so runtime remains predictable. Commit only the first
transition, then advance the horizon.

Transition feasibility is determined from the actual source time difference. The
per-joint angular displacement must be within the velocity-derived limit and the
25-degree safety cap. Acceleration is evaluated from two consecutive transitions.

The lexicographic cost order is:

1. valid 6D pose;
2. collision-free state and interpolated edge;
3. velocity feasibility;
4. acceleration cost;
5. joint-limit margin;
6. singularity margin;
7. position and orientation residual;
8. joint travel.

Hard invalidity is never traded for a lower soft cost.

### Local recovery

If no complete horizon path exists, retain the best collision-free partial path.
For an otherwise pose-valid branch whose first transition exceeds the velocity
limit, move toward that branch by the allowed step rather than freezing. When valid
states exist on both sides of a short failed interval, attempt a bidirectional local
connection and accept it only if every interpolated edge is collision-free and all
velocity limits are respected.

Unreachable or collision-only candidates continue to hold the last safe posture.

### Output and diagnostics

The cache retains the realized joint path and adds:

- velocity-limit violations per frame;
- acceleration warnings per frame;
- branch count and chosen branch index;
- recovery mode (`none`, `limited_step`, `bidirectional`, or `hold`);
- singularity and joint-limit margin diagnostics.

Primary failure reasons remain physical collision or actual pose failure.
`joint_discontinuity` and planner recovery remain secondary diagnostic causes.

## Runtime control

Candidate generation is reused across overlapping rolling windows. Candidate sets
are memoized per frame and mount fingerprint. The planner uses a bounded beam and
fixed horizon, avoiding an unbounded full-episode graph. Deterministic random seeds
preserve reproducibility.

The new solver implementation and all planner parameters participate in the task
fingerprint, ensuring old strict caches cannot be resumed as new results.

## Testing

Unit tests cover:

- timestamp-derived transition limits and the 25-degree cap;
- periodic-joint shortest displacement;
- selection of a longer-lived branch over a greedy dead end;
- acceleration cost preference when pose quality is equal;
- safe limited-step recovery and unreachable-target hold behavior;
- collision rejection for states and interpolated edges;
- deterministic candidate deduplication;
- failure-reason primary/secondary classification.

Integration validation uses:

- Doosan `open-box-2`;
- Franka Panda `tube-left-upright`;
- ARX X5 `pick-from-high-left`.

For each task, compare the old stopped-run cache, the current limited-step solver,
and the rolling planner. Record successful frames, joint-discontinuity frames,
maximum velocity and acceleration, collision frames, runtime, position RMSE, and
orientation RMSE. The formal run may restart only after the stated validation gate
passes.

## Failure handling

Planner exhaustion produces a valid diagnostic failure, not a process error.
Malformed timestamps, missing joints/sites, non-finite candidates, or inconsistent
array shapes remain hard errors. Interrupted experiments resume only from artifacts
whose complete input fingerprint matches the new solver.
