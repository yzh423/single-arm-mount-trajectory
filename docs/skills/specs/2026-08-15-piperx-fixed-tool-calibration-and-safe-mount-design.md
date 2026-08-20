# PiperX Fixed Tool Calibration and Collision-Safe Mount Design

## Scope

Replace the frame-zero orientation mapping, then rebuild PiperX Mount search and fixed-time following around one locked cross-task calibration. Fold Box and Seal Bag share the same left/right calibration quaternions. The official PiperX URDF, `ee_frame`, gripper geometry, and flange-to-gripper transform remain authoritative and are never optimized.

## Fixed coordinate calibration

For side `s` and source frame `t`, the target TCP rotation is

`R_target[s,t] = R_source[s,t] @ R_offset[s]`.

`R_offset[left]` and `R_offset[right]` are constant over every frame, task, and Mount. Source quaternions are normalized and sign-continuized only; held-frame reconstruction, orientation smoothing, and per-frame correction are forbidden.

The search has two deterministic stages:

1. Enumerate the 24 proper signed permutation matrices (`det=+1`) independently for the left and right coordinate mappings.
2. Refine the best discrete pair with bounded XYZ local rotations. Each component is limited to `[-15 deg, +15 deg]` and the search is deterministic.

Calibration uses representative frames from both tasks and an ensemble of upright reference Mounts around the existing 0.702 m configuration. This prevents either task or one exact Mount from defining the tool mapping. Its lexical objective is strict synchronous 6D IK coverage, longest failure run, connected safe branch ratio, conditioning/limit margin, then residual error. Missing IK is always failure.

The selected calibration is written to a versioned JSON artifact containing both fixed quaternions, discrete axis mappings, local refinement angles, task/source fingerprints, reference Mounts, tolerances, and audit metrics. Mount search and rendering must load this artifact and reject missing or mismatched fingerprints.

## Mount search

Mounts remain upright with common Z. Search variables are independent left/right XY, left/right yaw, and shared Z. Bounds are:

- base separation approximately `0.40–0.80 m`; separation is a search variable, not a fixed target;
- shared base Z `0.79–1.00 m`;
- both arms begin outside the demonstrated workspace and work toward its centre;
- roll and pitch remain zero.

The search is centred on the existing approximately 0.702 m reference Mount. It uses the locked orientation calibration and joint paired candidate evaluation. Its lexical score is:

1. maximum synchronous safe strict 6D IK coverage;
2. minimum longest contiguous failure run;
3. maximum ratio of frames with a connectable safe paired branch;
4. maximum minimum targeted cross-arm clearance;
5. maximum singularity and joint-limit margins;
6. minimum joint travel and required retiming.

An unreachable frame counts as failed coverage and cannot improve the collision or clearance score. Sampled screening may rank incomplete evidence, but a final Mount requires complete source-timeline evaluation.

## Clearance and collision

State and swept-edge collision are hard constraints for the final execution path. In addition to penetration checks, the planner reports signed or nearest-geometry distance for:

- left link4 to right gripper_base;
- right link4 to left gripper_base;
- gripper_base to gripper_base;
- fingers to opposite wrist;
- elbows to opposite bases.

The simulation acceptance margin is at least 15 mm. Reports also expose a 30–50 mm real-hardware advisory margin. A state that is unreachable is not considered collision-safe evidence.

TCP separations of approximately 32–50 mm are classified before planning. Explicitly configured task contact pairs may be allowed. Otherwise the source target remains unchanged and the frame is reported as physically simultaneous-infeasible; the planner must not silently project or rotate it.

## IK candidate generation

Candidate seeds cover explicit shoulder, elbow, and wrist strata plus rolling temporal seeds. Optimization uses joint bounds directly. Pose residuals are normalized by the declared tolerances (`position / 1 mm`, `orientation / 1.5 deg`). Candidates near J4/J5 `±89 deg`, singular configurations, or joint limits receive strong risk penalties.

Candidate ordering combines normalized pose error, singular risk, joint-limit risk, distance from the preceding branch, and paired clearance. Diagnostics retain raw candidate counts and failure reasons.

## Joint paired planning

Each frame forms the Cartesian product of left/right IK candidates. State-colliding or margin-violating pairs are removed. Edges that violate source-time dynamics, joint-step bounds, or swept clearance are removed. A joint dynamic program selects the full paired path. If no safe edge exists, bounded safe recovery is explicit and the frame is marked cannot-follow.

Failure categories are mutually attributable: position unreachable, orientation unreachable, simultaneous geometry infeasible, state collision, swept collision, branch discontinuity, or dynamic limit.

## Acceptance

- official URDF tool transform unchanged;
- one locked calibration shared by Fold Box and Seal Bag;
- source quaternion relative motion preserved exactly up to sign;
- final state collision count `0`;
- final swept-edge collision count `0`;
- synchronous strict coverage target above `75%` without counting holds/unreachable states as success;
- substantially shorter longest failure run than the current baseline;
- full 1964-frame Fold Box audit and full 2439-frame Seal Bag audit;
- fixed-source-time front videos rendered only from verified cached trajectories.

