# Strict Bimanual Mount Screening Design

## Scope

Strengthen only the independent `factory_bimanual` Piper X mount search. Do not
change protected single-arm code or historical single-arm results. Both robots
remain upright on the same table at one shared base height; roll, pitch and tilt
are not searched.

## Safety semantics

Mount feasibility is a hard gate, not a score penalty.

- Reject every mount with any MuJoCo state contact or swept-transition contact
  classified as cross-arm, self, table or base collision.
- Reject large structural crossing of the two arm chains even when the meshes
  have not yet contacted.
- Permit small terminal-gripper projection overlap when there is no MuJoCo
  contact. This accommodates the demonstrated bag manipulation without allowing
  the forearms or elbows to exchange sides.
- A missing IK layer or disconnected transition ends the continuous feasible
  prefix. A later safe layer must not be treated as connected by clearing the
  previous layer.

Structural crossing is measured in the base-pair coordinate frame. The signed
ordering of corresponding non-gripper anchors (mid-arm and wrist bodies) is
checked against the left/right base order. More than 30 mm of reversed ordering
at a structural anchor is a hard crossing. TCP/gripper ordering may reverse by
up to 80 mm, but only when all exact MuJoCo contact checks remain clear. The
thresholds are configuration fields and are recorded in every search artifact.

## Search pipeline

### Gate 0: deterministic mount geometry

Generate an inclusive, deterministic XY/yaw/shared-height grid. Reject candidates
whose mounting discs leave the tabletop, whose bases overlap, whose shared height
is outside the allowed range, or whose left/right base ordering is invalid.
Do not thin the grid using list stride because that depends on enumeration order;
use explicit evenly distributed indices/cells.

### Gate 1: sparse paired feasibility

Use endpoints, XYZ extrema, minimum hand separation, maximum hand separation,
largest pose increments and uniformly distributed source rows. At every row,
generate candidates for both arms and construct collision-free pairs. Propagate
only through collision-free swept edges. Track structural crossing separately.

Any state collision, swept collision or structural crossing rejects the mount
from refinement. IK-unreachable rows reduce coverage but cannot reset path
connectivity. Sparse evaluation records exact source rows and rejection reasons.

### Gate 2: local refinement

Refine only Gate-1 survivors. Evaluate local XY and yaw neighbours jointly; do
not refine the two arms independently. Shared height remains identical. Every
refined candidate passes the same hard gates before ranking.

### Gate 3: full-timeline finalist audit

Run the collision-safe paired planner over all 2439 Seal Bag source rows for the
bounded finalist set. Audit every selected state and every swept edge again in
MuJoCo, plus structural topology. A mount is selectable only when all three hard
counts are zero:

1. state collision count;
2. swept-edge collision count;
3. structural large-crossing count.

The selected mount is ranked lexicographically by maximum collision-safe
following coverage, shortest longest HOLD interval, lowest relaxed-tier use,
lowest TCP error, best joint/singularity margin, then smaller base spacing.

## Data contract

Each record contains its stage, mount pose, sampled source rows, continuity
status, coverage, longest failure window, collision classes/counts, structural
crossing depth/count, gripper overlap depth/count, pose errors, margins and a
machine-readable rejection reason. The selected record must include
`status: valid_selection` and evidence that the full timeline was audited.

No code may select a mount by manually copying coordinates without the matching
full-audit fingerprint.

## Tests

- A mount with perfect IK but one state collision is rejected.
- A mount with one midpoint swept collision is rejected.
- A disconnected safe segment after an IK gap cannot restore continuous
  coverage by resetting history.
- Structural arm-chain reversal beyond 30 mm is rejected without mesh contact.
- Contact-free terminal gripper overlap up to 80 mm is accepted.
- A full-audit-free record cannot become `valid_selection`.
- Deterministic candidate generation and tie-breaking are stable.
- Existing single-arm isolation tests and the complete factory-bimanual suite
  remain green.

## Performance boundary

Dense/full IK is limited to hard-gate survivors. The sparse stage finds an
ordering; local refinement is not run at every trajectory point. Candidate,
pair and finalist counts remain explicit bounded configuration values.
