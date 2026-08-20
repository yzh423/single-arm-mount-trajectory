# Project Handbook Baseline

This project adopts `E:\YZH123123\handbook` as its evidence and reporting baseline.
The handbook's measured constants are not copied across architectures without a new
measurement, but the following rules are non-negotiable.

## MUST

- The selection metric == reporting metric. Dense selection, cache status, report
  status, and render eligibility use the same chronological evaluator contract.
- Active mount search coordinates are exactly `[x, y, z, yaw]`. Exported six-field
  mounts have tilt and roll equal to zero.
- No formal score is trusted while required data, coordinate, TCP, or arm provenance
  remains silently unverified.
- Every check must be capable of failing; causal claims require a suitable control.
- Before reporting a result, render and inspect the actual reached TCP on the worst episode.
- Multi-episode reports contain mean + threshold count + p10, plus deterministic tie count.
- Long episodes are never silently truncated; padding uses an explicit validity mask.
- Solver scores are lower bounds. Reports state solver budget, observed search noise,
  and that the result is best found under the declared budget rather than globally proven.
- Dataset repairs, dropped episodes, and unmodelled risks are explicit.
- The validation timeline preserves source time and is the timing/synchronization evidence.
- The showcase timeline is display-only; long dwell compression is labelled and accompanied
  by a source-to-display mapping.

## Project Decisions

- Position tolerance is 1 mm and orientation tolerance is 1.5 degrees until a separate
  controlled study approves a change.
- Local refinement targets 64 candidates distributed across diverse regions.
- Full chronological dense IK is reserved for leading candidates and bounded challengers.
- After first success, inspect distinct regions, the incumbent neighbourhood, and top
  challengers before stopping under an explicit budget.

## Architecture-Specific Guidance

- The handbook's shortlist size of 400 belongs to a batched GPU architecture where
  candidate count is nearly free. GPU ranking and strict MuJoCo/CPU costs are measured
  separately here; the number 400 is not a project invariant.
- Handbook example tolerances and TCP constants are evidence examples, not permission to
  silently replace this project's declared contracts.

## Verification States

Allowed provenance states are `VERIFIED`, `UNVERIFIED`, `NOT_APPLICABLE`, and `REJECTED`.
Promotion to `VERIFIED` is an evidence-bearing event. Formal reports cannot silently promote
or omit an `UNVERIFIED` dependency.
