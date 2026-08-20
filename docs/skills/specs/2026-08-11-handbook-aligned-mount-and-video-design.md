# Handbook-Aligned Mount Search, Verification, and Video Design

Date: 2026-08-11

Status: Approved direction; implementation pending

## 1. Objective

Make the project conform to the evidence and verification rules in `E:\YZH123123\handbook`, while preserving the project's yaw-only mount requirement and strict tracking tolerances.

The change must:

- search only mount `x`, `y`, `z`, and `yaw`; tilt and roll remain exactly zero;
- reduce repeated IK work during ranking and local refinement;
- use one continuous-path evaluator for dense selection and final reporting;
- continue searching a bounded set of challengers after the first successful solution;
- distinguish intrinsic pose infeasibility from branch-tracking failure;
- separate evidence-preserving validation videos from dwell-compressed showcase videos;
- make every reported result traceable to its data, conventions, solver contract, cache, and viewed render.

## 2. Handbook Rules Adopted as Project Baseline

### 2.1 Mandatory rules

1. No score may be treated as trusted until the dataset and coordinate conventions have a falsifiable verification record.
2. Selection and reporting must use the same success definition and continuous evaluator.
3. Every formal result must have an opened and inspected render of the actual reached TCP; the worst relevant episode is the default diagnostic episode.
4. Checks must be capable of failing and must state their effective sensitivity or control case where applicable.
5. Aggregate reports must include mean, count above the declared threshold, and p10. A single-episode minimum is not an arm property.
6. Ties must use a deterministic tie-break and report the tie count.
7. Solver scores are lower-bound estimates. Budget sensitivity and observed search noise must be reported.
8. Episode filtering must have one authoritative gate. Dropped episodes and repair counts must be explicit.
9. Long episodes must not be silently truncated. Any padding must carry a validity mask.
10. Unmodelled risks and unverified assumptions must be explicit.

### 2.2 Project-specific decisions

- Search coordinates are `[x, y, z, yaw]`; exported six-coordinate mounts are `[x, y, z, 0, yaw, 0]`.
- Strict tracking tolerances remain 1 mm position and 1.5 degrees orientation until a separate controlled tolerance study approves a change.
- Handbook shortlist size 400 is not a universal constant. It applies to a batched GPU architecture where candidates are nearly free. This project will measure GPU ranking and strict CPU/MuJoCo refinement separately.
- The local candidate target is approximately 64, subject to measured recall and runtime.

## 3. Provenance and Admission Gates

Add a project baseline document and machine-readable verification records with separate states for:

- dataset episode health;
- coordinate frame and rotation layout;
- approach axis and sign;
- recorded point/TCP meaning;
- arm DOF, tip link, joint limits, mesh availability, and URDF source date;
- common flange-to-TCP policy;
- modelled and unmodelled collision classes;
- human render inspection.

Allowed states are `VERIFIED`, `UNVERIFIED`, `NOT_APPLICABLE`, and `REJECTED`. A formal report must not silently promote `UNVERIFIED` inputs. Overrides must be explicit, recorded, and visually labelled.

Dataset gates remain dataset-specific. The implementation must not blindly copy duration-normalized or absolute thresholds across datasets. Teleports, empty recordings, repairs, and dropped episodes must be printed and stored.

## 4. Search Architecture

### 4.1 Stage A: cheap admissibility gate

Generate only yaw-only candidates. Reject or penalize candidates using inexpensive tests:

- install bounds and table-edge rules;
- base and coarse environment collision;
- position workspace feasibility;
- coarse orientation feasibility where cheap and demonstrably valid.

This stage ranks possibilities; it must not claim continuous episode success.

### 4.2 Stage B: broad GPU ranking

GPU ranking operates on `[x, y, z, yaw]` only. It must not optimize a tilted pose and later erase tilt/roll.

Representative evidence consists of 3–5 short contiguous windows chosen to cover task phases and difficult motion. Disconnected frames must not be passed to a sequential solver as if adjacent. If independent frames are used, the metric must be explicitly labelled independent-pose feasibility and must not include a sequential jump interpretation.

The ranking key is deterministic and records all components. It includes at least coverage, longest failure run inside windows, error terms, install feasibility, and collision state.

### 4.3 Stage C: local refinement

Expand diverse high-ranked regions, not only the single best point. Generate approximately 64 local candidates across those regions. Reuse prior feasibility and window results where their solver contract and frame identity exactly match.

Local refinement establishes an ordering. It does not run full dense IK for every grid point.

### 4.4 Stage D: continuous dense promotion

Promote only the leading candidates to the authoritative evaluator. Dense evaluation must process every valid frame in chronological order with rolling multibranch state. It may use chunking or padding for efficiency, but cannot merge independently solved frame subsets into a synthetic episode result.

Dense success means exactly the same thing as final reported success:

- position tolerance satisfied;
- orientation tolerance satisfied;
- branch-jump constraint satisfied with time/stride-consistent semantics;
- enabled collision constraints satisfied;
- all valid episode frames processed in order.

### 4.5 Stage E: post-success verification

After the first successful dense candidate, continue a bounded confirmation search. At minimum inspect:

- challengers from three distinct search regions where available;
- the refined neighbourhood of the incumbent;
- the two highest-ranked remaining challengers;
- a higher-budget replay of the incumbent and competitive challenger set.

Stopping is allowed after the configured challenger/region/time budget is consumed and no candidate improves the deterministic objective. Stop reasons and counts are mandatory output fields.

## 5. Authoritative Evaluation and Ranking

There is one authoritative continuous evaluator shared by dense promotion, final selection, cache generation, reporting, and rendering. A result cannot be overwritten by a second solver with a different contract.

Candidate ranking is lexicographic and deterministic:

1. full-episode success;
2. valid-frame follow coverage;
3. negative longest consecutive failure duration;
4. negative collision frame count;
5. position error summary;
6. orientation error summary;
7. installation preference and stable coordinate tie-break.

The report records the number of candidates tied under the declared primary objective. Exact floating-point equality is not assumed; tie tolerances must be declared and tested.

## 6. Solver-Failure Diagnosis

`solver_failure` is not a sufficient final diagnosis. For failed runs, diagnostics classify frames into:

- position unreachable;
- full-pose intrinsically infeasible under current joint limits;
- collision constrained;
- rolling branch lost or unable to recover;
- jump-limit violation;
- numerical/iteration exhaustion;
- unknown.

The classifier uses falsifiable controls:

- position-only feasibility;
- independent multi-restart full-pose IK at representative failed frames;
- successful control frames under the same budget;
- targeted joint-limit relaxation plus unrelated-joint controls when making causal limit claims.

For ARX-X5 specifically, the design must preserve the observed distinction between initially infeasible poses and later poses that independent IK can solve but rolling tracking loses.

## 7. Reporting Contract

Every run artifact records:

- dataset/task/episode identifiers and gate version;
- arm and model provenance;
- exact mount with tilt and roll asserted zero;
- tolerances, stride/time semantics, solver seed and budget;
- stage candidate counts and actual strict IK frame solves;
- first-success point and post-success work;
- tie count and deterministic tie-break;
- per-episode coverage and failure-run duration;
- aggregate mean, threshold count, and p10 when multiple episodes are reported;
- budget-upgrade comparison and observed monotonicity violations/noise;
- cache and renderer contract hashes or equivalent version identifiers;
- render inspection status and inspected file;
- unverified conventions and unmodelled risks.

The formal report must not call a lower-bound solver result a global optimum. It may state that the selected solution is `best found under the declared search and verification budget` and summarize the challenger evidence.

## 8. Video Contract

### 8.1 Validation video

Validation video preserves the original evaluation timeline. It shows the actual reached TCP and target/error state and is suitable for checking stalls, synchronization, branch failures, and failure colouring. It must not compress dwell.

For new dataset/convention admission, produce the handbook-style evidence view: source RGB, synchronized fixed-axis 3D, and shared-scale multi-episode overlay, or record why a source panel is unavailable. The generated file must be opened and its decoded frame count verified.

### 8.2 Showcase video

Showcase video is display-only and cannot be used as timing evidence. Apply the approved policy:

- dwell at or below 0.20 seconds remains unchanged;
- dwell above 0.20 seconds is compressed to 0.10 seconds;
- non-dwell motion remains at real-time speed;
- evaluation cache, scores, and solver trajectory remain unchanged;
- emit a JSON source-to-display timeline mapping;
- visibly label the video as dwell-compressed.

## 9. Required Self-Checks and Tests

The implementation is incomplete unless automated tests cover:

- all generated/exported mounts have zero tilt and roll;
- GPU and local generation accept only four active coordinates;
- disconnected samples cannot be evaluated as one sequential trajectory;
- dense and final evaluation call the same evaluator contract;
- a cached successful result cannot be replaced by a different replay result;
- deterministic ranking and tie count;
- padding masks exclude padded frames from score and failure runs;
- stride/time changes scale jump semantics correctly;
- provenance gating and explicit override behaviour;
- failure-reason classification controls;
- validation timeline unchanged;
- showcase dwell mapping and duration policy;
- renderer refuses failed, stale, or incompatible caches;
- report contains required aggregate, noise, provenance, and inspection fields.

## 10. Acceptance Criteria

1. Existing regression tests remain green.
2. A three-task rerun includes XArm6/open-box-2, Doosan/cap-left, and ARX-X5/cap-left or equivalent approved representatives.
3. No rerun searches or returns non-zero tilt/roll.
4. Dense selected status and final reported status cannot differ because of evaluator-contract changes.
5. The ARX report separates intrinsic full-pose infeasibility from recoverable branch loss.
6. XArm validation video preserves source dwell; showcase video compresses it and includes a mapping JSON.
7. The worst failed episode/configuration is rendered and opened before the final numerical report.
8. Runtime reports GPU ranking time, strict evaluator time, candidate counts, and frame-solve counts separately.
9. The report states the solution as budget-qualified rather than proven global optimum.

## 11. Implementation Boundaries

This work does not silently change task data, tracking tolerances, collision policy, TCP length, or accepted arm models. Any such change requires its own controlled design and evidence. Old tilted results remain historical artifacts and must not be mixed with new yaw-only results.
