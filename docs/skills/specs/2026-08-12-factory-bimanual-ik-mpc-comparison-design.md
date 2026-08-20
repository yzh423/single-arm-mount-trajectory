# Factory Bimanual IK and MPC Comparison Design

Date: 2026-08-12

Status: Approved design; implementation pending

## 1. Objective

Evaluate four same-model dual-arm systems in MuJoCo on two complete factory handheld demonstrations:

- two xArm6 robots;
- two Franka Panda robots, with all seven joints active and J3 not locked;
- two I2RT YAM robots;
- two UR5 robots.

Tasks:

- `data/factory/8-11/Screw_Cap/handheld_20260811_162854.csv`;
- `data/factory/8-12/PourRawMaterial/handheld_20260812_111542.csv`.

Only bimanual experiments are in scope. No single-arm experiment is included.

## 2. Source Data Contract

Use each CSV from beginning to end without temporal cropping, task-phase extraction, spatial scaling, or alteration of left/right relative motion.

Measured source properties:

- Screw Cap: 2,524 rows, approximately 30.181 s;
- Pour Raw Material: 5,942 rows, approximately 71.017 s;
- all recorded TCP validity fields are true;
- no consecutive TCP translation jump exceeds 5 cm;
- coordinate frame is `vr_world` throughout.

The source `t` column is authoritative. Duplicate controller frame counters do not authorize dropping CSV rows. Resampling to the controller rate must preserve physical source time and must not shorten stationary intervals.

Pour Raw Material uses the recorded left/right gripper angles. Screw Cap has no gripper-angle fields; it uses a declared fixed gripper opening and records that limitation in every report.

## 3. Task Registration

Each task receives one shared rigid transform from `vr_world` to the MuJoCo task frame:

\[
T^{world}_{hand}(t) = T^{register}_{task} T^{vr}_{hand}(t)
\]

The transform may include:

- one fixed coordinate-basis conversion supported by source semantics;
- one yaw rotation around gravity;
- one translation that places the complete bimanual trajectory relative to the table.

It may not include scale, reflection, per-hand offsets, per-robot changes, or time-varying alignment. Both hands use the same task transform, and all four robot types use the same registered task trajectory.

The transform must be recorded, invertible, and accompanied by fixed-scale top/front/side trajectory plots. Unverified source conventions remain explicitly labelled and prevent an external-release claim.

## 4. Fixed Dual-Arm Scenes

Construct four scenes containing a common table and two upright robots of the same model. Bases are placed symmetrically along the table rear edge. Base tilt, roll, and yaw are fixed; only the symmetric scalar spacing differs by robot type.

Each robot type selects one spacing that is reused for both tasks. Spacing selection is a bounded one-dimensional engineering scan, not a mount-pose search.

The deterministic spacing objective is lexicographic:

1. maximize the worse-task bimanual synchronous coverage;
2. minimize the worse-task longest consecutive failure duration;
3. minimize cross-arm collision frames;
4. minimize table/base collision frames;
5. minimize aggregate TCP error;
6. prefer the smaller physically installable spacing.

The selected spacing, candidate range, resolution, tie count, and installation clearances are reported.

Existing native robot geometry, joint limits, TCP definitions, and visual meshes are preserved. xArm6 and UR5 reuse existing dual-arm scene patterns. Franka Panda and I2RT YAM require scene/controller adapters. The external `doosan_teleop` EasyIK/MPC implementation is hard-coded to six joints, so the factory package contains an isolated native-DOF port of the same algorithms. The port replaces fixed `6` dimensions with each arm's model-derived DOF, but does not change objectives, filtering, candidate policy, safety rollback semantics, or controller tuning equations. External controller files and the single-arm project remain unmodified.

## 5. Solvers Under Comparison

### 5.1 Solver A: strict bimanual rolling multibranch IK

Extend the existing single-arm strict rolling multibranch concept to a joint bimanual state:

\[
Q(t) = [q_L(t), q_R(t)]
\]

For each frame, generate multiple valid IK candidates for each side and select left/right branch pairs with a bounded beam/frontier. Candidate and transition validity include:

- both TCP pose errors;
- both joint-limit sets;
- timestamp-consistent joint velocity and jump limits;
- enabled self-collision policy;
- cross-arm collision;
- robot/table collision;
- robot/opposite-base collision;
- transition collision between consecutive paired states.

This produces an offline kinematic feasibility path. It does not claim controller execution success.

### 5.2 Solver B: existing DualArmEasyIKPVT

Use the project's existing `DualArmEasyIKPVT` algorithm through the isolated native-DOF port in the same dual-arm scene. For 6-DOF robots, parity tests compare the port against the external implementation on identical scenes, states, targets, and configuration. For Panda, the same equations operate over seven joint columns. Both arms are enabled and receive synchronized targets. Each source target receives a declared, identical convergence budget across robot types.

EasyIK output is audited using the same external MuJoCo TCP, joint, time, and collision evaluator used for Solver A. Internal solver convergence alone is not the reported success metric.

### 5.3 Existing DualArmMPCPVT

Use the existing `DualArmMPCPVT` algorithm and `MPCConfig` through the isolated native-DOF port; do not redesign the MPC. Six-DOF parity tests must match the external controller within declared numerical tolerance. The MPC follows the original registered TCP trajectories, not Solver A's joint path.

The only A/B MPC variable is the initial bimanual joint branch:

- MPC-A starts from Solver A's valid first-frame bimanual state;
- MPC-B starts from EasyIK's valid first-frame bimanual state.

All later TCP targets, timing, control frequency, filtering, limits, collision parameters, seeds, and reporting are identical.

## 6. Conditional MPC Comparison

For every robot/task pair, compare the valid first-frame states from A and EasyIK using wrapped joint differences:

\[
\Delta q_{max} = \max(|wrap(q_A-q_B)|)
\]

Rules:

- if either initializer fails, run MPC only from each initializer that produced a valid collision-free first state and record the missing mode;
- if both succeed and `delta_q_max < 5 degrees` for both arms, treat the initial branches as equivalent and run one canonical MPC trial;
- if both succeed and `delta_q_max >= 5 degrees` for either arm, run both MPC-A and MPC-B;
- do not infer equivalence from TCP error alone.

The canonical equivalent-branch MPC trial starts from the Solver A state, because it has passed the stricter multibranch/transition contract. Its report records that EasyIK produced an equivalent branch.

Thus the mandatory IK matrix has 16 runs:

\[
4 robots \times 2 tasks \times 2 IK solvers = 16
\]

The MPC matrix contains between 8 and 16 runs depending on branch equivalence and initializer validity.

## 7. Common Success and Diagnostic Metrics

The common strict frame threshold is:

- position error no more than 1 mm per TCP;
- orientation error no more than 1.5 degrees per TCP;
- no disallowed joint jump or velocity violation;
- no enabled collision.

Report separately:

- left-arm strict coverage;
- right-arm strict coverage;
- bimanual synchronous coverage, requiring both arms valid on the same frame;
- full-episode bimanual success;
- longest consecutive synchronous failure in frames and seconds;
- position and orientation RMSE/p95/max for each arm;
- joint jumps and velocity violations;
- self, table, base, and cross-arm collision frames;
- solver failure and recovery/reinitialization counts;
- MPC safety rollbacks, target lag, control timing, and actual-TCP tracking;
- solver wall time and declared budgets.

IK planning success and MPC execution success are separate result columns and may not be substituted for one another.

## 8. Failure Handling

The pipeline must preserve the full source timeline even when a solver fails. Failed targets receive explicit reason codes; output arrays remain aligned with CSV row indices.

Allowed diagnoses include:

- position unreachable;
- full-pose infeasible;
- joint-limit failure;
- branch lost;
- velocity/jump violation;
- self collision;
- cross-arm collision;
- table/base collision;
- EasyIK convergence exhausted;
- MPC rollback/hold;
- unsupported model/controller contract;
- unknown.

No mode may silently hold a pose while counting it as successful. Recovery and target filtering must be visible in the metrics.

## 9. Verification and Visual Evidence

Before running the full matrix:

1. render each new dual-arm scene at true scale and inspect base, TCP, joint, and collision ownership;
2. validate obvious colliding and non-colliding synthetic states;
3. verify Panda has 7 active joints per arm and I2RT uses the audited TCP axis/length;
4. render both registered task trajectories without robots in shared fixed-scale views;
5. run a short prefix through A, EasyIK, and MPC for xArm6;
6. then validate UR5, Panda, and I2RT adapters individually.

Every final robot/task pair produces an original-time validation video showing:

- both target TCPs;
- both actual TCPs;
- robot geometry and table;
- mode, source time, per-side state, and failure reason;
- collision/rollback state.

Failed experiments are rendered and retained. Worst-result videos are opened and inspected before numerical reporting.

## 10. Outputs

Use a new experiment namespace that does not overwrite single-arm or historical tilted results. Store:

- normalized source task files and registration manifests;
- fixed-scale registration figures;
- four dual-arm scene manifests;
- spacing scan results;
- A and EasyIK caches/audits;
- branch-equivalence decisions;
- conditional MPC-A/MPC-B caches/audits;
- original-time validation videos;
- a matrix JSON/CSV and human-readable report;
- provenance, software/config fingerprints, and visual-inspection records.

Final conclusions use the phrase `best observed under the declared fixed-spacing and solver/control budgets`; they do not claim global optimality.

## 11. Out of Scope

- single-arm experiments;
- arbitrary six-dimensional mount search;
- task cropping or trajectory scaling;
- redesigning EasyIK or MPC objectives, candidate policy, filtering, or rollback semantics; native-DOF dimensional generalization is explicitly in scope;
- force/torque control;
- material, bottle, cap, container, or fluid contact simulation;
- claiming real-hardware feasibility from MuJoCo alone.
