# PiperX controller-event v4 evidence note

## Scope

This evidence set diagnoses the low fixed-time coverage without changing the
source duration or the 1 mm / 0.5 degree acceptance gate. It separates:

- host polling rows from synchronized controller update events;
- the fixed tool-frame source target from smoothed or time-varying wrist targets;
- geometric MuJoCo following from hardware-dynamics feasibility; and
- complete baseline runs from the equal-length 300-event four-mount comparison.

## Complete baseline results

| Task | Host poll rows | Controller events | Raw-target coverage | Collision / edge / topology |
|---|---:|---:|---:|---:|
| Seal_Bag/161504 | 2439 | 1872 | 100.00% | 0 / 0 / 0 |
| Fold_Box/161044 | 1478 | 755 | 94.83% | 0 / 0 / 0 |

Controller-event time is the paired left/right `receive_monotonic_s` timestamp.
Rows are collapsed only when both controller frame counters remain unchanged.
Every retained event stores its original CSV poll-row index.

## Four-mount comparison

The equal-length comparison uses the first 300 controller events of each task.

| Task | Baseline | Upright table | Horizontal wall | Inverted |
|---|---:|---:|---:|---:|
| Seal_Bag/161504 | 100.00% | 59.67% | 0.00% | 0.00% |
| Fold_Box/161044 | 87.00% | 41.67% | 8.33% | 0.00% |

All eight comparison shards have zero frame collisions, zero swept-edge
collisions, and zero topology-invalid frames. A failed frame is a safe HOLD and
is not counted as following.

## Evidence boundary

These results establish improved raw-target geometric tracking for two
representative tasks. They do not establish complete following for all 27
recordings and do not establish hardware executability. The full baseline joint
paths still exceed the Piper SDK V2 configurable 3 rad/s velocity and 5 rad/s^2
acceleration ceilings. This proves that the published joint paths are not
hardware-ready; it does not prove that no alternate continuous IK branch can
meet the same raw targets. Spatially retargeted results, if added later, must
be reported separately from raw-target strict following.

## Literature used for design decisions

- Gao et al., *Motion Mappings for Continuous Bilateral Teleoperation*, IEEE
  RA-L 6(3), 2021. DOI: 10.1109/LRA.2021.3068924.
- Wen et al., *Collaborative Bimanual Manipulation Using Optimal Motion
  Adaptation and Interaction Control*, IEEE Robotics & Automation Magazine
  31(4), 2024. DOI: 10.1109/MRA.2023.3270222.
- Rakita et al., *A Motion Retargeting Method for Effective Mimicry-based
  Teleoperation of Robot Arms*, HRI 2017. DOI: 10.1145/2909824.3020254.
- Rakita et al., *RelaxedIK*, RSS 2018. DOI: 10.15607/RSS.2018.XIV.043.
- Pham and Pham, *TOPP-RA*, IEEE T-RO 34, 2018.
  DOI: 10.1109/TRO.2018.2819195.
- AgileX Robotics, Piper SDK V2 interface documentation:
  https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD
