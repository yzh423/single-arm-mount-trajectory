# Claim-to-source ledger

| Claim | Evidence type | Source |
|---|---|---|
| 108 formal shards, 409,760 valid paired frames | Repository experiment evidence | `raw_source_audit_summary.json` |
| Weighted conditioned-target coverage is 11.8811%; fixed-tool-mapped unconditioned source-target coverage is 1.8313% | Recomputed from formal actual TCP arrays and reconstructed target contracts | `raw_source_shard_audit.csv`; `scripts/audit_piperx_fixed_time_evidence.py` |
| Conditioning exceeds the 1 mm / 0.5° strict budget in 108/108 shards | Direct target-track comparison | `raw_source_shard_audit.csv` |
| Formal execution has 0 frame collisions, 0 swept-edge collisions, and 0 topology-invalid frames | Formal NPZ safety arrays | `raw_source_audit_summary.json`; existing bundle validator |
| 35/108 shards pass both 1 rad/s and 4 rad/s²; the same 35 pass 3 rad/s and 5 rad/s²; all have zero coverage | Formal joint arrays on unchanged source timestamps | `raw_source_shard_audit.csv` |
| Maximum measured source TCP segment rates are 15.054 m/s and 45.176 rad/s | Raw target differences divided by exact adjacent source intervals | `raw_source_audit_summary.json` |
| Timing/parameterization is the mechanism for satisfying path velocity and acceleration constraints | Peer-reviewed primary paper | Pham & Pham, IEEE T-RO 2018, DOI 10.1109/TRO.2018.2819195 |
| Point-wise IK can be discontinuous and should account for velocity, acceleration, jerk, self-collision, and singularity | Peer-reviewed primary paper | Rakita et al., RSS 2018, DOI 10.15607/RSS.2018.XIV.043 |
| Concurrent Jacobian and SQP solvers improve bounded generic IK robustness | Peer-reviewed primary paper | Beeson & Ames, Humanoids 2015, DOI 10.1109/HUMANOIDS.2015.7363472 |
| Strict hierarchies of equality and inequality constraints are practical for online robot motion generation | Peer-reviewed primary paper | Escande et al., IJRR 2014, DOI 10.1177/0278364914521306 |
| Sequential convex trajectory optimization supports continuous collision checking | Peer-reviewed primary paper | Schulman et al., IJRR 2014, DOI 10.1177/0278364914528132 |
| Selectively damped least squares adjusts damping by singular direction | Peer-reviewed primary paper | Buss & Kim, Journal of Graphics Tools 2005, DOI 10.1080/2151237X.2005.10129202 |
| Piper J1–J6 configurable speed and acceleration ceilings are 3 rad/s and 5 rad/s² | Manufacturer primary documentation | AgileX Robotics Piper SDK V2 `INTERFACE_V2.MD` |
