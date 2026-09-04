# Third-party robot model notice

This directory contains upstream or vendor-supplied robot descriptions used for reproducible geometry, IK, collision, and rendering experiments. These assets are not relicensed by this repository. Each model remains subject to its own upstream license or vendor terms.

The following snapshots include an upstream license in their vendored tree: Franka, I2RT, Kinova, OpenArm, PiPER, Universal Robots, xArm, and their derived copies under `official_derived/`.

PiPER-X records its public repository and pinned revision in `piperx/SOURCE.md`. Nero and Willow record the supplied snapshot identity and hashes in their respective `SOURCE.md` files, but the supplied packages did not include an independently verifiable redistribution license. Their redistribution status therefore remains unresolved; no license is inferred here.

The ARX and Doosan snapshot directories do not currently contain a complete upstream license/provenance record. They are retained to preserve the model inputs used by the experiments, but downstream users must verify the applicable vendor terms before redistribution or commercial use.

Corrupt redundant ZIP files were removed from the publication. The former `incomplete/` directory was also removed because it contained failed or partial downloads of PiPER, Universal Robots, and Doosan sources rather than additional qualified robot models; those models remain in their complete expanded directories. The expanded model trees remain authoritative, and every remaining top-level ZIP is covered by an archive-integrity test.
