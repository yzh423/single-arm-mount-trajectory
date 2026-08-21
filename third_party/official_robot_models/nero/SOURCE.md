# Nero model provenance

- Selected embodiment: seven-axis Nero arm with its fixed gripper assembly.
- Vendored source snapshot: the unscaled `nero/nero` package supplied with the
  original workspace model collection.
- URDF SHA-256: `9ddd8bc3f614b1dea55d7fc65eb5d75ae121c089dbf8d4f033e11ec08ad1ad4`.
- Meshes: 24 source DAE/STL assets copied byte-for-byte into `meshes/` and
  verified after copying.
- Source status: no independently verifiable public upstream revision was
  included in the supplied package, so this is identified as a pinned vendor
  snapshot rather than falsely claiming a public official repository commit.
- Geometry policy: native joint origins, inertials, limits, and mesh sizes are
  used directly; no arm-length normalization is applied.
- TCP policy: the source exposes a gripper assembly but no named TCP frame, so
  the project-wide fixed 130 mm tool convention is attached to `gripper_base`.

