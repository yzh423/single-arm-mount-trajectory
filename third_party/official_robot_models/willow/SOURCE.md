# Willow model provenance

- Selected embodiment: six-axis Ragtime Willow model exported by the
  SolidWorks-to-URDF exporter.
- Vendored source snapshot: `Willow_original.urdf` and its supplied mesh set.
- URDF SHA-256: `bedb1139ea9a298ad5663a5b5c903a8ca2a0f49d1508a48c54d49e3b6f2b5e74`.
- Meshes: 10 source STL assets copied byte-for-byte into `meshes/` and verified
  after copying.
- Source status: the package contains exporter metadata but no public upstream
  repository revision, so this is recorded as a pinned vendor snapshot.
- Geometry policy: the original native joint origins and unscaled meshes are
  authoritative. The former canonical derivative is intentionally not used.
- MuJoCo derivative: `urdf/Willow_mujoco.urdf` differs only by replacing the
  over-200k-face `link3.STL` with the supplied, geometrically identical
  `link3_part1.STL` and `link3_part2.STL` pair for importer compatibility.
- TCP policy: the original description has no named TCP frame; the project-wide
  fixed 130 mm tool is attached along the physical negative-X tool direction of
  `link6`.
