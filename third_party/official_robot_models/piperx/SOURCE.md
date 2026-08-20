# PiPER-X model provenance

- Selected embodiment: AgileX PiPER-X, not the earlier PiPER model.
- Official repository: `https://github.com/agilexrobotics/agx_arm_urdf`.
- Pinned revision: `f6642ce0d7872c686f29c99e9e10cd23d1d49313`.
- Official source subtree: `piper_x/` (not the older `piper/` model).
- Expanded source snapshot: `GoodGoodArmDayDayUp-Learning/Assets/piperx/urdf/PiperX.urdf`.
- Source artifact SHA-256: `9d5b0490df5d3469fa08fae355d5dbda3761af5dace5624d49eda56896b72ecb`.
- Vendored XML SHA-256: `9c5433d94fdad29c050d00c1cdd86e02807f87b51abc429697129d2a6232f663`.
- Meshes: byte-identical source meshes are vendored beside this file under
  `meshes/`; all runtime resolution stays inside `third_party`.
- TCP: the model-defined `ee_frame`, fixed 115 mm from `gripper_base`.
- Factory-bimanual normalization scale: `1.0` (native physical dimensions).
- Vendor identity reference: AgileX Robotics' PiPER-X launch announcement,
  which distinguishes PiPER-X by its revised J4/J5 geometry.

The XML was formatting-normalized when vendored; joint origins, limits,
inertials, mesh identities, and fixed-frame transforms remain unchanged. The
runtime uses these native dimensions directly and performs no morphology
rescaling.
