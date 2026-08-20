"""Single source of truth for the thirteen-arm kinematic experiment."""

COMMON_TOOL_LENGTH_M = 0.130
# Legacy/default direction only. Canonical models supply their actual flange axis.
DEFAULT_FLANGE_TCP_TRANSLATION_M = (0.0, 0.0, COMMON_TOOL_LENGTH_M)
PANDA_LOCKED_J3_RANGE_RAD = (-0.0001, 0.0001)
MOUNT_ROTATION_ORDER = "Rz(yaw) @ Ry(tilt_pitch) @ Rx(roll)"
