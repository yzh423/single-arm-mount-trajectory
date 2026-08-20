from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BimanualTaskFrames:
    left: torch.Tensor
    right: torch.Tensor


def mirrored_mount_transforms(spacing_m: torch.Tensor, base_y_m: torch.Tensor,
                              base_z_m: torch.Tensor, roll_rad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirrored dual-arm mounts; roll is about world Y (forward axis)."""
    spacing_m, base_y_m, base_z_m, roll_rad = torch.broadcast_tensors(
        spacing_m, base_y_m, base_z_m, roll_rad)
    dtype, device = spacing_m.dtype, spacing_m.device

    def make(sign: float) -> torch.Tensor:
        angle = sign * roll_rad; cosine, sine = torch.cos(angle), torch.sin(angle)
        transform = torch.eye(4, dtype=dtype, device=device).expand(angle.shape + (4, 4)).clone()
        transform[..., 0, 0] = cosine; transform[..., 0, 2] = sine
        transform[..., 2, 0] = -sine; transform[..., 2, 2] = cosine
        transform[..., 0, 3] = sign * .5 * spacing_m
        transform[..., 1, 3] = base_y_m; transform[..., 2, 3] = base_z_m
        return transform

    return make(-1.0), make(1.0)


def quaternion_wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(quaternion.shape[:-1] + (3, 3))


def make_transform(xyz: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    transform = torch.eye(4, dtype=xyz.dtype, device=xyz.device).expand(xyz.shape[:-1] + (4, 4)).clone()
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = xyz
    return transform


def canonical_egodex_targets(
    left_relative_xyz: torch.Tensor,
    left_relative_quat_wxyz: torch.Tensor,
    right_relative_xyz: torch.Tensor,
    right_relative_quat_wxyz: torch.Tensor,
    *,
    anchor_left_xyz=(-0.18, -0.45, 0.35),
    anchor_right_xyz=(0.18, -0.45, 0.35),
) -> BimanualTaskFrames:
    """Map EgoDex relative wrist motion onto a common tabletop grasp frame.

    The gripper/tool +Z axis points down, +X points toward the operator's
    forward direction (-world Y). Relative translations are correctly rotated
    by this initial grasp frame instead of being added in an unrelated world
    frame. Both robots therefore receive exactly the same ideal TCP motion.
    """
    dtype, device = left_relative_xyz.dtype, left_relative_xyz.device
    anchor_rotation = torch.tensor(((0.0, -1.0, 0.0),
                                    (-1.0, 0.0, 0.0),
                                    (0.0, 0.0, -1.0)), dtype=dtype, device=device)

    def convert(xyz, quaternion, anchor_xyz):
        relative_rotation = quaternion_wxyz_to_matrix(quaternion)
        world_xyz = torch.as_tensor(anchor_xyz, dtype=dtype, device=device) + xyz @ anchor_rotation.T
        world_rotation = anchor_rotation @ relative_rotation
        return make_transform(world_xyz, world_rotation)

    return BimanualTaskFrames(convert(left_relative_xyz, left_relative_quat_wxyz, anchor_left_xyz),
                              convert(right_relative_xyz, right_relative_quat_wxyz, anchor_right_xyz))


def world_to_base(targets: torch.Tensor, base_world: torch.Tensor) -> torch.Tensor:
    """Transform world TCP targets into a robot base frame."""
    rotation = base_world[:3, :3]
    translation = base_world[:3, 3]
    result = targets.clone()
    result[..., :3, :3] = rotation.T @ targets[..., :3, :3]
    result[..., :3, 3] = (targets[..., :3, 3] - translation) @ rotation
    return result


def world_to_base_population(targets: torch.Tensor, base_world: torch.Tensor) -> torch.Tensor:
    """World targets [T,4,4] to D independently mounted base frames."""
    rotation = base_world[:, :3, :3]; translation = base_world[:, :3, 3]
    result = targets[None].expand(base_world.shape[0], -1, -1, -1).clone()
    result[..., :3, :3] = rotation[:, None].transpose(-1, -2) @ targets[None, :, :3, :3]
    delta = targets[None, :, :3, 3] - translation[:, None]
    result[..., :3, 3] = (rotation[:, None].transpose(-1, -2) @ delta[..., None]).squeeze(-1)
    return result
