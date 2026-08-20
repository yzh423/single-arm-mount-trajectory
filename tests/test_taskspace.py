import torch

from design_optimization.taskspace import canonical_egodex_targets, mirrored_mount_transforms


def test_zero_relative_pose_equals_canonical_anchor():
    xyz = torch.zeros((3, 3), dtype=torch.float64)
    quat = torch.zeros((3, 4), dtype=torch.float64); quat[:, 0] = 1.0
    targets = canonical_egodex_targets(xyz, quat, xyz, quat)
    expected_left = torch.tensor((-0.18, -0.45, 0.35), dtype=torch.float64)
    expected_right = torch.tensor((0.18, -0.45, 0.35), dtype=torch.float64)
    torch.testing.assert_close(targets.left[:, :3, 3], expected_left.expand(3, 3))
    torch.testing.assert_close(targets.right[:, :3, 3], expected_right.expand(3, 3))
    assert torch.all(torch.det(targets.left[:, :3, :3]) > 0.999999)


def test_relative_translation_is_expressed_in_grasp_frame():
    xyz = torch.tensor(((0.1, 0.0, 0.0),), dtype=torch.float64)
    quat = torch.tensor(((1.0, 0.0, 0.0, 0.0),), dtype=torch.float64)
    targets = canonical_egodex_targets(xyz, quat, xyz, quat)
    # Local +X is canonical world -Y.
    torch.testing.assert_close(targets.left[0, :3, 3], torch.tensor((-0.18, -0.55, 0.35), dtype=torch.float64))


def test_mirrored_mount_roll_endpoints():
    scalar = torch.tensor([.7], dtype=torch.float64); zero = torch.zeros(1, dtype=torch.float64)
    left, right = mirrored_mount_transforms(scalar, zero, zero, torch.tensor([torch.pi / 2]))
    torch.testing.assert_close(left[0, :3, 3], torch.tensor((-.35, 0., 0.), dtype=torch.float64))
    torch.testing.assert_close(right[0, :3, 3], torch.tensor((.35, 0., 0.), dtype=torch.float64))
    # Local base +Z points outward in opposite world-X directions at 90 degrees.
    torch.testing.assert_close(left[0, :3, 2], torch.tensor((-1., 0., 0.), dtype=torch.float64), atol=1e-7, rtol=0)
    torch.testing.assert_close(right[0, :3, 2], torch.tensor((1., 0., 0.), dtype=torch.float64), atol=1e-7, rtol=0)
