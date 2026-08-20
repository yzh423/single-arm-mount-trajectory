import numpy as np

from factory_bimanual.mount_topology import (
    MountTopologyConfig,
    evaluate_mount_topology_positions,
)


def _evaluate(structural_left, structural_right, tcp_left, tcp_right):
    return evaluate_mount_topology_positions(
        left_base=np.array([-1., 0., 0.]),
        right_base=np.array([1., 0., 0.]),
        left_structural=np.asarray(structural_left, float),
        right_structural=np.asarray(structural_right, float),
        left_tcp=np.asarray(tcp_left, float),
        right_tcp=np.asarray(tcp_right, float),
        config=MountTopologyConfig(),
    )


def test_rejects_structural_order_reversal_over_30_mm():
    report = _evaluate(
        [[.016, 0., .5], [-.4, 0., .5]],
        [[-.015, 0., .5], [.4, 0., .5]],
        [-.02, 0., .6], [.02, 0., .6])

    assert not report.valid
    assert np.isclose(report.maximum_structural_crossing_m, .031)
    assert report.structural_crossing_count == 1


def test_accepts_contact_free_terminal_overlap_up_to_80_mm():
    report = _evaluate(
        [[-.4, 0., .5], [-.2, 0., .5]],
        [[.4, 0., .5], [.2, 0., .5]],
        [.04, 0., .6], [-.04, 0., .6])

    assert report.valid
    assert np.isclose(report.gripper_overlap_m, .08)
    assert report.gripper_overlap_count == 1


def test_rejects_terminal_overlap_over_80_mm():
    report = _evaluate(
        [[-.4, 0., .5], [-.2, 0., .5]],
        [[.4, 0., .5], [.2, 0., .5]],
        [.0405, 0., .6], [-.0405, 0., .6])

    assert not report.valid
    assert np.isclose(report.gripper_overlap_m, .081)


def test_thresholds_are_inclusive_and_zero_base_distance_is_invalid():
    boundary = _evaluate(
        [[.015, 0., .5], [-.2, 0., .5]],
        [[-.015, 0., .5], [.2, 0., .5]],
        [.04, 0., .6], [-.04, 0., .6])
    assert boundary.valid

    try:
        evaluate_mount_topology_positions(
            left_base=np.zeros(3), right_base=np.zeros(3),
            left_structural=np.zeros((2, 3)),
            right_structural=np.zeros((2, 3)),
            left_tcp=np.zeros(3), right_tcp=np.zeros(3))
    except ValueError as error:
        assert "base" in str(error)
    else:
        raise AssertionError("coincident bases must be rejected")
