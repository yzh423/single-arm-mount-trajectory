from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import mujoco

from factory_bimanual.complete_follow import (
    CompleteFollowRunner,
    retime_complete_source_path,
)
from factory_bimanual.piperx_recommended import load_recommended_config
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene
from factory_bimanual.strict_bimanual_ik import IKCandidate
from factory_bimanual.trajectory_conditioning import bounded_savgol_se3


def test_complete_retiming_preserves_every_source_target_in_order():
    source = np.zeros((3, 12))
    source[1, 0] = 0.65
    source[2, 0] = 0.70

    result = retime_complete_source_path(
        source,
        np.asarray([0.0, 1 / 60, 2 / 60]),
        periodic=np.zeros(12, dtype=bool),
        branch_guard_rad=0.30,
    )

    assert result.source_reached.tolist() == [True, True, True]
    assert result.fixed_time_accepted.tolist() == [True, False, False]
    assert result.execution_source_index.tolist() == [0, 1, 1, 1, 2]
    assert result.execution_state.tolist() == [
        "FOLLOW", "RETIMED_TRANSITION", "RETIMED_TRANSITION",
        "FOLLOW_RETIMED", "FOLLOW",
    ]
    assert result.execution_q[-1] == pytest.approx(source[-1])
    assert result.source_execution_index.tolist() == [0, 3, 4]
    assert result.inserted_transition_frames == 2


def test_complete_retiming_uses_short_periodic_joint_delta():
    source = np.zeros((2, 2))
    source[0, 0] = np.pi - 0.05
    source[1, 0] = -np.pi + 0.05

    result = retime_complete_source_path(
        source,
        np.asarray([0.0, 0.1]),
        periodic=np.asarray([True, False]),
        branch_guard_rad=0.30,
    )

    assert result.fixed_time_accepted.all()
    assert len(result.execution_q) == 2
    assert abs(result.execution_q[1, 0] - result.execution_q[0, 0]) < 0.11


def test_complete_retiming_stretches_time_for_velocity_and_acceleration():
    source = np.asarray([[0.0], [0.2], [0.0]])
    result = retime_complete_source_path(
        source,
        np.asarray([0.0, 0.1, 0.2]),
        periodic=np.asarray([False]),
        branch_guard_rad=0.30,
        maximum_velocity_rad_s=1.0,
        maximum_acceleration_rad_s2=4.0,
    )

    assert result.time_scale > 1.0
    assert result.maximum_velocity_rad_s <= 1.0 + 1e-12
    assert result.maximum_acceleration_rad_s2 <= 4.0 + 1e-12
    assert result.fixed_time_accepted.tolist() == [True, False, False]
    assert result.execution_q[:, 0].tolist() == pytest.approx([0.0, 0.2, 0.0])


def test_complete_retiming_stretches_only_the_dynamic_bottleneck():
    source = np.zeros((11, 1))
    source[5, 0] = 0.2
    source_time = np.arange(11, dtype=float) * 0.1

    result = retime_complete_source_path(
        source,
        source_time,
        periodic=np.asarray([False]),
        branch_guard_rad=0.30,
        maximum_velocity_rad_s=1.0,
        maximum_acceleration_rad_s2=4.0,
    )

    duration = result.execution_time_s[-1] - result.execution_time_s[0]
    assert 1.0 < duration < 2.0
    assert result.maximum_velocity_rad_s <= 1.0 + 1e-12
    assert result.maximum_acceleration_rad_s2 <= 4.0 + 1e-12


def test_complete_retiming_includes_zero_velocity_start_and_stop_boundaries():
    source = np.asarray([[0.0], [0.2]])

    result = retime_complete_source_path(
        source,
        np.asarray([0.0, 0.2]),
        periodic=np.asarray([False]),
        branch_guard_rad=0.30,
        maximum_velocity_rad_s=1.0,
        maximum_acceleration_rad_s2=4.0,
    )

    assert result.execution_time_s[-1] > 0.2
    assert result.maximum_velocity_rad_s <= 1.0 + 1e-12
    assert result.maximum_acceleration_rad_s2 <= 4.0 + 1e-12


def test_complete_retiming_rejects_collision_edge():
    source = np.zeros((2, 12))
    source[1, 0] = 0.4

    with pytest.raises(ValueError, match="collision-free transition"):
        retime_complete_source_path(
            source,
            np.asarray([0.0, 0.1]),
            periodic=np.zeros(12, dtype=bool),
            branch_guard_rad=0.30,
            transition_valid=lambda _previous, _current: False,
        )


def test_bounded_savgol_se3_honors_five_mm_and_one_degree_bounds():
    position = np.zeros((21, 3))
    position[:, 0] = np.linspace(0.0, 0.1, 21)
    position[10, 1] = 0.03
    angles = np.linspace(0.0, 20.0, 21)
    angles[10] += 12.0
    xyzw = Rotation.from_euler("z", angles[:, None], degrees=True).as_quat()
    quaternion = xyzw[:, [3, 0, 1, 2]]

    conditioned = bounded_savgol_se3(
        SimpleNamespace(
            left_position_m=position,
            left_quaternion_wxyz=quaternion,
        ),
        sides=("left",),
        maximum_position_deviation_m=0.005,
        maximum_orientation_deviation_rad=np.deg2rad(1.0),
    )

    position_error = np.linalg.norm(
        conditioned.task.left_position_m - position, axis=1)
    original = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])
    smooth = Rotation.from_quat(
        conditioned.task.left_quaternion_wxyz[:, [1, 2, 3, 0]])
    orientation_error = np.linalg.norm(
        (original.inv() * smooth).as_rotvec(), axis=1)
    assert position_error.max() <= 0.005 + 1e-12
    assert orientation_error.max() <= np.deg2rad(1.0) + 1e-12
    assert conditioned.audit.window == 9
    assert conditioned.audit.polyorder == 3
    assert conditioned.audit.maximum_position_deviation_m == pytest.approx(
        position_error.max())


def test_collision_on_warm_pair_expands_global_candidates_before_selection():
    class Report:
        def __init__(self, valid):
            self.valid = bool(valid)
            self.classes = ()

    class Checker:
        @staticmethod
        def state(left, right):
            values = (float(left[0]), float(right[0]))
            return Report(values == (0.0, 0.0) or values == (0.2, 0.2))

        @staticmethod
        def transition(_previous, current):
            return Checker.state(*current)

    class Generator:
        @staticmethod
        def reset():
            return None

    def candidate(value):
        return IKCandidate(
            q=np.full(6, value), branch_index=0,
            pose_cost=0.0, joint_limit_margin_rad=1.0,
            singularity_margin=1.0, wrist_risk=0.0,
        )

    class Runner(CompleteFollowRunner):
        def __init__(self):
            self.task = SimpleNamespace(time_s=np.asarray([0.0, 0.1]))
            self.generator = Generator()
            self.checker = Checker()
            self.config = SimpleNamespace(
                accept=SimpleNamespace(branch_guard_rad=0.30))
            self.periodic = {
                "left": np.zeros(6, dtype=bool),
                "right": np.zeros(6, dtype=bool),
            }
            self.global_calls = []

        def _warm(self, side, row, reference):
            assert row == 1 and reference is not None
            return candidate(0.1)

        def _global(self, side, row):
            self.global_calls.append((side, row))
            return [candidate(0.0 if row == 0 else 0.2)]

    runner = Runner()
    selected, counts, rescued = runner._source_path()

    assert selected[1].tolist() == pytest.approx([0.2] * 12)
    assert counts["left"][1] == 2 and counts["right"][1] == 2
    assert rescued["left"].tolist() == [True, True]
    assert rescued["right"].tolist() == [True, True]


def test_complete_runner_reaches_every_static_pose_strictly(tmp_path):
    contract = ROBOT_CONTRACTS["piperx"]
    scene = tmp_path / "dual_piperx.xml"
    build_same_model_scene(contract, 0.8, scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    values = {"time_s": np.arange(4) / 60.0}
    mapped = {}
    for side in ("left", "right"):
        site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[site])
        values[f"{side}_position_m"] = np.repeat(
            data.site_xpos[site][None], 4, axis=0)
        mapped[side] = np.repeat(quaternion[None], 4, axis=0)
    task = SimpleNamespace(**values)

    result = CompleteFollowRunner(
        model, task, mapped, load_recommended_config()).run()

    assert result.source_reached.all()
    assert result.fixed_time_accepted.all()
    assert result.inserted_transition_frames == 0
    assert not result.source_collision.any()
    assert result.execution_source_index.tolist() == [0, 1, 2, 3]
