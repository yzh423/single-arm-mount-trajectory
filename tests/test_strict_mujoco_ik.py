import unittest

import mujoco
import numpy as np

from scripts.strict_mujoco_ik import (continuity_jump, continuity_recovery_step,
                                      generate_pose_candidate_layers,
                                      joint_discontinuity_from_recovery_modes,
                                      joint_periodic_mask, solve_pose_path,
                                      solve_pose_path_layered,
                                      solve_pose_path_multibranch, solve_position_path,
                                      wrapped_joint_delta)
from scripts.strict_mujoco_model import sampled_maximum_tcp_reach_m
from scripts.solve_strict_urdf_task_cache import collision_flags
from design_optimization.search_policy import dimension_aware_candidate_budget, staged_sobol_candidates


class StrictMujocoIKTest(unittest.TestCase):
    @staticmethod
    def redundant_planar_model():
        return mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody><body>
          <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
          <geom type="sphere" size=".01" mass=".1"/>
          <body pos=".25 0 0"><joint name="j2" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
            <geom type="sphere" size=".01" mass=".1"/>
            <body pos=".25 0 0"><joint name="j3" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
              <geom type="sphere" size=".01" mass=".1"/>
              <body pos=".20 0 0"><joint name="j4" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <geom type="sphere" size=".01" mass=".1"/><site name="tcp" pos=".15 0 0"/>
              </body>
            </body>
          </body>
        </body></worldbody></mujoco>""")

    @staticmethod
    def poses_from_q(model, q_rows):
        data = mujoco.MjData(model)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        positions, quaternions = [], []
        for q in q_rows:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, data.site_xmat[site])
            positions.append(data.site_xpos[site].copy())
            quaternions.append(quaternion)
        return np.asarray(positions), np.asarray(quaternions)

    def test_layer_generator_returns_multiple_deduplicated_real_ik_branches(self) -> None:
        model = self.redundant_planar_model()
        positions, quaternions = self.poses_from_q(
            model, ((.2, .6, -.4, .1),))

        layers = generate_pose_candidate_layers(
            model, "tcp", ("j1", "j2", "j3", "j4"), positions, quaternions,
            candidates_per_frame=6, global_seed_count=12, iterations=100,
            position_tolerance_m=1e-4, orientation_tolerance_rad=1e-4,
            candidate_collision_free=lambda q: bool(abs(q[0]) < .8),
            rng_seed=17)

        self.assertEqual(len(layers), 1)
        self.assertGreaterEqual(len(layers[0]), 2)
        for left, right in zip(layers[0], layers[0][1:]):
            self.assertGreater(float(np.max(np.abs(left.q - right.q))), np.deg2rad(1.0))
        self.assertTrue(all(
            item.collision_free == bool(abs(item.q[0]) < .8)
            for item in layers[0]))

    def test_layered_solver_tracks_path_and_exposes_candidate_counts(self) -> None:
        model = self.redundant_planar_model()
        positions, quaternions = self.poses_from_q(
            model, ((.2, .6, -.4, .1), (.22, .58, -.38, .1), (.24, .56, -.36, .1)))

        result = solve_pose_path_layered(
            model, "tcp", ("j1", "j2", "j3", "j4"), positions, quaternions,
            time_s=np.asarray((0.0, .1, .2)), candidates_per_frame=6,
            global_seed_count=12, horizon=2, beam_width=6, iterations=100,
            velocity_limit_rad_s=4.0, maximum_frame_jump_rad=.5,
            position_tolerance_m=1e-4, orientation_tolerance_rad=1e-4,
            candidate_collision_free=lambda q: True,
            transition_collision_free=lambda previous, current: True,
            rng_seed=19)

        self.assertEqual(result.q.shape, (3, 4))
        self.assertTrue(result.success.all())
        self.assertTrue(np.all(result.branch_count >= 2))
        self.assertTrue(np.all(result.chosen_branch_index >= 0))

    def test_layered_solver_holds_when_every_swept_edge_is_rejected(self) -> None:
        model = self.redundant_planar_model()
        positions, quaternions = self.poses_from_q(
            model, ((.2, .6, -.4, .1), (.3, .5, -.3, .1)))

        result = solve_pose_path_layered(
            model, "tcp", ("j1", "j2", "j3", "j4"), positions, quaternions,
            time_s=np.asarray((0.0, .1)), candidates_per_frame=5,
            global_seed_count=10, horizon=2, beam_width=5, iterations=100,
            velocity_limit_rad_s=8.0, maximum_frame_jump_rad=1.0,
            position_tolerance_m=1e-4, orientation_tolerance_rad=1e-4,
            candidate_collision_free=lambda q: True,
            transition_collision_free=lambda previous, current: bool(np.allclose(previous, current)),
            rng_seed=23)

        np.testing.assert_allclose(result.q[1], result.q[0])
        self.assertFalse(result.success[1])
        self.assertIn("hold", result.recovery_mode[1])
        self.assertFalse(result.joint_discontinuity[1])

    def test_layered_joint_discontinuity_does_not_absorb_pose_or_collision_failures(self) -> None:
        modes = np.asarray((
            "none", "limited_step", "hold_wrist_branch_switch",
            "hold_no_feasible_edge", "hold_recovery_collision",
            "hold_no_candidate"))

        actual = joint_discontinuity_from_recovery_modes(modes)

        np.testing.assert_array_equal(actual, (False, True, True, False, False, False))

    def test_periodic_joint_delta_uses_shortest_equivalent_rotation(self) -> None:
        previous = np.asarray((np.pi - 0.01, 0.2))
        candidate = np.asarray((-np.pi + 0.01, 0.4))
        delta = wrapped_joint_delta(candidate, previous, np.asarray((True, False)))
        np.testing.assert_allclose(delta, (0.02, 0.2), atol=1e-12)

    def test_continuity_is_not_enforced_before_first_valid_pose(self) -> None:
        self.assertFalse(continuity_jump(False, np.deg2rad(120.0), np.deg2rad(25.0)))
        self.assertTrue(continuity_jump(True, np.deg2rad(120.0), np.deg2rad(25.0)))

    def test_continuity_recovery_advances_without_exceeding_frame_limit(self) -> None:
        previous = np.deg2rad(np.asarray((0.0, 170.0)))
        candidate = np.deg2rad(np.asarray((80.0, -170.0)))

        recovered = continuity_recovery_step(
            candidate, previous, np.asarray((False, True)), np.deg2rad(25.0))

        delta = wrapped_joint_delta(recovered, previous, np.asarray((False, True)))
        self.assertLessEqual(float(np.max(np.abs(delta))), np.deg2rad(25.0) + 1e-12)
        np.testing.assert_allclose(np.degrees(delta), (25.0, 20.0), atol=1e-10)

    def test_continuity_recovery_respects_per_joint_step_limits(self) -> None:
        recovered = continuity_recovery_step(
            np.asarray((0.20, 0.20)), np.zeros(2), np.zeros(2, dtype=bool),
            np.asarray((0.20, 0.005)),
        )
        np.testing.assert_allclose(recovered, (0.20, 0.005), atol=1e-12)

    def test_only_unlimited_hinge_joints_are_periodic(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody><body>
          <joint name="limited_wide" type="hinge" range="-6.283185 6.283185"/>
          <joint name="limited_slide" type="slide" range="-10 10"/>
          <joint name="continuous" type="hinge" limited="false"/>
          <geom type="sphere" size=".01" mass=".1"/>
        </body></worldbody></mujoco>""")
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
               for name in ("limited_wide", "limited_slide", "continuous")]
        np.testing.assert_array_equal(joint_periodic_mask(model, ids), (False, False, True))

    def test_limited_wide_hinge_does_not_wrap_across_hard_stops(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody><body>
          <joint name="j1" type="hinge" axis="0 0 1" range="-3.141592653589793 3.141592653589793"/>
          <geom type="sphere" size=".01" mass=".1"/><site name="tcp" pos=".2 0 0"/>
        </body></worldbody></mujoco>""")
        data = mujoco.MjData(model)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        targets, quaternions = [], []
        for angle in (np.pi - .01, -np.pi + .01):
            data.qpos[0] = angle
            mujoco.mj_forward(model, data)
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, data.site_xmat[site])
            targets.append(data.site_xpos[site].copy())
            quaternions.append(quaternion)
        result = solve_pose_path(
            model, "tcp", ("j1",), np.asarray(targets), np.asarray(quaternions),
            initial_q=np.asarray((np.pi - .01,)), time_s=np.asarray((0.0, 0.1)),
            velocity_limit_rad_s=.25, maximum_frame_jump_rad=.025,
            position_tolerance_m=1e-6, orientation_tolerance_rad=1e-6,
            iterations=80, restarts=2,
        )
        self.assertTrue(result.joint_discontinuity[1])
        self.assertTrue(result.joint_limits_respected)
        self.assertGreaterEqual(result.q.min(), -np.pi - 1e-12)
        self.assertLessEqual(result.q.max(), np.pi + 1e-12)

    def test_dimension_aware_search_is_not_a_sparse_96_point_screen(self) -> None:
        budget = dimension_aware_candidate_budget(10)
        self.assertGreaterEqual(budget, 1024)
        lower = np.full(10, -1.0); upper = np.full(10, 1.0); incumbent = np.zeros(10)
        candidates = staged_sobol_candidates(lower, upper, incumbent, budget=budget, seed=750)
        self.assertEqual(candidates.shape, (budget, 10))
        self.assertTrue(np.allclose(candidates[0], incumbent))
        self.assertTrue(np.all(candidates >= lower) and np.all(candidates <= upper))

    def test_collision_audit_ignores_internal_inactive_gripper_sibling_overlap(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody><body name="base"><geom type="sphere" size="0.03"/>
          <body name="arm" pos="0 0 0.1"><joint name="arm_joint"/><geom type="sphere" size="0.03"/>
            <body name="finger1" pos="0 0 0.1"><joint name="finger_joint1"/><geom type="sphere" size="0.04"/></body>
            <body name="finger2" pos="0 0 0.1"><joint name="finger_joint2"/><geom type="sphere" size="0.04"/></body>
          </body></body></worldbody></mujoco>
        """)
        table, self_collision = collision_flags(model, ("arm_joint",), np.zeros((1, 1)))
        self.assertFalse(bool(table[0]))
        self.assertFalse(bool(self_collision[0]))

    def test_collision_audit_rejects_distal_link_penetrating_mount_pedestal(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <geom name="strict_pedestal" type="cylinder" size=".08 .3" pos="0 0 .3"/>
          <body name="base"><joint name="j1"/><geom type="sphere" size=".01"/>
            <body name="middle" pos="0 0 .7"><joint name="j2"/><geom type="sphere" size=".01"/>
              <body name="distal" pos="0 0 -.4"><joint name="j3"/><geom type="sphere" size=".12"/>
              </body>
            </body>
          </body>
        </worldbody></mujoco>""")

        mount_collision, _ = collision_flags(model, ("j1", "j2", "j3"), np.zeros((1, 3)))

        self.assertTrue(bool(mount_collision[0]))

    def test_collision_audit_rejects_actuated_link_penetrating_mount_adapter(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <geom name="strict_mount_adapter" type="cylinder" size=".08 .04" pos="0 0 .3"/>
          <body name="root"><joint name="j1"/><geom name="root_geom" type="sphere" size=".02"/>
            <body name="moving" pos="0 0 .3"><joint name="j2"/><geom name="moving_geom" type="sphere" size=".1"/></body>
          </body>
        </worldbody></mujoco>""")
        mount_collision, _ = collision_flags(model, ("j1", "j2"), np.zeros((1, 2)))
        self.assertTrue(bool(mount_collision[0]))

    def test_collision_audit_allows_only_explicit_nonadjacent_pair(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody><body name="one"><joint name="j1"/><geom name="one_geom" type="sphere" size=".1"/>
          <body name="two" pos="0 0 .3"><joint name="j2"/><geom name="two_geom" type="sphere" size=".01"/>
            <body name="three" pos="0 0 -.3"><joint name="j3"/><geom name="three_geom" type="sphere" size=".1"/></body>
          </body>
        </body></worldbody></mujoco>""")
        _, collision = collision_flags(model, ("j1", "j2", "j3"), np.zeros((1, 3)))
        self.assertTrue(bool(collision[0]))
        _, allowed = collision_flags(
            model, ("j1", "j2", "j3"), np.zeros((1, 3)),
            allowed_collision_pairs={("one", "three")})
        self.assertFalse(bool(allowed[0]))

    def test_native_geometry_measurement_does_not_modify_chain_or_mesh(self) -> None:
        spec = mujoco.MjSpec.from_string("""
        <mujoco><compiler angle="radian"/><asset><mesh name="part" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset><worldbody>
          <body name="base"><geom type="mesh" mesh="part"/><body name="link1" pos="0.4 0 0">
            <joint name="j1" type="hinge"/><geom type="mesh" mesh="part"/><body name="link2" pos="0.4 0 0">
              <joint name="j2" type="hinge"/><geom type="mesh" mesh="part"/><site name="flange" pos="0 0 0"/>
            </body></body></body>
        </worldbody></mujoco>
        """)
        mesh_scale_before = np.asarray(spec.meshes[0].scale).copy()
        spec.body("link2").add_site(name="tcp", pos=[0.1, 0.0, 0.0])
        model = spec.compile()
        self.assertAlmostEqual(sampled_maximum_tcp_reach_m(model, ("j1", "j2"), "flange"), 0.8, places=6)
        flange = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "flange")
        tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        self.assertAlmostEqual(float(np.linalg.norm(data.site_xpos[tcp] - data.site_xpos[flange])), 0.1, places=9)
        self.assertTrue(np.allclose(spec.meshes[0].scale, mesh_scale_before))

    def test_xarm6_tracks_reachable_continuous_path(self) -> None:
        xml = """
        <mujoco><compiler angle="radian"/><worldbody>
          <body name="base">
            <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
            <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
            <body name="link2" pos="0.3 0 0">
              <joint name="j2" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
              <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
              <site name="tcp" pos="0.3 0 0" size="0.01"/>
            </body>
          </body>
        </worldbody></mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        targets = np.array([[0.50, 0.10, 0.0], [0.46, 0.18, 0.0], [0.40, 0.25, 0.0]])
        result = solve_position_path(model, "tcp", ("j1", "j2"), targets)
        self.assertEqual(result.q.shape, (3, 2))
        self.assertLess(float(result.position_error_m.max()), 1e-4)
        self.assertTrue(result.joint_limits_respected)
        self.assertTrue(np.isfinite(result.reached_xyz_m).all())

    def test_pose_path_tracks_tcp_position_and_orientation(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody><body><geom type="sphere" size="0.01" mass="0.1"/>
          <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
          <body pos="0.3 0 0"><geom type="sphere" size="0.01" mass="0.1"/><joint name="j2" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
            <body pos="0.3 0 0"><geom type="sphere" size="0.01" mass="0.1"/><joint name="j3" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
              <site name="tcp" pos="0.2 0 0"/>
            </body>
          </body>
        </body></worldbody></mujoco>
        """)
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j{i}") for i in range(1, 4)]
        addresses = [model.jnt_qposadr[j] for j in joint_ids]
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        data = mujoco.MjData(model)
        poses = []
        for q in ([0.2, 0.4, -0.1], [0.3, 0.35, -0.05], [0.4, 0.3, 0.0]):
            data.qpos[addresses] = q; mujoco.mj_forward(model, data)
            quat = np.zeros(4); mujoco.mju_mat2Quat(quat, data.site_xmat[site])
            poses.append((data.site_xpos[site].copy(), quat))
        result = solve_pose_path(model, "tcp", ("j1", "j2", "j3"),
                                 np.asarray([p for p, _ in poses]), np.asarray([q for _, q in poses]),
                                 position_tolerance_m=1e-4, orientation_tolerance_rad=1e-4)
        self.assertLess(float(result.position_error_m.max()), 1e-4)
        self.assertLess(float(result.orientation_error_rad.max()), 1e-4)
        self.assertTrue(result.success.all())
        self.assertEqual(result.branch_count.shape, result.success.shape)
        self.assertEqual(result.recovery_mode.shape, result.success.shape)
        self.assertTrue(np.all(result.branch_count >= 1))

    def test_pose_path_uses_timestamp_velocity_limit(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody><body>
          <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
          <geom type="sphere" size=".01" mass=".1"/><site name="tcp" pos=".2 0 0"/>
        </body></worldbody></mujoco>""")
        data = mujoco.MjData(model); site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        poses = []
        for angle in (0.0, 0.2):
            data.qpos[0] = angle; mujoco.mj_forward(model, data)
            quat = np.zeros(4); mujoco.mju_mat2Quat(quat, data.site_xmat[site])
            poses.append((data.site_xpos[site].copy(), quat))
        result = solve_pose_path(
            model, "tcp", ("j1",), np.asarray([x for x, _ in poses]),
            np.asarray([q for _, q in poses]), time_s=np.asarray((0.0, 0.01)),
            velocity_limit_rad_s=1.0, maximum_frame_jump_rad=0.5,
            position_tolerance_m=1e-5, orientation_tolerance_rad=1e-5,
        )
        self.assertTrue(result.joint_discontinuity[1])
        self.assertFalse(result.velocity_violation[1])
        self.assertLessEqual(abs(result.q[1, 0] - result.q[0, 0]), 0.01 + 1e-9)

    def test_multibranch_velocity_diagnostic_uses_realized_bounded_motion(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody><body>
          <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
          <geom type="sphere" size=".01" mass=".1"/><site name="tcp" pos=".2 0 0"/>
        </body></worldbody></mujoco>""")
        data = mujoco.MjData(model)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        targets, quaternions = [], []
        for angle in (0.0, 0.2):
            data.qpos[0] = angle
            mujoco.mj_forward(model, data)
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, data.site_xmat[site])
            targets.append(data.site_xpos[site].copy())
            quaternions.append(quaternion)

        result = solve_pose_path_multibranch(
            model, "tcp", ("j1",), np.asarray(targets), np.asarray(quaternions),
            time_s=np.asarray((0.0, 0.01)), branch_candidates=2, beam_width=2,
            velocity_limit_rad_s=1.0, maximum_frame_jump_rad=0.5,
            position_tolerance_m=1e-5, orientation_tolerance_rad=1e-5,
        )

        self.assertTrue(result.joint_discontinuity[1])
        self.assertFalse(result.velocity_violation[1])
        self.assertLessEqual(abs(result.q[1, 0] - result.q[0, 0]), 0.01 + 1e-9)

    def test_multibranch_pose_path_returns_rolling_diagnostics(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody><body><geom type="sphere" size=".01" mass=".1"/>
          <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
          <site name="tcp" pos=".2 0 0"/>
        </body></worldbody></mujoco>""")
        data = mujoco.MjData(model); site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        targets, quaternions = [], []
        for angle in (0.0, 0.05, 0.1):
            data.qpos[0] = angle; mujoco.mj_forward(model, data)
            quaternion = np.zeros(4); mujoco.mju_mat2Quat(quaternion, data.site_xmat[site])
            targets.append(data.site_xpos[site].copy()); quaternions.append(quaternion)
        result = solve_pose_path_multibranch(
            model, "tcp", ("j1",), np.asarray(targets), np.asarray(quaternions),
            time_s=np.asarray((0.0, 0.1, 0.2)), branch_candidates=3, horizon=3,
            iterations=80, position_tolerance_m=1e-5, orientation_tolerance_rad=1e-5,
        )
        self.assertEqual(result.q.shape, (3, 1))
        self.assertTrue(result.success.all())
        self.assertTrue(np.all(result.branch_count >= 1))

    def test_multibranch_recovery_holds_when_collision_checker_rejects_edge(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody><body><geom type="sphere" size=".01" mass=".1"/>
          <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
          <site name="tcp" pos=".2 0 0"/>
        </body></worldbody></mujoco>""")
        data=mujoco.MjData(model);site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"tcp")
        targets=[];quaternions=[]
        for angle in (0.0, 0.2):
            data.qpos[0]=angle;mujoco.mj_forward(model,data);quat=np.zeros(4);mujoco.mju_mat2Quat(quat,data.site_xmat[site])
            targets.append(data.site_xpos[site].copy());quaternions.append(quat)
        result=solve_pose_path_multibranch(
            model,"tcp",("j1",),np.asarray(targets),np.asarray(quaternions),
            time_s=np.asarray((0.0,0.1)),branch_candidates=2,horizon=2,iterations=60,
            transition_collision_free=lambda previous,current: bool(np.allclose(previous,current)),
        )
        np.testing.assert_allclose(result.q[1], result.q[0])
        self.assertEqual(result.recovery_mode[1], "hold")

    def test_multibranch_never_holds_an_initial_seed_rejected_by_collision(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody><body>
          <geom type="sphere" size=".01" mass=".1"/>
          <joint name="j1" type="hinge" axis="0 0 1" range="-1 1"/>
          <site name="tcp" pos=".2 0 0"/>
        </body></worldbody></mujoco>""")
        target = np.asarray(((5.0, 5.0, 5.0),))
        quaternion = np.asarray(((1.0, 0.0, 0.0, 0.0),))

        result = solve_pose_path_multibranch(
            model, "tcp", ("j1",), target, quaternion,
            time_s=np.asarray((0.0,)), branch_candidates=4, beam_width=4,
            initial_q=np.asarray((-0.5,)), iterations=20,
            candidate_collision_free=lambda q: bool(q[0] >= 0.0),
        )

        assert np.all(result.q[:, 0] >= 0.0)

    def test_pose_path_holds_last_valid_posture_when_target_is_unreachable(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody><body><geom type="sphere" size=".01" mass=".1"/>
          <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
          <body pos="0.3 0 0"><geom type="sphere" size=".01" mass=".1"/><joint name="j2" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
            <body pos="0.3 0 0"><geom type="sphere" size=".01" mass=".1"/><joint name="j3" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
              <site name="tcp" pos="0.2 0 0"/>
            </body>
          </body>
        </body></worldbody></mujoco>""")
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        quaternion = np.zeros(4); mujoco.mju_mat2Quat(quaternion, data.site_xmat[site])
        targets = np.asarray([data.site_xpos[site].copy(), [5.0, 5.0, 5.0]])

        result = solve_pose_path(model, "tcp", ("j1", "j2", "j3"), targets,
                                 np.tile(quaternion, (2, 1)), restarts=4, iterations=30)

        self.assertTrue(result.success[0])
        self.assertFalse(result.success[1])
        np.testing.assert_allclose(result.q[1], result.q[0])

    def test_candidate_mode_exposes_best_invalid_posture_for_frontier_recovery(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody><body><geom type="sphere" size=".01" mass=".1"/>
          <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
          <body pos=".3 0 0"><geom type="sphere" size=".01" mass=".1"/>
            <joint name="j2" type="hinge" axis="0 0 1" range="-3.14 3.14"/><site name="tcp" pos=".3 0 0"/>
          </body></body></worldbody></mujoco>""")
        data=mujoco.MjData(model);mujoco.mj_forward(model,data);site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"tcp")
        quat=np.zeros(4);mujoco.mju_mat2Quat(quat,data.site_xmat[site])
        result=solve_pose_path(model,"tcp",("j1","j2"),np.asarray(((1.0,1.0,0.0),)),
                               np.asarray((quat,)),iterations=40,restarts=4,hold_invalid=False)
        self.assertFalse(result.success[0])
        self.assertFalse(np.allclose(result.q[0], np.zeros(2)))

    def test_multibranch_does_not_walk_toward_an_invalid_pose_solution(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody><body><geom type="sphere" size=".01" mass=".1"/>
          <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
          <body pos=".3 0 0"><geom type="sphere" size=".01" mass=".1"/>
            <joint name="j2" type="hinge" axis="0 0 1" range="-3.14 3.14"/><site name="tcp" pos=".3 0 0"/>
          </body></body></worldbody></mujoco>""")
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, data.site_xmat[site])
        target = np.asarray((5.0, 5.0, 0.0))
        result = solve_pose_path_multibranch(
            model, "tcp", ("j1", "j2"), np.asarray((target, target)),
            np.asarray((quat, quat)), time_s=np.asarray((0.0, 0.1)),
            branch_candidates=3, beam_width=3, iterations=40)
        np.testing.assert_allclose(result.q[0], result.q[1])
        self.assertTrue(np.all(result.recovery_mode == "hold"))


if __name__ == "__main__":
    unittest.main()
