from types import SimpleNamespace

import mujoco
import numpy as np

from factory_bimanual.piperx_recommended import load_recommended_config
from factory_bimanual.recommended_follow import RecommendedFollowRunner
from factory_bimanual.robot_contracts import ROBOT_CONTRACTS
from factory_bimanual.scene_builder import build_same_model_scene


def _static_piperx_task(tmp_path, frames=3):
    contract = ROBOT_CONTRACTS["piperx"]
    xml = tmp_path / "dual_piperx.xml"
    build_same_model_scene(contract, .8, xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    values = {"time_s": np.arange(frames) / 60.0}
    mapped = {}
    for side in ("left", "right"):
        site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[site])
        values[f"{side}_position_m"] = np.repeat(
            data.site_xpos[site][None, :], frames, axis=0)
        values[f"{side}_quaternion_wxyz"] = np.repeat(
            quaternion[None, :], frames, axis=0)
        mapped[side] = values[f"{side}_quaternion_wxyz"].copy()
    return model, SimpleNamespace(**values), mapped


def test_static_real_mujoco_path_is_strict_and_collision_free(tmp_path):
    model, task, mapped = _static_piperx_task(tmp_path)
    result = RecommendedFollowRunner(
        model, task, mapped, load_recommended_config()).run()

    assert result.source_state.tolist() == ["FOLLOW", "FOLLOW", "FOLLOW"]
    assert result.source_accepted.all()
    assert not result.collision.any()
    for side in ("left", "right"):
        assert np.all(result.position_error_m[side] <= .001 + 1e-12)
        assert np.all(
            result.orientation_error_rad[side] <= np.deg2rad(.5) + 1e-12)


def test_runner_exposes_warm_start_and_rescue_candidate_counts(tmp_path):
    model, task, mapped = _static_piperx_task(tmp_path)
    result = RecommendedFollowRunner(
        model, task, mapped, load_recommended_config()).run()
    assert result.anchor_restart_count == 40
    assert result.warm_start_attempted.tolist() == [False, True, True]
    assert np.all(result.candidate_pair_count >= 1)


def test_execution_qpos_and_source_mapping_stay_aligned(tmp_path):
    model, task, mapped = _static_piperx_task(tmp_path)
    result = RecommendedFollowRunner(
        model, task, mapped, load_recommended_config()).run()
    assert result.execution_qpos.shape[1] == model.nq
    assert len(result.execution_qpos) == len(result.source_index)
    assert set(result.source_index.tolist()) == {0, 1, 2}


def test_runner_propagates_pdf_dls_step_limit(tmp_path):
    model, task, mapped = _static_piperx_task(tmp_path)

    runner = RecommendedFollowRunner(
        model, task, mapped, load_recommended_config())

    assert runner.generator.config.maximum_step_rad == 0.18
