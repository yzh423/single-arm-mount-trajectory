import json
from pathlib import Path

import numpy as np
import pytest

from factory_bimanual.artifacts import FrameDiagnostics
from factory_bimanual.video import (
    VideoRenderConfig, build_realtime_timing, decode_check_mp4, interpolation_sample,
    execution_status_label, is_replanned_transition,
    render_mujoco_mp4, write_video_provenance,
)


def test_video_title_is_robot_specific():
    config = VideoRenderConfig(title="Dual PiperX trajectory follow")
    assert config.title == "Dual PiperX trajectory follow"


def test_interpolation_sample_moves_through_retimed_interval():
    lower, upper, alpha = interpolation_sample(np.array([0.0, 1.0, 1.1]), .4)
    assert (lower, upper) == (0, 1)
    assert alpha == pytest.approx(.4)


def test_replanned_transition_requires_actual_joint_motion():
    assert not is_replanned_transition(np.zeros(2), np.zeros(2), .04, 60)
    assert is_replanned_transition(np.zeros(2), np.array([.3, 0]), .04, 60)
    assert not is_replanned_transition(np.zeros(2), np.array([.01, 0]), .01, 60)


def test_combined_retimed_collision_status_keeps_both_facts_visible():
    reason, _ = execution_status_label(
        "RETIMED_TRANSITION_COLLISION", ok=True)
    assert reason == "RETIMED TRANSITION - COLLISION AUDIT"


def test_realtime_timing_preserves_irregular_source_duration():
    source_t = np.array([4.0, 4.03, 4.20, 4.21, 4.50])
    timing = build_realtime_timing(source_t, fps=100.0)
    assert timing.source_duration_s == 0.5
    assert abs(timing.encoded_duration_s - 0.5) <= 0.01
    assert timing.frame_source_indices[0] == 0
    assert timing.frame_source_indices[-1] == 4
    # The long pause is retained as repeated frames, rather than deleting rows.
    assert timing.frame_source_indices.tolist().count(1) > 1


def test_realtime_timing_is_causal_floor_zero_order_hold():
    timing = build_realtime_timing(np.array([0.0, 0.06, 0.2]), fps=10.0)
    # At encoded t=.1 the .06 state is current; the future .2 state is forbidden.
    assert timing.encoded_time_s.tolist() == pytest.approx([0.0, 0.1, 0.2])
    assert timing.frame_source_indices.tolist() == [0, 1, 2]


def test_realtime_timing_always_renders_fractional_final_source_pose():
    timing = build_realtime_timing(
        np.array([0.0, 1.0 / 60.0, 2.4 / 60.0]), fps=60.0)
    assert timing.encoded_time_s[-1] == pytest.approx(2.4 / 60.0)
    assert timing.frame_source_indices[-1] == 2


def test_video_sidecar_has_per_frame_source_and_failure_provenance(tmp_path: Path):
    timing = build_realtime_timing(np.array([0.0, 0.1, 0.2]), fps=10)
    diagnostics = [
        FrameDiagnostics(i, i / 10, "ok" if i != 1 else "solver_failure", "ok", False, False, "x.csv")
        for i in range(3)
    ]
    path = write_video_provenance(tmp_path / "video.json", timing, diagnostics)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["playback_speed"] == 1.0
    assert any(row["left_failure_reason"] == "solver_failure" for row in payload["encoded_frames"])
    assert {row["source_row_index"] for row in payload["encoded_frames"]} == {0, 1, 2}


def test_execution_timeline_sidecar_names_execution_knots(tmp_path: Path):
    timing = build_realtime_timing(np.array([0.0, 0.2]), fps=10)
    diagnostics = [
        FrameDiagnostics(
            i, i / 60.0, "FOLLOW", "FOLLOW", False, False, "take.csv",
            execution_index=i, execution_time_s=i * 0.2,
            execution_state="FOLLOW",
        )
        for i in range(2)
    ]

    path = write_video_provenance(
        tmp_path / "execution.json", timing, diagnostics,
        timeline_domain="execution", interpolate_states=True)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["timeline_domain"] == "execution"
    assert payload["timing_method"].startswith("constant-fps linear interpolation")
    assert "execution_knots" in payload and "source_frames" not in payload


def test_interpolated_execution_provenance_audits_the_incoming_edge(tmp_path: Path):
    timing = build_realtime_timing(np.array([0.0, 0.2]), fps=10)
    diagnostics = [
        FrameDiagnostics(
            0, 0.0, "FOLLOW", "FOLLOW", False, False, "take.csv",
            execution_index=0, execution_time_s=0.0,
            execution_state="FOLLOW", state_collision=False,
            incoming_transition_collision=False,
        ),
        FrameDiagnostics(
            1, 1 / 60, "RETIMED_TRANSITION_COLLISION",
            "RETIMED_TRANSITION_COLLISION", True, False, "take.csv",
            execution_index=1, execution_time_s=0.2,
            execution_state="RETIMED_TRANSITION_COLLISION",
            state_collision=False, incoming_transition_collision=True,
        ),
    ]

    path = write_video_provenance(
        tmp_path / "execution.json", timing, diagnostics,
        timeline_domain="execution", interpolate_states=True)
    rows = json.loads(path.read_text(encoding="utf-8"))["encoded_frames"]

    assert rows[0]["audit_scope"] == "knot"
    assert rows[0]["audit_execution_index"] == 0
    assert rows[0]["collision"] is False
    assert rows[1]["audit_scope"] == "incoming_edge"
    assert rows[1]["audit_execution_index"] == 1
    assert rows[1]["collision"] is True
    assert rows[1]["execution_state"] == "RETIMED_TRANSITION_COLLISION"
    # The exact destination knot is audited as a state, not as its incoming edge.
    assert rows[2]["audit_scope"] == "knot"
    assert rows[2]["audit_execution_index"] == 1
    assert rows[2]["collision"] is False
    assert rows[2]["execution_state"] == "RETIMED_TRANSITION"


def test_tiny_headless_mujoco_mp4_smoke(tmp_path: Path):
    xml = tmp_path / "scene.xml"
    xml.write_text('''<mujoco><worldbody>
      <geom name="table" type="box" size=".5 .5 .02" pos="0 0 -.02"/>
      <body name="left"><joint name="left_j"/><geom type="capsule" size=".03 .2" pos="0 0 .2"/></body>
      <body name="right" pos=".3 0 0"><joint name="right_j"/><geom type="capsule" size=".03 .2" pos="0 0 .2"/></body>
      <body name="left_target" mocap="true"><geom type="sphere" size=".025" rgba="1 0 0 1"/></body>
      <body name="right_target" mocap="true"><geom type="sphere" size=".025" rgba="0 0 1 1"/></body>
    </worldbody></mujoco>''', encoding="utf-8")
    times = np.array([0.0, 0.1, 0.2])
    diagnostics = [FrameDiagnostics(i, t, "ok", "ok", False, False, "x.csv")
                   for i, t in enumerate(times)]
    try:
        result = render_mujoco_mp4(
            xml, tmp_path / "smoke.mp4", times,
            qpos=np.array([[0, 0], [.1, -.1], [.2, -.2]]), diagnostics=diagnostics,
            left_targets=np.array([[0, .2, .2], [0, .25, .2], [0, .3, .2]]),
            right_targets=np.array([[.3, .2, .2], [.3, .25, .2], [.3, .3, .2]]),
            config=VideoRenderConfig(width=160, height=120, fps=10),
        )
    except RuntimeError as exc:
        if "renderer preflight failed" in str(exc).lower():
            pytest.skip(str(exc))
        raise
    check = decode_check_mp4(result.video_path, expected_frames=3,
                             expected_resolution=(160, 120), expected_duration_s=.2)
    assert check.frame_count == 3
    assert result.provenance_path.exists()
