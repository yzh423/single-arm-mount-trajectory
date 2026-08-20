import csv
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from scripts.episode_3d_desktop import (  # noqa: E402
    EpisodeViewerWindow,
    load_episode,
    nearest_index,
    quaternion_to_matrix,
)


def test_matplotlib_uses_a_chinese_capable_font():
    from matplotlib import rcParams

    assert rcParams["font.sans-serif"][0] == "Microsoft YaHei"


def write_episode(path: Path, include_left=True, include_right=True) -> Path:
    fields = ["t", "coordinate_frame"]
    for side, included in (("right", include_right), ("left", include_left)):
        if included:
            fields.extend(
                [
                    f"{side}_tcp_pos_x", f"{side}_tcp_pos_y", f"{side}_tcp_pos_z",
                    f"{side}_tcp_quat_w", f"{side}_tcp_quat_x",
                    f"{side}_tcp_quat_y", f"{side}_tcp_quat_z",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, time_s in enumerate((0.0, 0.5, 1.0)):
            row = {"t": time_s, "coordinate_frame": "vr_world"}
            for side, included in (("right", include_right), ("left", include_left)):
                if included:
                    offset = 1 if side == "right" else -1
                    row.update(
                        {
                            f"{side}_tcp_pos_x": offset * index,
                            f"{side}_tcp_pos_y": index + 1,
                            f"{side}_tcp_pos_z": index + 2,
                            f"{side}_tcp_quat_w": 1,
                            f"{side}_tcp_quat_x": 0,
                            f"{side}_tcp_quat_y": 0,
                            f"{side}_tcp_quat_z": 0,
                        }
                    )
            writer.writerow(row)
    return path


def test_quaternion_identity_and_z_rotation():
    np.testing.assert_allclose(quaternion_to_matrix([1, 0, 0, 0]), np.eye(3))
    half = np.sqrt(0.5)
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(quaternion_to_matrix([half, 0, 0, half]), expected, atol=1e-7)


def test_quaternion_rejects_zero_norm():
    with pytest.raises(ValueError, match="quaternion"):
        quaternion_to_matrix([0, 0, 0, 0])


def test_nearest_index_clamps_and_selects_nearest():
    times = np.array([0.0, 0.5, 1.0])
    assert nearest_index(times, -1) == 0
    assert nearest_index(times, 0.4) == 1
    assert nearest_index(times, 3) == 2


def test_load_episode_parses_both_tcp_series(tmp_path):
    episode = load_episode(write_episode(tmp_path / "both.csv"))
    assert episode.frame == "vr_world"
    assert episode.duration_s == 1.0
    assert set(episode.series) == {"right", "left"}
    assert episode.series["right"].positions.shape == (3, 3)
    assert episode.series["left"].quaternions.shape == (3, 4)


def test_load_episode_allows_one_arm_but_rejects_neither(tmp_path):
    one = load_episode(write_episode(tmp_path / "left.csv", include_right=False))
    assert set(one.series) == {"left"}
    with pytest.raises(ValueError, match="TCP"):
        load_episode(write_episode(tmp_path / "none.csv", include_left=False, include_right=False))


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_window_loads_both_arms_and_advances_frame(qt_app, tmp_path):
    window = EpisodeViewerWindow()
    assert window.windowTitle() == "TCP 工具坐标系动画查看器"
    for name in ("open_button", "play_button", "time_slider", "speed_combo", "loop_check"):
        assert getattr(window, name) is not None

    window.load_path(write_episode(tmp_path / "window.csv"))
    assert window.episode is not None
    assert set(window.arm_artists) == {"right", "left"}
    assert window.play_button.isEnabled()

    window.set_time(0.6)
    assert window.current_time_s == pytest.approx(0.6)
    assert window.current_indices == {"right": 1, "left": 1}
    window.close()
