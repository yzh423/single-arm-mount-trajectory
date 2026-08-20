"""Native desktop viewer for animated left/right TCP tool coordinate frames."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class TcpSeries:
    side: str
    times: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray


@dataclass(frozen=True)
class Episode:
    path: Path
    frame: str
    series: dict[str, TcpSeries]
    start_t: float
    duration_s: float


def quaternion_to_matrix(quaternion) -> np.ndarray:
    """Convert a w,x,y,z quaternion to a 3x3 rotation matrix."""
    values = np.asarray(quaternion, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("quaternion must contain four finite values")
    norm = np.linalg.norm(values)
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = values / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def nearest_index(times: np.ndarray, target: float) -> int:
    if len(times) == 0:
        raise ValueError("times must not be empty")
    insertion = int(np.searchsorted(times, target, side="left"))
    if insertion <= 0:
        return 0
    if insertion >= len(times):
        return len(times) - 1
    return insertion - 1 if target - times[insertion - 1] <= times[insertion] - target else insertion


def _finite_values(row: dict[str, str], columns: list[str]) -> list[float] | None:
    try:
        values = [float(row[column]) for column in columns]
    except (KeyError, TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def load_episode(path: Path) -> Episode:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    if "t" not in fields:
        raise ValueError("CSV 缺少 t 时间列")

    valid_times = []
    for row in rows:
        value = _finite_values(row, ["t"])
        if value is not None:
            valid_times.append(value[0])
    if not valid_times:
        raise ValueError("CSV 没有有效时间数据")
    start_t = min(valid_times)
    series: dict[str, TcpSeries] = {}
    for side in ("right", "left"):
        position_columns = [f"{side}_tcp_pos_{axis}" for axis in "xyz"]
        quaternion_columns = [f"{side}_tcp_quat_{axis}" for axis in "wxyz"]
        if not set(position_columns + quaternion_columns).issubset(fields):
            continue
        times, positions, quaternions = [], [], []
        for row in rows:
            time_value = _finite_values(row, ["t"])
            position = _finite_values(row, position_columns)
            quaternion = _finite_values(row, quaternion_columns)
            if time_value is None or position is None or quaternion is None:
                continue
            if np.linalg.norm(quaternion) < 1e-12:
                continue
            times.append(time_value[0] - start_t)
            positions.append(position)
            quaternions.append(quaternion)
        if times:
            order = np.argsort(times)
            series[side] = TcpSeries(
                side=side,
                times=np.asarray(times, dtype=float)[order],
                positions=np.asarray(positions, dtype=float)[order],
                quaternions=np.asarray(quaternions, dtype=float)[order],
            )
    if not series:
        raise ValueError("CSV 中没有可用的左/右 TCP 位置与四元数数据")
    duration_s = max(float(item.times[-1]) for item in series.values())
    frame = next((row.get("coordinate_frame", "") for row in rows if row.get("coordinate_frame")), "未知")
    return Episode(path.resolve(), frame, series, start_t, duration_s)


class EpisodeViewerWindow(QMainWindow):
    ARM_COLORS = {"right": "#d97706", "left": "#1677a6"}
    AXIS_COLORS = ("#d33f31", "#2e8b57", "#2d65c4")

    def __init__(self, initial_path: Path | None = None):
        super().__init__()
        self.setWindowTitle("TCP 工具坐标系动画查看器")
        self.resize(1280, 820)
        self.episode: Episode | None = None
        self.current_time_s = 0.0
        self.current_indices: dict[str, int] = {}
        self.arm_artists: dict[str, dict] = {}
        self.axis_length = 0.05
        self._slider_updating = False

        self.figure = Figure(figsize=(10, 7), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self._build_controls()
        self._build_layout()
        self._configure_empty_axes()

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        if initial_path is not None:
            self.load_path(initial_path)

    def _build_controls(self) -> None:
        self.open_button = QPushButton("打开 CSV")
        self.open_button.clicked.connect(self.open_dialog)
        self.play_button = QPushButton("播放")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_playback)
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 10000)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self._slider_changed)
        self.time_label = QLabel("0.00 / 0.00 s")
        self.speed_combo = QComboBox()
        for label, speed in (("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0), ("4×", 4.0)):
            self.speed_combo.addItem(label, speed)
        self.speed_combo.setCurrentIndex(2)
        self.loop_check = QCheckBox("循环")
        self.loop_check.setChecked(True)
        self.right_check = QCheckBox("右 TCP")
        self.left_check = QCheckBox("左 TCP")
        self.right_check.setChecked(True)
        self.left_check.setChecked(True)
        self.right_check.toggled.connect(self._apply_visibility)
        self.left_check.toggled.connect(self._apply_visibility)
        self.status_label = QLabel("打开一个 episode CSV 开始播放")

    def _build_layout(self) -> None:
        controls = QHBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.time_slider, 1)
        controls.addWidget(self.time_label)
        controls.addWidget(QLabel("速度"))
        controls.addWidget(self.speed_combo)
        controls.addWidget(self.loop_check)
        controls.addWidget(self.right_check)
        controls.addWidget(self.left_check)
        root = QVBoxLayout()
        root.addLayout(controls)
        root.addWidget(self.status_label)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas, 1)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

    def _configure_empty_axes(self) -> None:
        self.axes.set_xlabel("X (m)")
        self.axes.set_ylabel("Y (m)")
        self.axes.set_zlabel("Z (m)")
        self.axes.set_title("等待 CSV 数据")
        self.axes.grid(True, alpha=0.25)

    def open_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择 Episode CSV", "", "CSV 文件 (*.csv)")
        if filename:
            try:
                self.load_path(Path(filename))
            except (OSError, ValueError) as exc:
                QMessageBox.critical(self, "无法打开 CSV", str(exc))

    def load_path(self, path: Path) -> None:
        episode = load_episode(path)
        self.pause()
        self.episode = episode
        self.current_time_s = 0.0
        self._create_artists()
        self.time_slider.setEnabled(True)
        self.play_button.setEnabled(episode.duration_s > 0)
        self.status_label.setText(
            f"{episode.path.name}  |  {episode.frame}  |  "
            f"{', '.join(side.upper() for side in episode.series)} TCP"
        )
        self.set_time(0.0)

    def _create_artists(self) -> None:
        self.axes.clear()
        self._configure_empty_axes()
        self.axes.set_title(self.episode.path.name if self.episode else "")
        self.arm_artists.clear()
        all_positions = np.vstack([item.positions for item in self.episode.series.values()])
        extent = np.ptp(all_positions, axis=0)
        scale = max(float(np.max(extent)), 0.05)
        self.axis_length = scale * 0.10
        self._set_equal_limits(all_positions)
        labels = {"right": "右 TCP", "left": "左 TCP"}
        for side, series in self.episode.series.items():
            color = self.ARM_COLORS[side]
            full, = self.axes.plot(*series.positions.T, color=color, alpha=0.22, linewidth=1.2)
            traversed, = self.axes.plot([], [], [], color=color, linewidth=2.4, label=labels[side])
            origin, = self.axes.plot([], [], [], marker="o", color=color, markersize=7)
            triad = [self.axes.plot([], [], [], color=axis_color, linewidth=2.2)[0] for axis_color in self.AXIS_COLORS]
            self.arm_artists[side] = {"full": full, "traversed": traversed, "origin": origin, "triad": triad}
        self.axes.legend(loc="upper right")
        self._apply_visibility()

    def _set_equal_limits(self, positions: np.ndarray) -> None:
        minimum = np.min(positions, axis=0)
        maximum = np.max(positions, axis=0)
        center = (minimum + maximum) / 2
        half = max(float(np.max(maximum - minimum)) / 2, 0.025) * 1.12
        self.axes.set_xlim(center[0] - half, center[0] + half)
        self.axes.set_ylim(center[1] - half, center[1] + half)
        self.axes.set_zlim(center[2] - half, center[2] + half)
        self.axes.set_box_aspect((1, 1, 1))

    def set_time(self, time_s: float) -> None:
        if self.episode is None:
            return
        self.current_time_s = float(np.clip(time_s, 0.0, self.episode.duration_s))
        self.current_indices = {}
        for side, series in self.episode.series.items():
            index = nearest_index(series.times, self.current_time_s)
            self.current_indices[side] = index
            artists = self.arm_artists[side]
            path = series.positions[: index + 1]
            artists["traversed"].set_data_3d(*path.T)
            origin = series.positions[index]
            artists["origin"].set_data_3d([origin[0]], [origin[1]], [origin[2]])
            rotation = quaternion_to_matrix(series.quaternions[index])
            for axis_index, line in enumerate(artists["triad"]):
                endpoint = origin + rotation[:, axis_index] * self.axis_length
                line.set_data_3d([origin[0], endpoint[0]], [origin[1], endpoint[1]], [origin[2], endpoint[2]])
        self._slider_updating = True
        slider_value = 0 if self.episode.duration_s == 0 else round(self.current_time_s / self.episode.duration_s * 10000)
        self.time_slider.setValue(slider_value)
        self._slider_updating = False
        self.time_label.setText(f"{self.current_time_s:.2f} / {self.episode.duration_s:.2f} s")
        self.canvas.draw_idle()

    def _slider_changed(self, value: int) -> None:
        if not self._slider_updating and self.episode is not None:
            self.set_time(self.episode.duration_s * value / 10000)

    def toggle_playback(self) -> None:
        if self.timer.isActive():
            self.pause()
        elif self.episode is not None:
            if self.current_time_s >= self.episode.duration_s:
                self.set_time(0.0)
            self.timer.start()
            self.play_button.setText("暂停")

    def pause(self) -> None:
        self.timer.stop() if hasattr(self, "timer") else None
        if hasattr(self, "play_button"):
            self.play_button.setText("播放")

    def _tick(self) -> None:
        if self.episode is None:
            self.pause()
            return
        next_time = self.current_time_s + self.timer.interval() / 1000 * float(self.speed_combo.currentData())
        if next_time >= self.episode.duration_s:
            if self.loop_check.isChecked():
                next_time %= max(self.episode.duration_s, 1e-12)
            else:
                next_time = self.episode.duration_s
                self.pause()
        self.set_time(next_time)

    def _apply_visibility(self) -> None:
        checks = {"right": self.right_check, "left": self.left_check}
        for side, artists in self.arm_artists.items():
            visible = checks[side].isChecked()
            for key, artist in artists.items():
                if key == "triad":
                    for line in artist:
                        line.set_visible(visible)
                else:
                    artist.set_visible(visible)
        self.canvas.draw_idle()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, help="Optional episode CSV to open immediately")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        window = EpisodeViewerWindow(args.csv)
    except (OSError, ValueError) as exc:
        QMessageBox.critical(None, "无法打开 CSV", str(exc))
        return 2
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
