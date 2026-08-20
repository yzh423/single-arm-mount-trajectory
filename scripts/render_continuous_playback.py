from __future__ import annotations

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=ROOT / "reports/learning/hourly/0800_continuous_playback.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/learning/hourly/0800_continuous_playback.gif")
    parser.add_argument("--title", default="Held-out EgoDex: continuous branch playback")
    args = parser.parse_args(); data = np.load(args.input)
    robots = ("doosan", "xarm6", "ur5", "kinova")
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    artists = []
    for axis, name in zip(axes.flat, robots):
        metrics = data[f"{name}_metrics"]
        axis.set_title(f"{name}: {metrics[0]:.1f} mm, {metrics[1]:.1f} deg, {100*metrics[2]:.1f}%")
        axis.set_xlim(-.85, .85); axis.set_ylim(-.2, 1.05); axis.set_aspect("equal"); axis.grid(alpha=.2)
        axis.set_xlabel("world x (m)"); axis.set_ylabel("world z (m)")
        left_line, = axis.plot([], [], "o-", lw=3, ms=3, color="#1676d2")
        right_line, = axis.plot([], [], "o-", lw=3, ms=3, color="#ef7b2d")
        left_target, = axis.plot([], [], "x", ms=9, mew=2, color="#00a65a")
        right_target, = axis.plot([], [], "x", ms=9, mew=2, color="#b5179e")
        status = axis.text(.02, .03, "", transform=axis.transAxes)
        artists.append((name, left_line, right_line, left_target, right_target, status))

    def update(frame):
        changed = []
        for name, ll, rl, lt, rt, status in artists:
            left, right = data[f"{name}_left"][frame], data[f"{name}_right"][frame]
            left_target = data[f"{name}_left_target"][frame]
            right_target = data[f"{name}_right_target"][frame]
            ll.set_data(left[:, 0], left[:, 2]); rl.set_data(right[:, 0], right[:, 2])
            lt.set_data([left_target[0]], [left_target[2]])
            rt.set_data([right_target[0]], [right_target[2]])
            clearance = 1000 * data[f"{name}_clearance"][frame]
            status.set_text(f"frame {frame:03d}  clearance {clearance:+.1f} mm")
            status.set_color("crimson" if clearance < 0 else "black")
            changed.extend((ll, rl, lt, rt, status))
        return changed

    animation = FuncAnimation(figure, update, frames=len(data["doosan_left"]), interval=66,
                              blit=True)
    figure.suptitle(args.title)
    figure.tight_layout()
    animation.save(args.output, writer=PillowWriter(fps=15), dpi=105); print(args.output)


if __name__ == "__main__": main()
