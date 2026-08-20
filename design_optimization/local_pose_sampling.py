"""Sampling rules for Local validation/test trajectories."""
from __future__ import annotations

import numpy as np


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.asarray(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                       (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                       (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))


def trim_episode_edges(time_s: np.ndarray, *, seconds: float = 0.15) -> np.ndarray:
    """Select frames after the first second and before the final second."""
    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or len(time) < 2 or seconds < 0:
        raise ValueError("invalid episode time array or trim duration")
    return (time >= time[0] + seconds) & (time <= time[-1] - seconds)


def relative_pose_sample(data, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    keep = trim_episode_edges(data["time_s"], seconds=0.15)
    valid = np.flatnonzero(keep)
    if len(valid) < 2:
        raise ValueError("episode has fewer than two frames after 1 s edge trimming")
    indices = valid[np.linspace(0, len(valid)-1, frame_count).round().astype(int)]
    quaternion = np.asarray(data["quaternion_wxyz"][indices], dtype=float)
    position = np.asarray(data["position_m"][indices], dtype=float)
    position = (position - position[0]) @ _quat_matrix(quaternion[0])
    inverse = quaternion[0] * np.asarray((1.0, -1.0, -1.0, -1.0))
    w1,x1,y1,z1=inverse; w2,x2,y2,z2=np.moveaxis(quaternion,-1,0)
    relative=np.stack((w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,
                       w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2),-1)
    return position, relative
