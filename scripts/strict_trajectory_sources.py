"""Traceable full-resolution trajectories for strict MuJoCo replays."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from design_optimization.local_pose_sampling import trim_episode_edges

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR_ROTATION = np.asarray(((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)))
HAND_TO_TCP_ROTATION = np.asarray(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))


@dataclass(frozen=True)
class RelativeTrajectory:
    time_s: np.ndarray
    position_m: np.ndarray
    quaternion_wxyz: np.ndarray
    hand: str
    source: str


def resample_trajectory(trajectory: RelativeTrajectory, maximum_frames: int = 300) -> RelativeTrajectory:
    """Interpolate a bounded trajectory on a uniform physical-time grid."""
    if maximum_frames < 2:
        raise ValueError("maximum_frames must be at least two")
    source_time = np.asarray(trajectory.time_s, dtype=float)
    if np.any(np.diff(source_time) <= 0):
        raise ValueError("trajectory time must be strictly increasing")
    target_time = np.linspace(source_time[0], source_time[-1], min(len(source_time), maximum_frames))
    position = np.column_stack([
        np.interp(target_time, source_time, trajectory.position_m[:, axis]) for axis in range(3)
    ])
    source_xyzw = np.asarray(trajectory.quaternion_wxyz)[:, (1, 2, 3, 0)]
    target_xyzw = Slerp(source_time, Rotation.from_quat(source_xyzw))(target_time).as_quat()
    quaternion = target_xyzw[:, (3, 0, 1, 2)]
    return RelativeTrajectory(
        time_s=target_time,
        position_m=position,
        quaternion_wxyz=quaternion,
        hand=trajectory.hand,
        source=trajectory.source,
    )


def place_relative_positions(relative_position_m: np.ndarray, *, hand: str,
                             anchor_z_m: float = 0.35) -> np.ndarray:
    """Place dataset/world-frame translations at one common tabletop anchor."""
    if hand not in ("left", "right"):
        raise ValueError(f"unknown hand: {hand}")
    anchor = np.asarray(((-0.18 if hand == "left" else 0.18), -0.45, anchor_z_m))
    relative = np.asarray(relative_position_m, dtype=float)
    if relative.ndim != 2 or relative.shape[1] != 3:
        raise ValueError("relative_position_m must have shape (frames, 3)")
    return anchor + relative


def minimum_safe_anchor_z(relative_min_z_m: float, *, clearance_m: float = 0.10,
                          lower_m: float = 0.20, upper_m: float = 0.45) -> float:
    """Choose the lowest common task anchor that keeps every TCP above the table."""
    required = float(clearance_m) - float(relative_min_z_m)
    if required > upper_m:
        raise ValueError(f"trajectory requires anchor z={required:.3f} m above allowed {upper_m:.3f} m")
    return max(float(lower_m), required)


@lru_cache(maxsize=None)
def local_task_anchor_z(task: str, clearance_m: float = 0.10) -> float:
    """Derive one robot-independent Z registration from every eligible Local episode."""
    root = ROOT / "data/processed/local_pose_benchmark"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    minima = []
    for episode in manifest["episodes"]:
        if episode["task"] != task or not episode.get("trajectory_edge_trim_eligible", True):
            continue
        values = np.load(root / episode["artifact"], allow_pickle=False)
        keep = trim_episode_edges(values["time_s"], seconds=0.15)
        position = np.asarray(values["position_m"][keep], dtype=float)
        if len(position) >= 2:
            minima.append(float(np.min(position[:, 2] - position[0, 2])))
    if not minima:
        raise ValueError(f"{task}: no eligible Local episodes for height registration")
    return minimum_safe_anchor_z(min(minima), clearance_m=clearance_m)


def _quat_inverse(q):
    return np.asarray(q) * np.asarray((1.0, -1.0, -1.0, -1.0))


def _quat_multiply(a, b):
    w1, x1, y1, z1 = np.moveaxis(np.broadcast_to(a, np.shape(b)), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack((w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2), axis=-1)


def _quat_matrix(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(((1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)),
                       (2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)),
                       (2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y))))


def task_anchor_rotations(task: str, relative_quaternions: np.ndarray) -> np.ndarray:
    """Apply one fixed hand-frame to TCP-frame calibration without task clamping."""
    relative = np.asarray([_quat_matrix(q) for q in relative_quaternions])
    hand_world = np.einsum("ij,njk->nik", DEFAULT_ANCHOR_ROTATION, relative)
    return np.einsum("nij,jk->nik", hand_world, HAND_TO_TCP_ROTATION)


def _relative(time_s, position, quaternion, hand, source):
    position = np.asarray(position, dtype=float); quaternion = np.asarray(quaternion, dtype=float)
    relative_q = _quat_multiply(_quat_inverse(quaternion[0]), quaternion)
    relative_q /= np.linalg.norm(relative_q, axis=1, keepdims=True)
    for i in range(1, len(relative_q)):
        if relative_q[i-1] @ relative_q[i] < 0:
            relative_q[i] *= -1
    time_s = np.asarray(time_s, dtype=float); time_s = time_s - time_s[0]
    # Position samples are already expressed in the dataset/world frame.  The
    # old implementation rotated translation by the first hand pose and then
    # rotated it a second time while placing the task in the MuJoCo scene.
    # That made the geometric path depend on the operator's initial wrist
    # attitude and could flip recorded upward motion through the tabletop.
    relative_position = position - position[0]
    return RelativeTrajectory(time_s, relative_position, relative_q, hand, source)


def load_relative_task_trajectory(domain: str, task: str, *, split: str = "test",
                                  episode_artifact: str | Path | None = None) -> RelativeTrajectory:
    if domain == "local":
        root = ROOT / "data/processed/local_pose_benchmark"
        if episode_artifact is not None:
            path = Path(episode_artifact)
            path = path if path.is_absolute() else root / path
            z = np.load(path, allow_pickle=False)
            keep = trim_episode_edges(z["time_s"], seconds=0.15)
            if int(keep.sum()) < 2:
                raise ValueError(f"{task}/{path.name}: insufficient frames after edge trimming")
            hand = str(z["hand"].item()) if "hand" in z else "left"
            return _relative(z["time_s"][keep], z["position_m"][keep],
                             z["quaternion_wxyz"][keep], hand, str(path))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        candidates = [x for x in manifest["episodes"] if x["task"] == task and x["split"] == split
                      and x.get("trajectory_edge_trim_eligible", True)]
        if not candidates:
            raise ValueError(f"{task}: no eligible Local {split} episode")
        scored = []
        for episode in candidates:
            values = np.load(root / episode["artifact"], allow_pickle=False)
            mask = trim_episode_edges(values["time_s"], seconds=0.15)
            path_length = float(np.linalg.norm(np.diff(values["position_m"][mask], axis=0), axis=1).sum())
            scored.append((path_length, episode))
        path_length, episode = max(scored, key=lambda item: item[0])
        if path_length <= 1e-3:
            raise ValueError(f"{task}: all eligible Local test episodes are static")
        path = root / episode["artifact"]; z = np.load(path, allow_pickle=False)
        keep = trim_episode_edges(z["time_s"], seconds=0.15)
        if int(keep.sum()) < 2:
            raise ValueError(f"{task}/{episode['hand']}: insufficient frames after edge trimming")
        return _relative(z["time_s"][keep], z["position_m"][keep], z["quaternion_wxyz"][keep], episode["hand"], str(path))
    rows = json.loads((ROOT / f"data/processed/external_domains/{domain}_samples.json").read_text(encoding="utf-8"))["splits"]["validation"]
    row = next(x for x in rows if x["task"] == task)
    if domain == "droid":
        directory, segment_text = row["source"].split(":")
        base = ROOT / "data/DROID" / directory
        path = next(base.glob("*.npz")); manifest = json.loads(next(base.glob("*manifest.json")).read_text(encoding="utf-8"))
        segment = manifest["segments"][int(segment_text)]; lo, hi = int(segment["start_frame"]), int(segment["end_frame"])+1
        cut1 = lo + int(.6*(hi-lo)); cut2 = lo + int(.8*(hi-lo)); z = np.load(path, allow_pickle=False)
        return _relative(z["time_s"][cut1:cut2], z["tcp_xyz"][cut1:cut2], z["tcp_quat_wxyz"][cut1:cut2], "right", str(path))
    if domain == "egodex":
        path = ROOT / "data/EgoDex/pose_only_test/egodex_pose_only_test.npz"; z = np.load(path, mmap_mode="r", allow_pickle=False)
        episode = int(row["episode"]); lo, hi = map(int, z["episode_offsets"][episode:episode+2]); hand = row["hand"]
        indices = np.flatnonzero(z[f"{hand}_confidence"][lo:hi] >= .8) + lo
        if len(indices) < 2:
            raise ValueError(f"{task}: fewer than two confident {hand} frames")
        # EgoDex relative arrays are already expressed in the episode anchor frame.
        time = np.asarray(z["time_s"][indices], dtype=float); position = np.asarray(z[f"{hand}_relative_xyz"][indices], dtype=float)
        quaternion = np.asarray(z[f"{hand}_relative_quat_wxyz"][indices], dtype=float)
        time -= time[0]; position -= position[0]
        quaternion = _quat_multiply(_quat_inverse(quaternion[0]), quaternion)
        quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
        return RelativeTrajectory(time, position, quaternion, hand, str(path))
    raise ValueError(f"unknown domain: {domain}")
