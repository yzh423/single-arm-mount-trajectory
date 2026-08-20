from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from design_optimization.egodex import EgoDexPoseDataset
from design_optimization.ik import solve_trajectory_multistart
from design_optimization.kinematics import assemble_from_deltas
from design_optimization.taskspace import canonical_egodex_targets
from design_optimization.trajectory import select_bimanual_collision_aware
from design_optimization.topology import load_templates

DEFAULT_DATA = Path(r"C:\Users\KelvinLM\Documents\Codex\Doosan_Dual_Quest3\data\EgoDex\pose_only_test\egodex_pose_only_test.npz")


def select_challenge_episode(dataset: EgoDexPoseDataset, episodes: np.ndarray,
                             maximum_frames: int) -> int:
    best_score, best_episode = -1.0, int(episodes[0])
    for episode in episodes:
        start, stop = int(dataset.offsets[episode]), int(dataset.offsets[episode + 1])
        if stop - start < 30: continue
        sample = np.linspace(start, stop - 1, min(maximum_frames, stop - start), dtype=np.int64)
        left = np.asarray(dataset.data["left_relative_xyz"][sample])
        right = np.asarray(dataset.data["right_relative_xyz"][sample])
        path = np.linalg.norm(np.diff(left, axis=0), axis=1).sum() + np.linalg.norm(
            np.diff(right, axis=0), axis=1).sum()
        span = np.linalg.norm(np.ptp(left, axis=0)) + np.linalg.norm(np.ptp(right, axis=0))
        score = float(path + 2 * span)
        if score > best_score: best_score, best_episode = score, int(episode)
    return best_episode


def richest_contiguous_window(episode: dict[str, np.ndarray], maximum_frames: int) -> np.ndarray:
    frame_count = len(episode["time_s"])
    if frame_count <= maximum_frames:
        return np.arange(frame_count, dtype=np.int64)
    left_step = np.linalg.norm(np.diff(episode["left_relative_xyz"], axis=0), axis=1)
    right_step = np.linalg.norm(np.diff(episode["right_relative_xyz"], axis=0), axis=1)
    motion = left_step + right_step
    window_motion = np.convolve(motion, np.ones(maximum_frames - 1), mode="valid")
    start = int(np.argmax(window_motion))
    return np.arange(start, start + maximum_frames, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collision-aware continuous EgoDex IK validation")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--robot", choices=("doosan", "xarm6", "ur5", "kinova"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--episode-id", type=int)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--beam", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "learning" / "continuous")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = EgoDexPoseDataset(args.data); split = getattr(dataset.split(), args.split)
    episode_id = args.episode_id if args.episode_id is not None else select_challenge_episode(
        dataset, split[:min(160, len(split))], args.max_frames)
    episode = dataset.episode(episode_id); frame_count = len(episode["time_s"])
    frame_indices = richest_contiguous_window(episode, args.max_frames)
    arrays = [torch.as_tensor(episode[key][frame_indices], device=args.device)
              for key in ("left_relative_xyz", "left_relative_quat_wxyz",
                          "right_relative_xyz", "right_relative_quat_wxyz")]
    task = canonical_egodex_targets(*arrays)
    learned = json.loads((ROOT / "reports" / "learning" /
                          f"{args.robot}_egodex_optimized.json").read_text())
    template = load_templates(ROOT / "reports" / "parametric_topology_audit.json",
                              device=args.device)[args.robot]
    design = assemble_from_deltas(template, torch.tensor(learned["optimized_deltas_m"],
                                                         device=args.device))
    best = learned["best"]; spacing, by, bz = best["base_spacing_m"], best["base_y_m"], best["base_z_m"]
    left_base = (-spacing / 2, by, bz); right_base = (spacing / 2, by, bz)

    def local_targets(world, base):
        result = world.clone(); result[:, :3, 3] -= torch.tensor(base, device=args.device)
        return result

    left = solve_trajectory_multistart(design, local_targets(task.left, left_base),
                                       seed_count=args.seeds, initial_iterations=args.iterations,
                                       tracking_iterations=max(20, args.iterations // 3))
    right = solve_trajectory_multistart(design, local_targets(task.right, right_base),
                                        seed_count=args.seeds, initial_iterations=args.iterations,
                                        tracking_iterations=max(20, args.iterations // 3), seed=20260807)
    path = select_bimanual_collision_aware(left, right, design, left_base, right_base,
                                           beam_width=args.beam)
    dt = float((episode["time_s"][frame_indices[-1]] - episode["time_s"][frame_indices[0]]) /
               max(1, len(frame_indices) - 1))
    metrics = path.metrics(dt)
    metadata = {"robot": args.robot, "split": args.split, "episode_id": episode_id,
                "task": str(dataset.tasks[episode_id]), "source": str(dataset.sources[episode_id]),
                "source_frames": frame_count, "evaluated_frames": len(frame_indices), "dt_s": dt,
                "seeds": args.seeds, "iterations": args.iterations, "beam": args.beam,
                "base_left_xyz": left_base, "base_right_xyz": right_base, "metrics": metrics}
    stem = f"{args.robot}_{args.split}_episode_{episode_id}"
    np.savez_compressed(args.output_dir / f"{stem}.npz", time_s=episode["time_s"][frame_indices],
                        left_q=path.left_q.cpu(), right_q=path.right_q.cpu(),
                        left_target=task.left.cpu(), right_target=task.right.cpu(),
                        self_clearance_m=path.self_clearance_m.cpu(),
                        dual_clearance_m=path.dual_clearance_m.cpu(),
                        table_clearance_m=path.table_clearance_m.cpu())
    (args.output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()
