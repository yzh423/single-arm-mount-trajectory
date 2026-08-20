from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from design_optimization.egodex import EgoDexPoseDataset
from design_optimization.ik import (IKResult, concatenate_ik_branches,
                                    deterministic_seeds, reverse_ik_time,
                                    solve_multistart, solve_trajectory_multistart)
from design_optimization.kinematics import assemble_from_deltas
from design_optimization.retiming import retime_joint_path
from design_optimization.taskspace import canonical_egodex_targets, mirrored_mount_transforms
from design_optimization.trajectory import select_bimanual_collision_aware
from design_optimization.topology import load_templates
from validate_continuous_egodex_cuda import richest_contiguous_window, select_challenge_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("finalists", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--episode-id", type=int)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--restart-interval", type=int, default=15)
    parser.add_argument("--restart-fraction", type=float, default=.25)
    parser.add_argument("--candidate-mode", choices=("warm", "independent"), default="warm")
    parser.add_argument("--bidirectional-candidates", action="store_true",
                        help="merge forward and reverse warm-start branches")
    parser.add_argument("--candidate", type=int, default=0,
                        help="rank after sorting Pareto candidates by failure objective")
    parser.add_argument("--beam", type=int, default=24)
    parser.add_argument("--max-joint-speed-rad-s", type=float)
    parser.add_argument("--max-joint-step-rad", type=float, default=.5,
                        help="hard continuity bound shared by candidate tracking and beam")
    parser.add_argument("--retime-to-joint-speed-rad-s", type=float, default=3.0)
    parser.add_argument("--outer-elbow-weight", type=float, default=0.0)
    parser.add_argument("--hard-collision", action="store_true")
    parser.add_argument("--hard-minimum-clearance-mm", type=float, default=0.0)
    parser.add_argument("--proxy-collision-weight", type=float, default=0.0,
                        help="Screening-only capsule cost; keep zero for fair topology ranking")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports" / "learning" / "continuous_formal")
    parser.add_argument("--candidate-cache-dir", type=Path,
                        help="content-addressed per-arm IK candidate cache")
    args = parser.parse_args(); payload = json.loads(args.finalists.read_text())
    config = payload["config"]; robot = config["topology"]
    candidates = sorted(payload["pareto_candidates"], key=lambda row: row["objectives"][0])
    candidate = candidates[args.candidate]
    dataset = EgoDexPoseDataset(config["data"]["path"]); episodes = getattr(dataset.split(), args.split)
    episode_id = args.episode_id if args.episode_id is not None else select_challenge_episode(
        dataset, episodes[:min(160, len(episodes))], args.max_frames)
    episode = dataset.episode(episode_id); indices = richest_contiguous_window(episode, args.max_frames)
    arrays = [torch.as_tensor(episode[key][indices], device=args.device)
              for key in ("left_relative_xyz", "left_relative_quat_wxyz",
                          "right_relative_xyz", "right_relative_quat_wxyz")]
    task = canonical_egodex_targets(*arrays)
    dt = float((episode["time_s"][indices[-1]]-episode["time_s"][indices[0]])/max(1,len(indices)-1))
    maximum_delta = (args.max_joint_speed_rad_s * dt
                     if args.max_joint_speed_rad_s is not None
                     else args.max_joint_step_rad)
    template = load_templates(ROOT / "reports" / "parametric_topology_audit.json",
                              device=args.device)[robot]
    design = assemble_from_deltas(template, torch.tensor(candidate["strict_deltas_m"], device=args.device))
    left_base, right_base = mirrored_mount_transforms(
        torch.tensor([candidate["base_spacing_m"]], device=args.device),
        torch.tensor([candidate["base_y_m"]], device=args.device),
        torch.tensor([candidate["base_z_m"]], device=args.device),
        torch.deg2rad(torch.tensor([candidate["mount_roll_deg"]], device=args.device)))

    def local_targets(world, base):
        return torch.linalg.inv(base[0])[None] @ world

    def progress(side):
        started = time.perf_counter()
        def report(done, total):
            if done == 0:
                print(f"[{robot} {side}] preparing {total} frames", flush=True)
            elif done == 1 or done == total or done % 10 == 0:
                elapsed = time.perf_counter() - started
                eta = elapsed * (total - done) / max(done, 1)
                print(f"[{robot} {side}] {done}/{total} frames, ETA {eta:.0f}s",
                      flush=True)
        return report

    cache_payload = {
        "finalists_sha256": hashlib.sha256(args.finalists.read_bytes()).hexdigest(),
        "split": args.split, "episode_id": int(episode_id),
        "indices_sha256": hashlib.sha256(
            np.asarray(indices, dtype=np.int64).tobytes()).hexdigest(),
        "candidate": args.candidate, "mode": args.candidate_mode,
        "bidirectional_candidates": args.bidirectional_candidates,
        "seeds": args.seeds, "iterations": args.iterations,
        "restart_interval": args.restart_interval,
        "restart_fraction": args.restart_fraction,
        "maximum_delta": maximum_delta,
    }
    cache_hash = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()[:16]

    def solve_side(side, targets, seed):
        def load_result(path):
            state = torch.load(path, map_location=args.device, weights_only=True)
            return IKResult(**{key: state[key].to(args.device) for key in
                              ("q", "position_error_m", "orientation_error_rad",
                               "sigma_min", "success")})

        def save_result(path, result, phase):
            state = {key: getattr(result, key).detach().cpu() for key in
                     ("q", "position_error_m", "orientation_error_rad",
                      "sigma_min", "success")}
            state["provenance"] = {**cache_payload, "phase": phase}
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(state, temporary)
            temporary.replace(path)
            print(f"[{robot} {side}] saved {phase} cache {path.name}", flush=True)

        cache = None
        if args.candidate_cache_dir:
            args.candidate_cache_dir.mkdir(parents=True, exist_ok=True)
            cache = args.candidate_cache_dir / (
                f"{robot}_{args.split}_{episode_id}_{cache_hash}_{side}.pt")
        if cache is not None and cache.exists():
            print(f"[{robot} {side}] loaded candidate cache {cache.name}", flush=True)
            return load_result(cache)
        if args.candidate_mode == "independent":
            result = solve_multistart(
                design, targets, deterministic_seeds(template, args.seeds, seed),
                iterations=args.iterations)
        else:
            forward_cache = (cache.with_name(cache.stem + "_forward.pt")
                             if cache is not None else None)
            if forward_cache is not None and forward_cache.exists():
                print(f"[{robot} {side}] loaded forward cache {forward_cache.name}",
                      flush=True)
                result = load_result(forward_cache)
            else:
                result = solve_trajectory_multistart(
                    design, targets, seed_count=args.seeds,
                    initial_iterations=args.iterations,
                    tracking_iterations=max(25, args.iterations // 3), seed=seed,
                    restart_interval=args.restart_interval,
                    restart_fraction=args.restart_fraction,
                    maximum_joint_delta_rad=maximum_delta,
                    progress_callback=progress(side))
                if forward_cache is not None:
                    save_result(forward_cache, result, "forward")
            if args.bidirectional_candidates:
                reverse_cache = (cache.with_name(cache.stem + "_reverse.pt")
                                 if cache is not None else None)
                if reverse_cache is not None and reverse_cache.exists():
                    print(f"[{robot} {side}] loaded reverse cache {reverse_cache.name}",
                          flush=True)
                    backward = load_result(reverse_cache)
                else:
                    backward = reverse_ik_time(solve_trajectory_multistart(
                        design, torch.flip(targets, dims=(0,)), seed_count=args.seeds,
                        initial_iterations=args.iterations,
                        tracking_iterations=max(25, args.iterations // 3),
                        seed=seed + 100000,
                        restart_interval=args.restart_interval,
                        restart_fraction=args.restart_fraction,
                        maximum_joint_delta_rad=maximum_delta,
                        progress_callback=progress(side + " reverse")))
                    if reverse_cache is not None:
                        save_result(reverse_cache, backward, "reverse_chronological")
                result = concatenate_ik_branches(
                    result, backward)
        if cache is not None:
            save_result(cache, result, "combined")
        return result

    left = solve_side("left", local_targets(task.left, left_base), 20260806)
    right = solve_side("right", local_targets(task.right, right_base), 20260807)
    path = select_bimanual_collision_aware(left, right, design, left_base[0], right_base[0],
                                           beam_width=args.beam,
                                           maximum_joint_step_rad=args.max_joint_step_rad,
                                           outer_elbow_weight=args.outer_elbow_weight,
                                           collision_weight=args.proxy_collision_weight,
                                           hard_collision=args.hard_collision,
                                           hard_minimum_clearance_m=1e-3 * args.hard_minimum_clearance_mm,
                                           progress_callback=progress("beam"))
    metrics = path.metrics(dt); args.output_dir.mkdir(parents=True,exist_ok=True)
    nominal_time = torch.as_tensor(episode['time_s'][indices], device=args.device)
    combined_q = torch.cat((path.left_q, path.right_q), dim=-1)
    retimed = retime_joint_path(combined_q, nominal_time,
                                torch.tensor(args.retime_to_joint_speed_rad_s,
                                             device=args.device))
    nominal_duration = float(nominal_time[-1] - nominal_time[0])
    retimed_duration = float(retimed.time_s[-1] - retimed.time_s[0])
    metrics.update({"retime_joint_speed_limit_rad_s": args.retime_to_joint_speed_rad_s,
                    "nominal_duration_s": nominal_duration,
                    "retimed_duration_s": retimed_duration,
                    "retime_slowdown": retimed_duration / max(nominal_duration, 1e-9),
                    "retimed_minimum_speed_scale": float(retimed.speed_scale.min())})
    stem = args.output_dir/f"{robot}_{args.split}_episode_{episode_id}"
    meta = {"robot":robot,"split":args.split,"episode_id":episode_id,
            "task":str(dataset.tasks[episode_id]),"frames":len(indices),"dt_s":dt,
            "source_finalists":str(args.finalists),"mount_roll_deg":candidate["mount_roll_deg"],
            "candidate_rank": args.candidate,
            "base_spacing_m":candidate["base_spacing_m"],
            "max_joint_speed_rad_s":args.max_joint_speed_rad_s,"metrics":metrics}
    meta["restart_interval"] = args.restart_interval
    meta["restart_fraction"] = args.restart_fraction
    meta["candidate_mode"] = args.candidate_mode
    meta["bidirectional_candidates"] = args.bidirectional_candidates
    meta["outer_elbow_weight"] = args.outer_elbow_weight
    meta["proxy_collision_weight"] = args.proxy_collision_weight
    meta["hard_collision"] = args.hard_collision
    meta["hard_minimum_clearance_mm"] = args.hard_minimum_clearance_mm
    meta["max_joint_step_rad"] = args.max_joint_step_rad
    meta["candidate_cache_key"] = cache_hash
    meta["candidate_availability"] = {
        "left_any_success_fraction": float(left.success.any(dim=-1).float().mean()),
        "right_any_success_fraction": float(right.success.any(dim=-1).float().mean()),
        "left_best_position_rmse_mm": float(
            1000 * torch.sqrt(left.position_error_m.amin(dim=-1).square().mean())),
        "right_best_position_rmse_mm": float(
            1000 * torch.sqrt(right.position_error_m.amin(dim=-1).square().mean())),
    }
    np.savez_compressed(stem.with_suffix('.npz'),time_s=episode['time_s'][indices],
                        left_q=path.left_q.cpu(),right_q=path.right_q.cpu(),
                        left_target=task.left.cpu(),right_target=task.right.cpu(),
                        self_clearance_m=path.self_clearance_m.cpu(),dual_clearance_m=path.dual_clearance_m.cpu(),
                        table_clearance_m=path.table_clearance_m.cpu())
    stem.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
