"""Run the isolated coarse-to-strict mount/IK fidelity pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOTS = ("xarm6", "openarm")
OUTPUT = ROOT / "reports/single_arm/mount_ik_fidelity_pilot"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", nargs="+", default=list(DEFAULT_ROBOTS))
    parser.add_argument("--coarse-candidates", type=int, default=4096)
    parser.add_argument("--proxy-retain", type=int, default=256)
    parser.add_argument("--strict-window-retain", type=int, default=128)
    parser.add_argument("--full-retain", type=int, default=24)
    parser.add_argument("--final-retain", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser


def stage_counts(args: argparse.Namespace) -> dict[str, int]:
    if args.smoke:
        return {"coarse": 32, "proxy": 12, "window": 6, "full": 2, "final": 2}
    counts = {
        "coarse": int(args.coarse_candidates),
        "proxy": int(args.proxy_retain),
        "window": int(args.strict_window_retain),
        "full": int(args.full_retain),
        "final": int(args.final_retain),
    }
    if not (counts["coarse"] >= counts["proxy"] >= counts["window"]
            >= counts["full"] >= counts["final"] >= 1):
        raise ValueError("pilot stage counts must be positive and non-increasing")
    return counts


def pilot_fingerprint(
    *, robot: str, episode_sha256: str, model_sha256: str,
    tcp_translation: tuple[float, float, float], counts: dict[str, int],
) -> str:
    payload = {
        "robot": robot,
        "episode_sha256": episode_sha256,
        "model_sha256": model_sha256,
        "tcp_translation": list(tcp_translation),
        "counts": counts,
        "policy": "mount_ik_fidelity_pilot_v2_atomic_stages",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def stage_input_fingerprint(
    *, robot_fingerprint: str, stage: str, candidates,
) -> str:
    payload = {
        "robot_fingerprint": robot_fingerprint,
        "stage": stage,
        "candidates": _jsonable(candidates),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_stage_records(path: Path, expected_fingerprint: str) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("stage_input_fingerprint") != expected_fingerprint:
            return []
        records = payload.get("records")
        return records if isinstance(records, list) else []
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []


def _save_stage_records(
    path: Path, stage_fingerprint: str, records: list[dict], *, complete: bool,
) -> None:
    _atomic_json(path, {
        "stage_input_fingerprint": stage_fingerprint,
        "complete": bool(complete),
        "record_count": len(records),
        "records": records,
    })


def coarse_fingerprint(
    *, robots, episode_sha256: str, candidate_count: int,
    model_contracts: dict[str, dict], algorithm_sha256: str = "",
) -> str:
    payload = {
        "robots": list(robots), "episode_sha256": episode_sha256,
        "candidate_count": int(candidate_count),
        "model_contracts": model_contracts,
        "algorithm_sha256": algorithm_sha256,
        "policy": "mount_ik_fidelity_coarse_v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def coarse_algorithm_fingerprint() -> str:
    paths = (
        ROOT / "scripts/run_thirteen_arm_dense_search.py",
        ROOT / "design_optimization/fidelity_funnel.py",
        ROOT / "design_optimization/ik.py",
        ROOT / "design_optimization/collision.py",
        ROOT / "design_optimization/urdf_chain.py",
        ROOT / "design_optimization/kinematics.py",
        ROOT / "design_optimization/search_policy.py",
    )
    payload = {
        str(path.relative_to(ROOT)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def official_model_contract(robot: str) -> dict:
    """Fingerprint source, referenced assets, TCP, and collision semantics."""
    from scripts.strict_urdf_model_audit import MODELS

    entry = MODELS[robot]
    source_hash = hashlib.sha256(entry.path.read_bytes()).hexdigest()
    asset_hashes: dict[str, str] = {}
    if entry.path.suffix.lower() == ".urdf":
        root = ET.parse(entry.path).getroot()
        for mesh in root.findall(".//mesh"):
            filename = mesh.get("filename")
            if not filename:
                continue
            if filename.startswith("package://"):
                package_path = filename[len("package://"):]
                package, relative = package_path.split("/", 1)
                package_root = (entry.package_roots or {}).get(package)
                path = None if package_root is None else package_root / relative
            else:
                path = entry.path.parent / filename
            if path is not None and path.is_file():
                asset_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                asset_hashes[filename] = "missing"
    collision_sources = (
        ROOT / "reports/single_arm/collision_proxy_profiles.json",
        ROOT / "scripts/solve_strict_urdf_task_cache.py",
        ROOT / "scripts/search_strict_urdf_mount.py",
        ROOT / "scripts/strict_mujoco_ik.py",
        ROOT / "scripts/rolling_multibranch_ik.py",
    )
    collision_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest()
        for path in collision_sources
    }
    payload = {
        "source_model": str(entry.path.resolve()),
        "source_sha256": source_hash,
        "asset_sha256": asset_hashes,
        "tcp_translation_m": list(entry.tool_translation_m),
        "tcp_quaternion_wxyz": list(entry.tool_quaternion_wxyz),
        "tcp_authority": entry.tcp_authority,
        "collision_sha256": collision_hashes,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def coarse_model_contract(robot: str) -> dict:
    """Use geometry/TCP/proxy evidence without coupling to layered IK code."""
    payload = official_model_contract(robot)
    strict_only = ("strict_mujoco_ik.py", "rolling_multibranch_ik.py")
    payload["collision_sha256"] = {
        key: value for key, value in payload["collision_sha256"].items()
        if not key.endswith(strict_only)
    }
    payload.pop("contract_sha256", None)
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def load_completed_robot_result(
    output: Path, robot: str, expected_fingerprint: str,
) -> dict | None:
    """Load an atomic robot checkpoint only when all strict evidence matches."""
    path = output / "robots" / f"{robot}.json"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        cache = Path(result["cache"])
        if not cache.is_absolute():
            cache = ROOT / cache
        if (result.get("robot") != robot
                or result.get("fingerprint") != expected_fingerprint
                or "winner" not in result or not cache.is_file()):
            return None
        with np.load(cache) as payload:
            if "q" not in payload or len(payload["q"]) == 0:
                return None
        return result
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def _selected_episode() -> dict:
    manifest = json.loads((
        ROOT / "data/processed/local_pose_benchmark/manifest.json"
    ).read_text(encoding="utf-8"))
    rows = [row for row in manifest["episodes"]
            if row["task"] == "pick-right-left"
            and Path(row["source_path"]).stem == "teleop_20260805_154215"]
    if len(rows) != 1:
        raise RuntimeError("pilot episode manifest row is missing or ambiguous")
    return rows[0]


def _full_mount(active_mount) -> np.ndarray:
    x, y, z, yaw = np.asarray(active_mount, dtype=float)
    return np.asarray((x, y, z, 0.0, yaw, 0.0))


def _proxy_score(row: dict) -> tuple[float, ...]:
    return (
        float(row["episode_success_rate"]), float(row["frame_coverage"]),
        -float(row["position_rmse_m"]), -float(row["orientation_rmse_rad"]),
    )


def full_episode_optimism_metrics(
    proxy_candidate_ids, proxy_episode_success,
    strict_candidate_ids, strict_episode_success,
) -> dict[str, object]:
    """Compare proxy episode claims only where strict full episodes were run."""
    proxy_ids = np.asarray(proxy_candidate_ids)
    proxy_success = np.asarray(proxy_episode_success, dtype=bool)
    strict_ids = np.asarray(strict_candidate_ids)
    strict_success = np.asarray(strict_episode_success, dtype=bool)
    if (proxy_ids.ndim != 1 or strict_ids.ndim != 1
            or proxy_success.shape != proxy_ids.shape
            or strict_success.shape != strict_ids.shape):
        raise ValueError("candidate IDs and episode outcomes must be one-dimensional")
    if len(strict_ids) == 0:
        raise ValueError("at least one strict full-episode candidate is required")
    lookup = {candidate_id: bool(outcome)
              for candidate_id, outcome in zip(proxy_ids.tolist(), proxy_success)}
    if len(lookup) != len(proxy_ids) or any(
            candidate_id not in lookup for candidate_id in strict_ids.tolist()):
        raise ValueError("strict candidate IDs must be a traceable proxy subset")
    optimistic = sum(
        lookup[candidate_id] and not bool(strict_outcome)
        for candidate_id, strict_outcome in zip(strict_ids.tolist(), strict_success))
    return {
        "full_episode_compared_candidate_count": int(len(strict_ids)),
        "optimistic_episode_pass_count": int(optimistic),
        "optimistic_episode_pass_rate": float(optimistic / len(strict_ids)),
    }


def _summary_record(record: dict) -> dict:
    excluded = {
        "q", "success", "position_error_m", "orientation_error_rad",
        "collision", "table_collision", "self_collision", "edge_collision",
        "joint_discontinuity", "velocity_violation", "branch_count",
        "chosen_branch_index", "recovery_mode", "joint_limit_margin",
        "singularity_margin",
    }
    summary = {key: value for key, value in record.items() if key not in excluded}
    for name in ("position_error_m", "orientation_error_rad", "branch_count"):
        if name in record:
            values = np.asarray(record[name], dtype=float)
            summary[name + "_mean"] = float(np.nanmean(values))
            summary[name + "_p95"] = float(np.nanquantile(values, .95))
            summary[name + "_max"] = float(np.nanmax(values))
    if "recovery_mode" in record:
        values, counts = np.unique(np.asarray(record["recovery_mode"]).astype(str), return_counts=True)
        summary["recovery_counts"] = {
            str(value): int(count) for value, count in zip(values, counts)}
    return summary


def _official_velocity_limits(robot: str, common_limit: float) -> tuple[np.ndarray, str]:
    from scripts.strict_urdf_model_audit import MODELS

    entry = MODELS[robot]
    root = ET.parse(entry.path).getroot()
    limits = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("velocity"):
            limits[joint.get("name")] = float(limit.get("velocity"))
    if all(name in limits and limits[name] > 0 for name in entry.joints):
        return np.asarray([limits[name] for name in entry.joints]), "official_urdf"
    return np.full(len(entry.joints), common_limit), "common_fallback_missing_vendor_velocity"


def _coarse_result_complete(
    path: Path, robots: list[str], shortlist_size: int,
    expected_fingerprint: str,
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (payload.get("pilot_coarse_fingerprint") == expected_fingerprint
                and all(len(payload["robots"][robot]["per_task"]["pick-right-left"]
                            ["diverse_shortlist"]) == shortlist_size
                        for robot in robots))
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _run_coarse(args, counts: dict[str, int], episode: dict, output: Path) -> Path:
    coarse = output / "coarse_candidates.json"
    contracts = {robot: coarse_model_contract(robot) for robot in args.robots}
    fingerprint = coarse_fingerprint(
        robots=args.robots, episode_sha256=episode["content_sha256"],
        candidate_count=counts["coarse"], model_contracts=contracts,
        algorithm_sha256=coarse_algorithm_fingerprint())
    if _coarse_result_complete(
            coarse, args.robots, counts["proxy"], fingerprint):
        return coarse
    command = [
        sys.executable, "scripts/run_thirteen_arm_dense_search.py",
        "--candidates", str(counts["coarse"]), "--candidate-batch", "64",
        "--robots", *args.robots,
        "--episode-artifact", episode["artifact"],
        "--task", "pick-right-left", "--shortlist-size", str(counts["proxy"]),
        "--output", str(coarse),
    ]
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f"GPU coarse shortlist failed with return code {result.returncode}")
    payload = json.loads(coarse.read_text(encoding="utf-8"))
    payload["pilot_coarse_fingerprint"] = fingerprint
    payload["pilot_model_contracts"] = contracts
    _atomic_json(coarse, payload)
    if not _coarse_result_complete(
            coarse, args.robots, counts["proxy"], fingerprint):
        raise RuntimeError("GPU coarse shortlist artifact is incomplete")
    return coarse


def _evaluate_one_mount(
    *, robot: str, candidate_id: int, active_mount: np.ndarray,
    targets: np.ndarray, quaternions: np.ndarray, time_s: np.ndarray,
    scope: str, smoke: bool, velocity_limit,
) -> dict:
    from scripts.search_strict_urdf_mount import _evaluate_layered_model_path
    from scripts.solve_strict_urdf_task_cache import build_model
    from scripts.strict_urdf_model_audit import MODELS

    mount = _full_mount(active_mount)
    model = build_model(robot, mount[:3], mount[3], mount[4], mount[5])
    record = _evaluate_layered_model_path(
        model=model, joint_names=MODELS[robot].joints,
        candidate_id=candidate_id, targets=targets,
        target_quaternions=quaternions, time_s=time_s, scope=scope,
        candidates_per_frame=3 if smoke else (4 if scope == "window" else 8),
        global_seed_count=4 if smoke else (8 if scope == "window" else 16),
        iterations=50 if smoke else (80 if scope == "window" else 120),
        velocity_limit_rad_s=velocity_limit,
        maximum_frame_jump_rad=np.deg2rad(25.0),
        horizon=4 if smoke else 12,
        beam_width=3 if smoke else 8,
    )
    record["mount"] = mount
    return record


def _window_record(
    *, robot: str, row: dict, targets: np.ndarray, quaternions: np.ndarray,
    time_s: np.ndarray, windows, smoke: bool, velocity_limit,
) -> dict:
    from scripts.search_strict_urdf_mount import _layered_record_metrics

    records = []
    for window in windows:
        records.append(_evaluate_one_mount(
            robot=robot, candidate_id=int(row["candidate_id"]),
            active_mount=np.asarray(row["mount"]),
            targets=targets[window], quaternions=quaternions[window],
            time_s=time_s[window], scope="window", smoke=smoke,
            velocity_limit=velocity_limit))
    concatenate = lambda name: np.concatenate([np.asarray(item[name]) for item in records])
    merged = _layered_record_metrics(
        scope="window", candidate_id=int(row["candidate_id"]),
        pose_success=concatenate("success"),
        position_error_m=concatenate("position_error_m"),
        orientation_error_rad=concatenate("orientation_error_rad"),
        state_collision=concatenate("table_collision") | concatenate("self_collision"),
        edge_collision=concatenate("edge_collision"))
    merged.update({
        "mount": _full_mount(row["mount"]),
        "proxy_score": _proxy_score(row),
        "window_ranges": [[int(window[0]), int(window[-1])] for window in windows],
        "window_records": [_summary_record(item) for item in records],
        "branch_count": concatenate("branch_count"),
        "recovery_mode": concatenate("recovery_mode"),
        "position_error_m": concatenate("position_error_m"),
        "orientation_error_rad": concatenate("orientation_error_rad"),
    })
    return merged


def _write_cache(
    *, path: Path, robot: str, record: dict, targets: np.ndarray,
    quaternions: np.ndarray, time_s: np.ndarray,
) -> None:
    from design_optimization.episode_follow_metrics import failure_reason_diagnostics
    from scripts.solve_strict_urdf_task_cache import build_model, evaluate_q_path
    from scripts.strict_urdf_model_audit import MODELS

    mount = np.asarray(record["mount"], dtype=float)
    model = build_model(robot, mount[:3], mount[3], mount[4], mount[5])
    reached, reached_quaternion, position_error, orientation_error = evaluate_q_path(
        model, MODELS[robot].joints, record["q"], targets, quaternions)
    failure = failure_reason_diagnostics(
        position_error_m=position_error, orientation_error_rad=orientation_error,
        table_collision=record["table_collision"],
        self_collision=record["self_collision"],
        joint_discontinuity=record["joint_discontinuity"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, q=record["q"], target_xyz_m=targets,
        target_quaternion_wxyz=quaternions, time_s=time_s,
        reached_xyz_m=reached, reached_quaternion_wxyz=reached_quaternion,
        position_error_m=position_error,
        orientation_error_rad=orientation_error,
        success=record["success"],
        joint_discontinuity=record["joint_discontinuity"],
        recovery_mode=record["recovery_mode"],
        failure_reason=failure["primary_per_frame"],
        table_collision=record["table_collision"],
        self_collision=record["self_collision"],
        edge_collision=record["edge_collision"],
        base_xyz_m=mount[:3], tilt_deg=np.asarray(mount[3]),
        yaw_deg=np.asarray(mount[4]), roll_deg=np.asarray(mount[5]),
        joint_names=np.asarray(MODELS[robot].joints))
    _atomic_json(path.with_suffix(".json"), {
        "robot": robot, "task": "pick-right-left",
        "status": "pass" if record["episode_success"] else "fail",
        "episode_success": record["episode_success"],
        "frame_coverage": record["frame_coverage"],
        "base_xyz_m": mount[:3], "tilt_deg": mount[3],
        "yaw_deg": mount[4], "roll_deg": mount[5],
        "candidate_id": record["candidate_id"],
        "planner_type": "layered_receding_horizon",
    })


def _render_video(*, robot: str, cache: Path, output: Path) -> Path:
    video = output / "videos" / f"{robot}.mp4"
    metadata = video.with_suffix(".json")
    if video.is_file() and video.stat().st_size > 0 and metadata.is_file():
        return video
    command = [
        sys.executable, "scripts/render_strict_single_arm_task.py",
        "--domain", "local", "--robot", robot, "--task", "pick-right-left",
        "--cache", str(cache), "--output", str(video),
    ]
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode or not video.is_file() or video.stat().st_size == 0:
        raise RuntimeError(
            f"MuJoCo render failed for {robot} with return code {result.returncode}")
    return video


def _restore_strict_record(record: dict) -> dict:
    """Restore NumPy arrays after a JSON final-stage checkpoint reload."""
    boolean = {
        "success", "collision", "table_collision", "self_collision",
        "edge_collision", "joint_discontinuity", "velocity_violation",
        "acceleration_warning",
    }
    integer = {"branch_count", "chosen_branch_index"}
    textual = {"recovery_mode"}
    for name in (
        "q", "success", "position_error_m", "orientation_error_rad",
        "collision", "table_collision", "self_collision", "edge_collision",
        "joint_discontinuity", "velocity_violation", "acceleration_warning",
        "branch_count", "chosen_branch_index", "recovery_mode",
        "joint_limit_margin", "singularity_margin", "mount",
    ):
        if name not in record:
            continue
        dtype = bool if name in boolean else (int if name in integer else
                (str if name in textual else float))
        record[name] = np.asarray(record[name], dtype=dtype)
    return record


def _run_robot(
    *, args, counts, episode, coarse_payload, targets, quaternions, time_s,
    windows, output,
) -> dict:
    from design_optimization.best_first_mount_search import generate_local_population
    from design_optimization.fidelity_funnel import (
        cross_fidelity_metrics, select_diverse_candidates)
    from design_optimization.installation_search_space import first_version_bounds
    from scripts.strict_urdf_model_audit import MODELS

    robot = args.current_robot
    started = time.perf_counter()
    entry = MODELS[robot]
    proxy_rows = coarse_payload["robots"][robot]["per_task"]["pick-right-left"]["diverse_shortlist"]
    active_mounts = np.asarray([row["mount"] for row in proxy_rows], dtype=float)
    proxy_scores = [_proxy_score(row) for row in proxy_rows]
    full_space = first_version_bounds(0)
    lower = full_space.lower[[0, 1, 2, 4]]
    upper = full_space.upper[[0, 1, 2, 4]]
    window_indices = select_diverse_candidates(
        active_mounts, proxy_scores, lower=lower, upper=upper,
        count=counts["window"], pool_size=len(active_mounts))
    common_velocity = np.full(len(entry.joints), np.pi)
    robot_fingerprint = pilot_fingerprint(
        robot=robot, episode_sha256=episode["content_sha256"],
        model_sha256=official_model_contract(robot)["contract_sha256"],
        tcp_translation=entry.tool_translation_m, counts=counts)
    stage_dir = output / "stages" / robot
    window_started = time.perf_counter()
    window_rows = [proxy_rows[index] for index in window_indices]
    window_signature = stage_input_fingerprint(
        robot_fingerprint=robot_fingerprint, stage="strict_window",
        candidates=[{"candidate_id": row["candidate_id"], "mount": row["mount"]}
                    for row in window_rows])
    window_path = stage_dir / "strict_window.partial.json"
    restored_window = {
        int(record["candidate_id"]): record
        for record in load_stage_records(window_path, window_signature)}
    window_records = []
    for position, row in enumerate(window_rows, start=1):
        candidate_id = int(row["candidate_id"])
        record = restored_window.get(candidate_id)
        if record is None:
            record = _summary_record(_window_record(
                robot=robot, row=row, targets=targets,
                quaternions=quaternions, time_s=time_s, windows=windows,
                smoke=args.smoke, velocity_limit=common_velocity))
        window_records.append(record)
        _save_stage_records(
            window_path, window_signature, window_records,
            complete=position == len(window_rows))
        print(json.dumps({
            "robot": robot, "stage": "strict_window", "done": position,
            "total": len(window_rows), "candidate_id": candidate_id,
            "coverage": record["frame_coverage"],
        }), flush=True)
    window_elapsed = time.perf_counter() - window_started
    window_mounts = np.asarray([np.asarray(record["mount"])[[0, 1, 2, 4]]
                                for record in window_records])
    window_scores = [record["rank"] for record in window_records]
    full_indices = select_diverse_candidates(
        window_mounts, window_scores, lower=lower, upper=upper,
        count=counts["full"], pool_size=len(window_mounts))
    full_started = time.perf_counter()
    full_sources = [window_records[index] for index in full_indices]
    full_base_signature = stage_input_fingerprint(
        robot_fingerprint=robot_fingerprint, stage="full_episode_base",
        candidates=[{"candidate_id": row["candidate_id"], "mount": row["mount"]}
                    for row in full_sources])
    full_base_path = stage_dir / "full_episode_base.partial.json"
    restored_full_base = {
        int(record["candidate_id"]): record
        for record in load_stage_records(full_base_path, full_base_signature)}
    full_base_records = []
    for position, source in enumerate(full_sources, start=1):
        candidate_id = int(source["candidate_id"])
        record = restored_full_base.get(candidate_id)
        if record is None:
            record = _summary_record(_evaluate_one_mount(
                robot=robot, candidate_id=candidate_id,
                active_mount=np.asarray(source["mount"])[[0, 1, 2, 4]],
                targets=targets, quaternions=quaternions, time_s=time_s,
                scope="full_episode", smoke=args.smoke,
                velocity_limit=common_velocity))
        full_base_records.append(record)
        _save_stage_records(
            full_base_path, full_base_signature, full_base_records,
            complete=position == len(full_sources))
        print(json.dumps({
            "robot": robot, "stage": "full_episode_base", "done": position,
            "total": len(full_sources), "candidate_id": candidate_id,
            "coverage": record["frame_coverage"],
        }), flush=True)

    # Exact-aware local refinement around up to four strict basins.
    full_mounts_for_refinement = np.asarray([
        np.asarray(record["mount"])[[0, 1, 2, 4]]
        for record in full_base_records])
    center_indices = select_diverse_candidates(
        full_mounts_for_refinement, [record["rank"] for record in full_base_records],
        lower=lower, upper=upper, count=min(4, len(full_base_records)),
        pool_size=len(full_base_records))
    centers = full_mounts_for_refinement[center_indices]
    local_mounts, _, _ = generate_local_population(
        centers, lower, upper, per_region=1 if args.smoke else 2, seed=814)
    next_candidate_id = max(int(row["candidate_id"]) for row in proxy_rows) + 1
    known = {tuple(np.round(np.asarray(record["mount"])[[0, 1, 2, 4]], 10))
             for record in full_base_records}
    local_rows = []
    for mount in local_mounts:
        if tuple(np.round(mount, 10)) in known:
            continue
        local_rows.append({"candidate_id": next_candidate_id, "mount": mount})
        next_candidate_id += 1
    local_signature = stage_input_fingerprint(
        robot_fingerprint=robot_fingerprint, stage="full_episode_local",
        candidates=local_rows)
    local_path = stage_dir / "full_episode_local.partial.json"
    restored_local = {
        int(record["candidate_id"]): record
        for record in load_stage_records(local_path, local_signature)}
    local_records = []
    for position, row in enumerate(local_rows, start=1):
        candidate_id = int(row["candidate_id"])
        record = restored_local.get(candidate_id)
        if record is None:
            record = _summary_record(_evaluate_one_mount(
                robot=robot, candidate_id=candidate_id,
                active_mount=np.asarray(row["mount"]), targets=targets,
                quaternions=quaternions, time_s=time_s,
                scope="full_episode", smoke=args.smoke,
                velocity_limit=common_velocity))
        local_records.append(record)
        _save_stage_records(
            local_path, local_signature, local_records,
            complete=position == len(local_rows))
        print(json.dumps({
            "robot": robot, "stage": "full_episode_local", "done": position,
            "total": len(local_rows), "candidate_id": candidate_id,
            "coverage": record["frame_coverage"],
        }), flush=True)
    full_records = full_base_records + local_records
    full_elapsed = time.perf_counter() - full_started

    finalists = sorted(full_records, key=lambda item: item["rank"], reverse=True)[:counts["final"]]
    native_velocity, native_authority = _official_velocity_limits(robot, np.pi)
    final_started = time.perf_counter()
    final_inputs = [{
        "candidate_id": int(record["candidate_id"]), "mount": record["mount"]}
        for record in finalists]
    common_signature = stage_input_fingerprint(
        robot_fingerprint=robot_fingerprint, stage="final_common",
        candidates=final_inputs)
    native_signature = stage_input_fingerprint(
        robot_fingerprint=robot_fingerprint, stage="final_native",
        candidates={"mounts": final_inputs, "velocity": native_velocity,
                    "authority": native_authority})
    common_path = stage_dir / "final_common.partial.json"
    native_path = stage_dir / "final_native.partial.json"
    restored_common = {
        int(record["candidate_id"]): _restore_strict_record(record)
        for record in load_stage_records(common_path, common_signature)}
    restored_native = {
        int(record["candidate_id"]): _restore_strict_record(record)
        for record in load_stage_records(native_path, native_signature)}
    final_common, final_native = [], []
    for position, record in enumerate(finalists, start=1):
        candidate_id = int(record["candidate_id"])
        active = np.asarray(record["mount"])[[0, 1, 2, 4]]
        common_record = restored_common.get(candidate_id)
        if common_record is None:
            common_record = _evaluate_one_mount(
                robot=robot, candidate_id=candidate_id,
                active_mount=active, targets=targets, quaternions=quaternions,
                time_s=time_s, scope="full_episode", smoke=args.smoke,
                velocity_limit=common_velocity)
        final_common.append(common_record)
        _save_stage_records(
            common_path, common_signature, final_common,
            complete=position == len(finalists))
        native_record = restored_native.get(candidate_id)
        if native_record is None:
            native_record = _evaluate_one_mount(
                robot=robot, candidate_id=candidate_id,
                active_mount=active, targets=targets, quaternions=quaternions,
                time_s=time_s, scope="full_episode", smoke=args.smoke,
                velocity_limit=native_velocity)
        final_native.append(native_record)
        _save_stage_records(
            native_path, native_signature, final_native,
            complete=position == len(finalists))
        print(json.dumps({
            "robot": robot, "stage": "final_common_native", "done": position,
            "total": len(finalists), "candidate_id": candidate_id,
            "common_coverage": common_record["frame_coverage"],
            "native_coverage": native_record["frame_coverage"],
        }), flush=True)
    final_elapsed = time.perf_counter() - final_started
    winner = max(final_common, key=lambda item: item["rank"])
    native_winner = max(final_native, key=lambda item: item["rank"])
    cache = output / "cache" / robot / "pick-right-left__teleop_20260805_154215.npz"
    _write_cache(path=cache, robot=robot, record=winner,
                 targets=targets, quaternions=quaternions, time_s=time_s)
    metrics = cross_fidelity_metrics(
        np.asarray([row["candidate_id"] for row in proxy_rows]), proxy_scores,
        np.asarray([record["candidate_id"] for record in window_records]),
        [record["rank"] for record in window_records],
        top_k=min(10, len(window_records)))
    metrics.update(full_episode_optimism_metrics(
        np.asarray([row["candidate_id"] for row in proxy_rows]),
        np.asarray([float(row["episode_success_rate"]) > 0.5
                    for row in proxy_rows]),
        np.asarray([record["candidate_id"] for record in full_base_records]),
        np.asarray([bool(record["episode_success"])
                    for record in full_base_records])))
    result = {
        "robot": robot,
        "fingerprint": robot_fingerprint,
        "velocity_policies": {
            "common_limit_rad_s": common_velocity,
            "native_limit_rad_s": native_velocity,
            "native_authority": native_authority,
        },
        "cross_fidelity": metrics,
        "stage_counts": {
            "proxy": len(proxy_rows), "window": len(window_records),
            "full_and_local": len(full_records), "final": len(final_common)},
        "stage_elapsed_s": {
            "strict_window": window_elapsed,
            "full_and_local": full_elapsed,
            "final_common_and_native": final_elapsed,
        },
        "window_records": [_summary_record(item) for item in window_records],
        "full_records": [_summary_record(item) for item in full_records],
        "final_common": [_summary_record(item) for item in final_common],
        "final_native": [_summary_record(item) for item in final_native],
        "winner": _summary_record(winner),
        "native_winner": _summary_record(native_winner),
        "cache": str(cache),
        "elapsed_s": time.perf_counter() - started,
    }
    _atomic_json(output / "robots" / f"{robot}.json", result)
    return result


def main() -> None:
    args = build_parser().parse_args()
    from design_optimization.best_first_mount_search import representative_frame_windows
    from scripts.solve_strict_urdf_task_cache import target_poses_for

    counts = stage_counts(args)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    episode = _selected_episode()
    spec = {
        "robots": args.robots, "episode": episode, "stage_counts": counts,
        "mount_dofs": "xyz_yaw_tilt0_roll0", "smoke": args.smoke}
    _atomic_json(output / "spec.json", spec)
    coarse_started = time.perf_counter()
    coarse_path = _run_coarse(args, counts, episode, output)
    spec["coarse_elapsed_s_this_run"] = time.perf_counter() - coarse_started
    _atomic_json(output / "spec.json", spec)
    coarse_payload = json.loads(coarse_path.read_text(encoding="utf-8"))
    targets, quaternions, time_s, _ = target_poses_for(
        "local", "pick-right-left", split="validation",
        episode_artifact=episode["artifact"])
    windows = representative_frame_windows(
        targets, quaternions, requested_windows=2 if args.smoke else 4,
        window_length=4 if args.smoke else 8)
    results = []
    from scripts.strict_urdf_model_audit import MODELS
    for robot in args.robots:
        args.current_robot = robot
        entry = MODELS[robot]
        expected_fingerprint = pilot_fingerprint(
            robot=robot, episode_sha256=episode["content_sha256"],
            model_sha256=official_model_contract(robot)["contract_sha256"],
            tcp_translation=entry.tool_translation_m, counts=counts)
        result = load_completed_robot_result(
            output, robot, expected_fingerprint)
        if result is None:
            result = _run_robot(
                args=args, counts=counts, episode=episode,
                coarse_payload=coarse_payload, targets=targets,
                quaternions=quaternions, time_s=time_s, windows=windows,
                output=output)
        else:
            print(json.dumps({"status": "resumed", "robot": robot}, ensure_ascii=False))
        results.append(result)
        video = _render_video(
            robot=robot, cache=Path(results[-1]["cache"]), output=output)
        results[-1]["video"] = str(video)
        _atomic_json(output / "robots" / f"{robot}.json", results[-1])
        _atomic_json(output / "matrix.partial.json", results)
    _atomic_json(output / "matrix.json", results)
    _atomic_json(output / "metrics.json", {
        "robots": {
            row["robot"]: {
                "cross_fidelity": row["cross_fidelity"],
                "stage_counts": row["stage_counts"],
                "stage_elapsed_s": row.get("stage_elapsed_s", {}),
                "winner": row["winner"],
            }
            for row in results
        },
        "coarse_elapsed_s_this_run": spec["coarse_elapsed_s_this_run"],
    })
    from scripts.build_mount_ik_fidelity_pilot_report import write_report
    report = write_report(
        output / "matrix.json", output, expected_robots=args.robots,
        verify_video_decode=True)
    if not report["complete"]:
        raise RuntimeError(
            "pilot artifacts failed completeness validation: "
            + "; ".join(report["completeness_errors"]))
    print(json.dumps(_jsonable({
        "status": "complete", "output": output,
        "report": str(output / "report.md"),
        "robots": [{"robot": row["robot"], "winner": row["winner"]}
                   for row in results]}), ensure_ascii=False))


if __name__ == "__main__":
    main()
