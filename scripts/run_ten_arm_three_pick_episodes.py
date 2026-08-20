"""Run ten arms on three different tasks with an independent mount per episode."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/single_arm/ten_arm_three_tasks_official_models_4096"
ROBOTS = ("doosan", "xarm6", "ur5", "kinova_gen3_lite", "arx_x5",
          "franka_panda", "franka_panda_locked_j3", "i2rt_yam", "openarm", "piperx")
TASKS = ("cap-left", "open-box-2", "pick-right-left")


def job_fingerprint(robot: str, episode: dict) -> str:
    # This file is normally launched as ``python scripts/<name>.py``.  In that
    # mode Python places ``scripts/`` (not the repository root) on sys.path,
    # so importing it again as a top-level package fails before any strict job
    # is submitted.  Import the sibling module directly, matching the other
    # executable scripts in this repository.
    try:
        from scripts.strict_urdf_model_audit import MODELS
    except ModuleNotFoundError:
        from strict_urdf_model_audit import MODELS
    entry = MODELS[robot]
    model_sha256 = hashlib.sha256(entry.path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps({
        "robot": robot, "episode_id": episode["episode_id"],
        "content": episode["content_sha256"], "model_sha256": model_sha256,
        "model_policy": "pinned_official_vendor_native_v1",
        "tcp_parent": entry.tcp_parent,
        "tcp_authority": entry.tcp_authority,
        "tool_translation_m": entry.tool_translation_m,
        "tool_quaternion_wxyz": entry.tool_quaternion_wxyz,
        "joint_limit_authority": entry.joint_limit_authority,
        "locked_joint_ranges": entry.locked_joint_ranges or {},
        "mount_scope": "per_episode", "mount_dofs": "xyz_yaw_tilt0_roll0",
        "planner": "rolling_multibranch", "collision": "vendor_pair_allowlist_v2",
    }, sort_keys=True).encode()).hexdigest()


def coarse_fingerprint(episode: dict) -> str:
    try:
        from scripts.strict_urdf_model_audit import MODELS
    except ModuleNotFoundError:
        from strict_urdf_model_audit import MODELS
    dependencies = [
        ROOT / "scripts/run_thirteen_arm_dense_search.py",
        ROOT / "reports/single_arm/collision_proxy_profiles.json",
    ]
    model_rows = {}
    for name in ROBOTS:
        entry = MODELS[name]
        model_rows[name] = {
            "sha256": hashlib.sha256(entry.path.read_bytes()).hexdigest(),
            "tcp_parent": entry.tcp_parent,
            "tcp_authority": entry.tcp_authority,
            "tool_translation_m": entry.tool_translation_m,
            "joint_limit_authority": entry.joint_limit_authority,
        }
    payload = {
        "episode_id": episode["episode_id"],
        "content_sha256": episode["content_sha256"],
        "models": model_rows,
        "dependency_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in dependencies
        },
        "candidate_count": 4096,
        "mount_dofs": "xyz_yaw_tilt0_roll0",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def selected_episodes() -> list[dict]:
    manifest = json.loads((ROOT / "data/processed/local_pose_benchmark/manifest.json").read_text(encoding="utf-8"))
    selected = []
    for task in TASKS:
        rows = [row for row in manifest["episodes"] if row["task"] == task
                and row["hand"] in {"left", "right"}
                and row.get("split") == "validation"
                and row.get("trajectory_edge_trim_eligible", True)]
        rows.sort(key=lambda row: row["source_path"])
        if not rows:
            raise RuntimeError(f"no eligible validation episode for {task}")
        selected.append(rows[0])
    return selected


def coarse_result_complete(path: Path, expected_fingerprint: str | None = None) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (set(payload.get("robots", {})) == set(ROBOTS)
            and (expected_fingerprint is None
                 or payload.get("input_fingerprint") == expected_fingerprint))


def run_one(robot: str, episode: dict, incumbent: Path) -> dict:
    task = episode["task"]
    episode_key = f"{task}__{Path(episode['source_path']).stem}"
    cache = OUT / "cache" / robot / f"{episode_key}.npz"
    video = OUT / "videos" / robot / f"{episode_key}.mp4"
    log_dir = OUT / "logs"; cache.parent.mkdir(parents=True, exist_ok=True)
    video.parent.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = job_fingerprint(robot, episode)
    search = [sys.executable, "scripts/search_strict_urdf_mount.py", "--domain", "local",
              "--robot", robot, "--task", task, "--search-policy", "best-first",
              "--episode-artifact", episode["artifact"], "--output", str(cache),
              "--incumbent-json", str(incumbent),
              "--input-fingerprint", fingerprint]
    prefix = log_dir / f"{robot}__{episode_key}"
    with prefix.with_suffix(".search.stdout.log").open("w", encoding="utf-8") as stdout, \
         prefix.with_suffix(".search.stderr.log").open("w", encoding="utf-8") as stderr:
        search_rc = subprocess.run(search, cwd=ROOT, stdout=stdout, stderr=stderr).returncode
    render_rc = None
    if search_rc in (0, 2) and cache.is_file():
        render = [sys.executable, "scripts/render_strict_single_arm_task.py", "--domain", "local",
                  "--robot", robot, "--task", task, "--cache", str(cache),
                  "--output", str(video)]
        with prefix.with_suffix(".render.stdout.log").open("w", encoding="utf-8") as stdout, \
             prefix.with_suffix(".render.stderr.log").open("w", encoding="utf-8") as stderr:
            render_rc = subprocess.run(render, cwd=ROOT, stdout=stdout, stderr=stderr).returncode
    return {"robot": robot, "task": task, "episode_id": episode["episode_id"],
            "episode_key": episode_key, "episode_artifact": episode["artifact"],
            "mount_scope": "robot_x_episode", "search_returncode": search_rc,
            "render_returncode": render_rc, "cache": str(cache), "video": str(video),
            "input_fingerprint": fingerprint}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    episodes = selected_episodes()
    (OUT / "experiment_spec.json").write_text(json.dumps({
        "robots": list(ROBOTS), "tasks": list(TASKS), "episodes": episodes, "jobs": 30,
        "mount_policy": "each robot-episode pair independently searches X/Y/Z/Yaw; Tilt=0 and Roll=0 are fixed",
        "searched_mount_fields": ["X", "Y", "Z", "Yaw"],
        "fixed_mount_fields": {"Tilt_deg": 0.0, "Roll_deg": 0.0},
        "gpu_coarse_candidates_per_episode": 4096,
        "model_policy": "pinned official manufacturer models at vendor-native dimensions",
        "official_model_gate": "reports/single_arm/official_model_provenance_gate.json",
        "baseline_for_report": "reports/single_arm/ten_arm_pick_right_left_three_episodes",
        "edge_trim_s": 0.15,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    coarse = {}
    for episode in episodes:
        task = episode["task"]
        episode_key = f"{task}__{Path(episode['source_path']).stem}"
        coarse_path = OUT / "coarse" / f"{episode_key}.json"
        coarse_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = coarse_fingerprint(episode)
        if coarse_result_complete(coarse_path, fingerprint):
            coarse[episode["episode_id"]] = coarse_path
            print(f"[coarse resume] {episode_key}", flush=True)
            continue
        command = [sys.executable, "scripts/run_thirteen_arm_dense_search.py",
                   "--candidates", "4096", "--candidate-batch", "64",
                   "--robots", *ROBOTS, "--episode-artifact", episode["artifact"],
                   "--task", task, "--output", str(coarse_path),
                   "--input-fingerprint", fingerprint]
        with (OUT / "logs" / f"coarse__{episode_key}.stdout.log").open("w", encoding="utf-8") as stdout, \
             (OUT / "logs" / f"coarse__{episode_key}.stderr.log").open("w", encoding="utf-8") as stderr:
            returncode = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr).returncode
        if returncode != 0 or not coarse_path.is_file():
            raise RuntimeError(f"per-episode GPU coarse search failed: {episode_key} rc={returncode}")
        coarse[episode["episode_id"]] = coarse_path
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, robot, episode, coarse[episode["episode_id"]])
                   for robot in ROBOTS for episode in episodes]
        for future in as_completed(futures):
            completed = future.result()
            rows.append(completed); rows.sort(key=lambda r: (ROBOTS.index(r["robot"]), r["episode_key"]))
            (OUT / "matrix.partial.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[{len(rows)}/30] {completed['robot']} {completed['episode_key']}", flush=True)
    (OUT / "matrix.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [sys.executable, "scripts/build_ten_arm_three_episode_report.py", str(OUT),
              "--baseline", str(ROOT / "reports/single_arm/ten_arm_pick_right_left_three_episodes")]
    report_rc = subprocess.run(report, cwd=ROOT).returncode
    if report_rc != 0:
        raise RuntimeError(f"report generation failed with return code {report_rc}")


if __name__ == "__main__":
    main()
