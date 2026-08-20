"""Build the deterministic three-domain strict-URDF video job matrix."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROBOTS = (
    "doosan", "xarm6", "ur5", "kinova_gen3_lite", "arx_x5", "big_yam",
    "franka_panda", "franka_panda_locked_j3", "i2rt_yam", "nero",
    "openarm", "piperx", "willow",
)
SOURCES = {
    "local": ROOT / "reports/single_arm/dense_search_results.json",
    "droid": ROOT / "reports/single_arm/domains/droid/dense_results.json",
    "egodex": ROOT / "reports/single_arm/domains/egodex/dense_results.json",
}


def task_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    robots = payload["robots"]
    missing = sorted(set(ROBOTS) - set(robots))
    if missing:
        raise RuntimeError(f"{path}: missing robots {missing}")
    reference = sorted(robots[ROBOTS[0]]["per_task"])
    for robot in ROBOTS[1:]:
        current = sorted(robots[robot]["per_task"])
        if current != reference:
            raise RuntimeError(f"{path}: task mismatch for {robot}")
    return reference


def main() -> None:
    jobs: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for domain, source in SOURCES.items():
        tasks = task_names(source)
        counts[domain] = len(tasks)
        for robot in ROBOTS:
            for task in tasks:
                relative = Path("videos/single_arm/strict_per_arm_task") / domain / robot / f"{task}.mp4"
                key = f"{domain}:{robot}:{task}"
                jobs.append({
                    "key": key,
                    "kind": "single_arm_task",
                    "domain": domain,
                    "robot": robot,
                    "task": task,
                    "optimization_source": str(source.relative_to(ROOT)).replace("\\", "/"),
                    "output": str(relative).replace("\\", "/"),
                    "status": "pending_model_qualification",
                })
        jobs.append({
            "key": f"{domain}:comparison",
            "kind": "thirteen_arm_comparison",
            "domain": domain,
            "robots": list(ROBOTS),
            "task_selection": "representative_after_strict_cache",
            "output": f"videos/single_arm/strict_comparison/thirteen_arm_{domain}_30s.mp4",
            "status": "pending_model_qualification",
        })
    keys = [row["key"] for row in jobs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate video job keys")
    expected = {"local": 25, "droid": 23, "egodex": 100}
    if counts != expected:
        raise RuntimeError(f"unexpected task counts: {counts}")
    payload = {
        "schema_version": 1,
        "renderer_contract": "real URDF/MJCF, finite table, matching IK/FK tracking_tcp",
        "robots": list(ROBOTS),
        "task_counts": counts,
        "single_arm_task_video_count": sum(value * len(ROBOTS) for value in counts.values()),
        "comparison_video_count": len(SOURCES),
        "total_video_count": len(jobs),
        "old_schematic_videos_accepted": False,
        "jobs": jobs,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    output = ROOT / "videos/single_arm/strict_video_job_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "task_counts", "single_arm_task_video_count", "comparison_video_count",
        "total_video_count", "manifest_sha256")}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
