"""Resumable renderer queue for every accepted strict trajectory cache."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("local", "droid", "egodex"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "videos/single_arm/strict_video_job_manifest.json").read_text(encoding="utf-8"))
    jobs = [j for j in manifest["jobs"] if j["kind"] == "single_arm_task" and j["domain"] == args.domain]
    summary = {"domain": args.domain, "total": len(jobs), "rendered": 0, "skipped": 0, "failed": []}
    renderer = ROOT / "scripts/render_strict_single_arm_task.py"
    checkpoint = ROOT / "reports/single_arm" / f"strict_render_batch_{args.domain}.json"
    for index, job in enumerate(jobs, 1):
        output = ROOT / job["output"]
        cache = ROOT / "videos/single_arm/strict_cache" / args.domain / job["robot"] / f"{job['task']}.npz"
        if output.is_file() and output.with_suffix(".json").is_file() and not args.overwrite:
            summary["skipped"] += 1
            continue
        if not cache.is_file():
            summary["failed"].append({"key": job["key"], "reason": "missing_cache"})
            continue
        completed = subprocess.run([sys.executable, str(renderer), "--domain", args.domain,
                                    "--robot", job["robot"], "--task", job["task"]], cwd=ROOT,
                                   text=True, capture_output=True)
        if completed.returncode:
            summary["failed"].append({"key": job["key"], "returncode": completed.returncode,
                                      "stderr": completed.stderr[-1200:]})
        else:
            summary["rendered"] += 1
        checkpoint.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[{index}/{len(jobs)}] {job['robot']} {job['task']} rc={completed.returncode}", flush=True)
    checkpoint.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
