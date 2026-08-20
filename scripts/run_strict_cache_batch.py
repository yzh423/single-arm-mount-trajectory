"""Resumable subprocess queue for strict real-model trajectory caches."""
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
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    manifest_path = ROOT / "videos/single_arm/strict_video_job_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = [row for row in manifest["jobs"] if row["kind"] == "single_arm_task" and row["domain"] == args.domain]
    summary = {"domain": args.domain, "total": len(jobs), "pass": 0, "fail": 0, "skipped": 0, "failures": []}
    solver = ROOT / "scripts/solve_strict_urdf_task_cache.py"
    for index, job in enumerate(jobs, 1):
        audit = ROOT / "videos/single_arm/strict_cache" / args.domain / job["robot"] / f"{job['task']}.json"
        if audit.is_file():
            previous = json.loads(audit.read_text(encoding="utf-8"))
            if previous.get("status") == "pass" or not args.retry_failed:
                summary["skipped"] += 1
                summary[previous.get("status", "fail")] += 1
                print(f"[{index}/{len(jobs)}] skip {job['robot']} {job['task']} {previous.get('status')}", flush=True)
                continue
        command = [sys.executable, str(solver), "--domain", args.domain, "--robot", job["robot"], "--task", job["task"]]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if audit.is_file():
            row = json.loads(audit.read_text(encoding="utf-8"))
            status = row.get("status", "fail")
        else:
            status = "fail"
        summary[status] += 1
        if status != "pass":
            summary["failures"].append({"key": job["key"], "returncode": completed.returncode, "stderr": completed.stderr[-1200:]})
        print(f"[{index}/{len(jobs)}] {job['robot']} {job['task']} {status}", flush=True)
        checkpoint = ROOT / "reports/single_arm" / f"strict_cache_batch_{args.domain}.json"
        checkpoint.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
