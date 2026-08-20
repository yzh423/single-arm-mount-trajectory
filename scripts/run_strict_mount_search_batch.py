"""Resumable mount-refinement queue for strict-cache failures."""
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
    parser.add_argument("--candidates", type=int, default=32)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "videos/single_arm/strict_video_job_manifest.json").read_text(encoding="utf-8"))
    jobs = [row for row in manifest["jobs"] if row["kind"] == "single_arm_task" and row["domain"] == args.domain]
    pending = []
    for job in jobs:
        audit = ROOT / "videos/single_arm/strict_cache" / args.domain / job["robot"] / f"{job['task']}.json"
        if audit.is_file() and json.loads(audit.read_text(encoding="utf-8")).get("status") != "pass":
            pending.append(job)
    summary = {"domain": args.domain, "initial_failures": len(pending), "searched": 0, "rescued": 0, "remaining_fail": 0, "rows": []}
    search = ROOT / "scripts/search_strict_urdf_mount.py"
    for index, job in enumerate(pending, 1):
        command = [sys.executable, str(search), "--domain", args.domain, "--robot", job["robot"], "--task", job["task"], "--candidates", str(args.candidates)]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        audit_path = ROOT / "videos/single_arm/strict_cache" / args.domain / job["robot"] / f"{job['task']}.json"
        row = json.loads(audit_path.read_text(encoding="utf-8"))
        passed = row.get("status") == "pass"
        summary["searched"] += 1
        summary["rescued" if passed else "remaining_fail"] += 1
        summary["rows"].append({"key": job["key"], "status": row.get("status"), "success_rate": row.get("success_rate"), "mean_error_m": row.get("position_error_m", {}).get("mean")})
        print(f"[{index}/{len(pending)}] {job['robot']} {job['task']} {'rescued' if passed else 'failed'}", flush=True)
        (ROOT / "reports/single_arm" / f"strict_mount_search_{args.domain}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
