"""Render every successful or failed cache from the fixed 4096 experiment."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/single_arm/ten_arm_two_single_tasks_handbook_fixed_4096"


def main() -> None:
    matrix = OUT / "matrix.json"
    rows = json.loads(matrix.read_text(encoding="utf-8"))
    failures = []
    for row in rows:
        destination = OUT / "videos" / row["robot"] / f'{row["task"]}.mp4'
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([
            sys.executable, "scripts/render_strict_single_arm_task.py",
            "--domain", "local", "--robot", row["robot"],
            "--task", row["task"], "--output", str(destination),
        ], cwd=ROOT)
        if result.returncode:
            failures.append({"robot": row["robot"], "task": row["task"],
                             "returncode": result.returncode})
    (OUT / "all_video_render_status.json").write_text(json.dumps({
        "expected": len(rows),
        "rendered": len(list((OUT / "videos").rglob("*.mp4"))),
        "failures": failures,
    }, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
