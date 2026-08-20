from __future__ import annotations

import re
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def last_matching(path: Path, pattern: re.Pattern[str]) -> str:
    if not path.exists():
        return "not started"
    matches = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
               if pattern.search(line)]
    return matches[-1] if matches else ("complete" if path.stat().st_size else "starting")


def main() -> None:
    frame_pattern = re.compile(r"\]\s+\d+/\d+ frames")
    for robot in ("doosan", "xarm6", "ur5", "kinova"):
        path = ROOT / "runs" / f"continuous_retimed_{robot}.out.log"
        result = ROOT / "reports/learning/continuous_retimed" / f"{robot}_test_episode_563.json"
        status = "complete" if result.exists() else last_matching(path, frame_pattern)
        print(f"continuous {robot:7s}: {status}")
    formal = ROOT / "reports/learning/hourly/1100_formal_test_summary.csv"
    print(f"formal held-out summary: {'complete' if formal.exists() else 'pending'}")
    for robot in ("ur5", "kinova"):
        run = ROOT / "runs/collision_formal_expanded" / robot
        final = run / "pareto_finalists.json"; history = run / "history.jsonl"
        if final.exists():
            print(f"expanded collision {robot:7s}: complete")
        elif history.exists():
            row = json.loads(history.read_text().splitlines()[-1])
            generation = int(row["generation"]) + 1; total = 64
            elapsed = float(row["elapsed_s"])
            eta = elapsed / max(generation, 1) * (total - generation)
            width = 20; filled = round(width * generation / total)
            bar = "#" * filled + "-" * (width - filled)
            print(f"expanded collision {robot:7s}: [{bar}] {generation}/{total}, "
                  f"ETA {eta/60:.1f} min")
        else:
            print(f"expanded collision {robot:7s}: pending")


if __name__ == "__main__":
    main()
