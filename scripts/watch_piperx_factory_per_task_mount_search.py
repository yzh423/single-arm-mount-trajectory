"""Supervise the resumable factory mount batch until completion."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/factory_bimanual/piperx_factory_per_task_mount_search"


def status_payload(restart_count, max_restarts, exit_code, batch_status):
    return {
        "schema": "piperx-factory-search-watchdog-v1",
        "restart_count": int(restart_count),
        "max_restarts": int(max_restarts),
        "last_exit_code": None if exit_code is None else int(exit_code),
        "batch_status": str(batch_status),
    }


def should_restart(*, exit_code, batch_status, restart_count, max_restarts):
    return (batch_status != "complete"
            and int(restart_count) < int(max_restarts))


def _batch_status():
    path = OUTPUT / "batch_status.json"
    if not path.exists():
        return "missing"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get(
            "status", "unknown"))
    except (OSError, json.JSONDecodeError):
        return "unreadable"


def _write_status(payload):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "watchdog_status.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(max_restarts=8, retry_delay_s=5):
    restart_count = 0
    while True:
        batch_status = _batch_status()
        if batch_status == "complete":
            _write_status(status_payload(
                restart_count, max_restarts, 0, batch_status))
            return 0
        command = [sys.executable, "-u", "-m",
                   "scripts.run_piperx_factory_per_task_mount_search", "--full"]
        log = OUTPUT / f"watchdog_attempt_{restart_count:02d}.log"
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"START attempt={restart_count}\n")
            stream.flush()
            process = subprocess.Popen(
                command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
            exit_code = process.wait()
            stream.write(f"EXIT code={exit_code}\n")
        batch_status = _batch_status()
        _write_status(status_payload(
            restart_count, max_restarts, exit_code, batch_status))
        if not should_restart(
                exit_code=exit_code, batch_status=batch_status,
                restart_count=restart_count, max_restarts=max_restarts):
            return int(exit_code) if exit_code else 1
        restart_count += 1
        time.sleep(float(retry_delay_s))


if __name__ == "__main__":
    raise SystemExit(main())
