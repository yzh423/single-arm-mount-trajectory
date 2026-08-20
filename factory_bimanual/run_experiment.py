"""Resumable orchestration for the isolated factory-bimanual experiment matrix."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from .robot_contracts import ROBOT_CONTRACTS


ROBOTS = ("xarm6", "franka_panda", "i2rt_yam", "ur5")
TASKS = ("screw_cap", "pour_raw_material")
IK_MODES = ("strict_a", "easyik")


@dataclass(frozen=True)
class ExperimentConfig:
    output_root: Path
    project_root: Path | None = None
    dry_run: bool = False
    short_prefix_rows: int | None = None
    render_videos: bool = False
    schema_version: int = 1

    def __post_init__(self):
        root = Path(self.output_root).resolve()
        project = Path(self.project_root).resolve() if self.project_root is not None else root.parents[1]
        if root != (project / "reports/factory_bimanual").resolve():
            raise ValueError("output_root must be reports/factory_bimanual inside project_root")
        if self.short_prefix_rows is not None and self.short_prefix_rows < 1:
            raise ValueError("short_prefix_rows must be positive")
        object.__setattr__(self, "output_root", root)
        object.__setattr__(self, "project_root", project)


@dataclass(frozen=True)
class ExperimentJob:
    stage: str
    robot: str
    task: str
    mode: str
    short_prefix_rows: int | None
    fingerprint: str
    source_fingerprint: str | None = None


Executor = Callable[[ExperimentJob], Mapping[str, Any] | None]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


class FactoryBimanualExperiment:
    def __init__(self, config: ExperimentConfig, *, ik_executor: Executor | None = None,
                 mpc_executor: Executor | None = None, video_executor: Executor | None = None):
        self.config = config
        self.ik_executor = ik_executor
        self.mpc_executor = mpc_executor
        self.video_executor = video_executor
        self.manifest_path = config.output_root / "experiment_manifest.json"

    def _fingerprint(self, stage: str, robot: str, task: str, mode: str,
                     source_fingerprint: str | None = None) -> str:
        payload = {"schema_version": self.config.schema_version, "stage": stage,
                   "robot": robot, "task": task, "mode": mode,
                   "short_prefix_rows": self.config.short_prefix_rows}
        payload["source_fingerprint"] = source_fingerprint
        contract = ROBOT_CONTRACTS.get(robot)
        if contract is not None:
            payload["model"] = {
                "source_sha256": hashlib.sha256(
                    contract.source_urdf.read_bytes()).hexdigest(),
                "joints": list(contract.arm_joint_names),
                "base_frame": contract.base_link_name,
                "tcp_frame": contract.tcp_link_name,
                "tcp_offset_m": list(contract.tcp_offset_m),
            }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _job(self, stage: str, robot: str, task: str, mode: str,
             source_fingerprint: str | None = None) -> ExperimentJob:
        return ExperimentJob(stage, robot, task, mode, self.config.short_prefix_rows,
                             self._fingerprint(stage, robot, task, mode, source_fingerprint),
                             source_fingerprint)

    def _load_completed(self) -> dict[str, dict[str, Any]]:
        if not self.manifest_path.exists():
            return {}
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {row["fingerprint"]: row for row in payload.get("jobs", [])
                if row.get("status") == "completed"}

    def _record(self, job: ExperimentJob, status: str, result: Mapping[str, Any] | None = None,
                error: str | None = None) -> dict[str, Any]:
        row = asdict(job) | {"status": status}
        if result is not None:
            row["result"] = dict(result)
        if error is not None:
            row["error"] = error
        # Per-job metadata makes interrupted matrices inspectable independently.
        _atomic_json(self.config.output_root / "jobs" / f"{job.fingerprint}.json", row)
        return row

    def _execute(self, job: ExperimentJob, executor: Executor | None,
                 completed: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        if job.fingerprint in completed:
            return completed[job.fingerprint]
        if self.config.dry_run:
            return self._record(job, "planned")
        if executor is None:
            return self._record(job, "failed", error="executor_not_configured")
        try:
            return self._record(job, "completed", executor(job) or {})
        except Exception as exc:  # one failed job must not abort the matrix
            return self._record(job, "failed", error=f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _mpc_modes(a: dict[str, Any], easy: dict[str, Any]) -> tuple[str, ...]:
        a_valid = a.get("status") == "completed" and a.get("result", {}).get("initializer_valid", True)
        e_valid = easy.get("status") == "completed" and easy.get("result", {}).get("initializer_valid", True)
        if not a_valid and not e_valid:
            return ()
        if not a_valid:
            return ("mpc_easyik",)
        if not e_valid:
            return ("mpc_a",)
        evidence = easy.get("result", {})
        left = float(evidence.get("left_delta_deg", 999.0))
        right = float(evidence.get("right_delta_deg", 999.0))
        return ("canonical_a",) if left < 5.0 and right < 5.0 else ("mpc_a", "mpc_easyik")

    def run(self) -> dict[str, Any]:
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        completed = self._load_completed()
        rows: list[dict[str, Any]] = []
        pair_results: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        for robot in ROBOTS:
            for task in TASKS:
                pair_results[(robot, task)] = {}
                for mode in IK_MODES:
                    row = self._execute(self._job("ik", robot, task, mode), self.ik_executor, completed)
                    rows.append(row)
                    pair_results[(robot, task)][mode] = row
        if not self.config.dry_run:
            for robot in ROBOTS:
                for task in TASKS:
                    pair = pair_results[(robot, task)]
                    for mode in self._mpc_modes(pair["strict_a"], pair["easyik"]):
                        rows.append(self._execute(self._job("mpc", robot, task, mode),
                                                  self.mpc_executor, completed))
        # Rendering is a separately fingerprinted post-job. It is therefore
        # independently resumable and cannot turn a successful solver result
        # into a failed compute result.
        if self.config.render_videos and not self.config.dry_run:
            completed_compute = [row for row in rows
                                 if row["stage"] in {"ik", "mpc"} and row["status"] == "completed"]
            for source in completed_compute:
                render_mode = f"{source['stage']}__{source['mode']}"
                render_job = self._job("render", source["robot"], source["task"], render_mode,
                                       source["fingerprint"])
                rows.append(self._execute(render_job, self.video_executor, completed))
        manifest = {"schema_version": self.config.schema_version,
                    "dry_run": self.config.dry_run,
                    "render_videos": self.config.render_videos,
                    "short_prefix_rows": self.config.short_prefix_rows,
                    "robots": list(ROBOTS), "tasks": list(TASKS), "jobs": rows}
        _atomic_json(self.manifest_path, manifest)
        return manifest
