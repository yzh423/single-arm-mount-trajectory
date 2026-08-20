"""Low-overhead aggregate telemetry for strict mount search."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Iterator, Literal


def _restart_histogram() -> dict[str, int]:
    return {"0": 0, "1-2": 0, "3-5": 0, "6-10": 0,
            "11-20": 0, "21-40": 0, "failed": 0}


@dataclass
class StageTelemetry:
    elapsed_s: float = 0.0
    evaluated_candidates: int = 0
    promoted_candidates: int = 0
    rejected_candidates: int = 0
    new_frames: int = 0
    reused_frames: int = 0
    solve_pose_calls: int = 0
    dls_iterations: int = 0
    first_frame_restart_histogram: dict[str, int] = field(default_factory=_restart_histogram)


class SearchTelemetry:
    def __init__(self) -> None:
        self._stages: dict[str, StageTelemetry] = {}

    def _get(self, name: str) -> StageTelemetry:
        if not name:
            raise ValueError("stage name must be non-empty")
        return self._stages.setdefault(name, StageTelemetry())

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self._get(name).elapsed_s += time.perf_counter() - started

    def record_candidate(
        self,
        stage: str,
        *,
        outcome: Literal["evaluated", "promoted", "rejected"],
        new_frames: int,
        reused_frames: int,
    ) -> None:
        if new_frames < 0 or reused_frames < 0:
            raise ValueError("frame counters must be non-negative")
        row = self._get(stage)
        if outcome == "evaluated":
            row.evaluated_candidates += 1
        elif outcome == "promoted":
            row.promoted_candidates += 1
        elif outcome == "rejected":
            row.rejected_candidates += 1
        else:
            raise ValueError(f"unknown candidate outcome: {outcome}")
        row.new_frames += new_frames
        row.reused_frames += reused_frames

    def record_ik(
        self,
        stage: str,
        *,
        first_frame: bool,
        restarts: int,
        iterations: int,
        success: bool,
    ) -> None:
        if restarts < 0 or iterations < 0:
            raise ValueError("IK counters must be non-negative")
        row = self._get(stage)
        row.solve_pose_calls += 1
        row.dls_iterations += iterations
        if not first_frame:
            return
        if not success:
            bucket = "failed"
        elif restarts == 0:
            bucket = "0"
        elif restarts <= 2:
            bucket = "1-2"
        elif restarts <= 5:
            bucket = "3-5"
        elif restarts <= 10:
            bucket = "6-10"
        elif restarts <= 20:
            bucket = "11-20"
        else:
            bucket = "21-40"
        row.first_frame_restart_histogram[bucket] += 1

    def to_dict(self) -> dict[str, object]:
        return {"stages": {name: asdict(row) for name, row in self._stages.items()}}
