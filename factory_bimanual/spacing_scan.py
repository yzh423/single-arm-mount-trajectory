"""Pure, deterministic fixed-spacing candidate generation and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SpacingCandidateRow:
    spacing_m: float
    task_name: str
    synchronous_coverage: float
    longest_failure_s: float
    cross_arm_collision_frames: int
    table_base_collision_frames: int
    aggregate_tcp_error: float
    installation_clearance_m: float | None = None
    elapsed_s: float | None = None


@dataclass(frozen=True)
class SpacingSelection:
    spacing_m: float
    task_names: tuple[str, ...]
    tie_count: int
    objective: tuple[float, float, int, int, float]
    rows: tuple[SpacingCandidateRow, ...]


def _inclusive_grid(start: float, stop: float, step: float) -> list[float]:
    if not all(isfinite(x) for x in (start, stop, step)) or step <= 0 or stop < start:
        raise ValueError("spacing bounds must be finite, ordered, and have positive step")
    count = int((stop - start) // step)
    values = [start + index * step for index in range(count + 1)]
    if not values or values[-1] < stop - 1e-12:
        values.append(stop)
    return values


def generate_spacing_candidates(
    minimum_m: float,
    maximum_m: float,
    coarse_step_m: float,
    *,
    finalists: Sequence[float] = (),
    local_step_m: float | None = None,
) -> tuple[float, ...]:
    """Return a stable union of a coarse grid and local neighbours of finalists."""
    values = _inclusive_grid(minimum_m, maximum_m, coarse_step_m)
    if finalists:
        if local_step_m is None or local_step_m <= 0 or not isfinite(local_step_m):
            raise ValueError("a positive finite local_step_m is required for finalists")
        for center in finalists:
            if not minimum_m <= center <= maximum_m:
                raise ValueError("finalist lies outside spacing bounds")
            values.extend((center - local_step_m, center, center + local_step_m))
    bounded = (x for x in values if minimum_m - 1e-12 <= x <= maximum_m + 1e-12)
    return tuple(sorted({round(x, 12) for x in bounded}))


def _objective(rows: Sequence[SpacingCandidateRow]) -> tuple[float, float, int, int, float]:
    # Values are oriented so lexicographic max implements the approved objective.
    return (
        min(row.synchronous_coverage for row in rows),
        -max(row.longest_failure_s for row in rows),
        -sum(row.cross_arm_collision_frames for row in rows),
        -sum(row.table_base_collision_frames for row in rows),
        -sum(row.aggregate_tcp_error for row in rows),
    )


def rank_spacing_candidates(rows: Iterable[SpacingCandidateRow]) -> SpacingSelection:
    """Select one spacing using all tasks and the approved lexicographic objective."""
    grouped: dict[float, list[SpacingCandidateRow]] = {}
    expected_tasks: set[str] | None = None
    for row in rows:
        numeric = (row.spacing_m, row.synchronous_coverage, row.longest_failure_s,
                   row.aggregate_tcp_error)
        if not row.task_name or not all(isfinite(x) for x in numeric):
            raise ValueError("candidate rows require a task and finite metrics")
        if not 0 <= row.synchronous_coverage <= 1 or row.longest_failure_s < 0:
            raise ValueError("invalid coverage or failure duration")
        if row.cross_arm_collision_frames < 0 or row.table_base_collision_frames < 0:
            raise ValueError("collision counts cannot be negative")
        grouped.setdefault(row.spacing_m, []).append(row)
    if not grouped:
        raise ValueError("at least one spacing candidate is required")
    for spacing_rows in grouped.values():
        names = [row.task_name for row in spacing_rows]
        if len(names) != len(set(names)):
            raise ValueError("each spacing must contain one row per task")
        tasks = set(names)
        expected_tasks = tasks if expected_tasks is None else expected_tasks
        if tasks != expected_tasks:
            raise ValueError("all spacings must be evaluated on the same tasks")

    scored = {spacing: _objective(spacing_rows) for spacing, spacing_rows in grouped.items()}
    best_objective = max(scored.values())
    tied = sorted(spacing for spacing, objective in scored.items() if objective == best_objective)
    selected = tied[0]
    return SpacingSelection(
        spacing_m=selected,
        task_names=tuple(sorted(expected_tasks or ())),
        tie_count=len(tied),
        objective=best_objective,
        rows=tuple(sorted(grouped[selected], key=lambda row: row.task_name)),
    )
