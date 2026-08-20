"""Incremental frame-result cache and deterministic IK seed selection."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FrameResult:
    q: np.ndarray
    success: bool
    position_error_m: float
    orientation_error_rad: float
    table_collision: bool
    self_collision: bool
    restarts: int
    iterations: int


@dataclass
class CandidateEvaluation:
    candidate_id: int
    frames: dict[int, FrameResult] = field(default_factory=dict)
    new_frames: int = 0
    reused_frames: int = 0


Solver = Callable[[int, np.ndarray | None, int], Mapping[str, object]]


def _nearest_success(frames: Mapping[int, FrameResult], frame_index: int) -> FrameResult | None:
    successful = [index for index, result in frames.items() if result.success]
    if not successful:
        return None
    nearest = min(successful, key=lambda index: (abs(index - frame_index), index))
    return frames[nearest]


class IncrementalMountEvaluator:
    def __init__(self, *, frame_count: int) -> None:
        if frame_count < 1:
            raise ValueError("frame count must be positive")
        self.frame_count = frame_count
        self._evaluations: dict[int, CandidateEvaluation] = {}

    def evaluation(self, candidate_id: int) -> CandidateEvaluation | None:
        return self._evaluations.get(candidate_id)

    def _seed_for(
        self,
        *,
        current: CandidateEvaluation,
        parent: CandidateEvaluation | None,
        frame_index: int,
    ) -> np.ndarray | None:
        if parent is not None:
            same_frame = parent.frames.get(frame_index)
            if same_frame is not None and same_frame.success:
                return same_frame.q.copy()
        nearest = _nearest_success(current.frames, frame_index)
        if nearest is not None:
            return nearest.q.copy()
        if parent is not None:
            nearest = _nearest_success(parent.frames, frame_index)
            if nearest is not None:
                return nearest.q.copy()
        return None

    def evaluate(
        self,
        *,
        candidate_id: int,
        frame_indices: np.ndarray,
        solver: Solver,
        restart_limit: int,
        parent_id: int | None = None,
    ) -> CandidateEvaluation:
        indices = np.asarray(frame_indices, dtype=int)
        if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
            raise ValueError("frame indices must be a unique one-dimensional array")
        if np.any(indices < 0) or np.any(indices >= self.frame_count):
            raise ValueError("requested frame is outside the trajectory")
        if restart_limit < 0:
            raise ValueError("restart limit must be non-negative")
        current = self._evaluations.setdefault(candidate_id, CandidateEvaluation(candidate_id))
        parent = self._evaluations.get(parent_id) if parent_id is not None else None
        reused = sum(int(index) in current.frames for index in indices)
        new = 0
        for raw_index in indices:
            frame_index = int(raw_index)
            if frame_index in current.frames:
                continue
            seed = self._seed_for(current=current, parent=parent, frame_index=frame_index)
            payload = solver(frame_index, seed, restart_limit)
            current.frames[frame_index] = FrameResult(
                q=np.asarray(payload["q"], dtype=float).copy(),
                success=bool(payload["success"]),
                position_error_m=float(payload["position_error_m"]),
                orientation_error_rad=float(payload["orientation_error_rad"]),
                table_collision=bool(payload["table_collision"]),
                self_collision=bool(payload["self_collision"]),
                restarts=int(payload["restarts"]),
                iterations=int(payload["iterations"]),
            )
            new += 1
        return CandidateEvaluation(
            candidate_id=candidate_id,
            frames=dict(current.frames),
            new_frames=new,
            reused_frames=reused,
        )
