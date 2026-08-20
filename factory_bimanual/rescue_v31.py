"""Pure branch-continuity and rescue-v3.1 scheduling primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class StrictGate:
    """The non-negotiable pose and branch-continuity ACCEPT gate."""

    position_tolerance_m: float = 0.001
    orientation_tolerance_rad: float = np.deg2rad(0.5)
    branch_guard_rad: float = 0.30

    def __post_init__(self) -> None:
        values = (
            self.position_tolerance_m,
            self.orientation_tolerance_rad,
            self.branch_guard_rad,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("strict gate limits must be positive and finite")

    def accepts(
        self,
        position_error_m: float,
        orientation_error_rad: float,
        joint_delta_rad: np.ndarray,
    ) -> bool:
        delta = np.asarray(joint_delta_rad, dtype=float)
        if delta.ndim != 1:
            raise ValueError("joint delta must be one-dimensional")
        if (
            not np.isfinite(position_error_m)
            or not np.isfinite(orientation_error_rad)
            or np.any(~np.isfinite(delta))
        ):
            return False
        return bool(
            position_error_m <= self.position_tolerance_m
            and orientation_error_rad <= self.orientation_tolerance_rad
            and np.all(np.abs(delta) <= self.branch_guard_rad)
        )

    def pose_accepts(
        self,
        position_error_m: float,
        orientation_error_rad: float,
    ) -> bool:
        return self.accepts(
            position_error_m,
            orientation_error_rad,
            np.zeros(1, dtype=float),
        )


def shortest_joint_delta(
    new: np.ndarray,
    old: np.ndarray,
    periodic: np.ndarray,
) -> np.ndarray:
    """Return ``new-old`` while wrapping only declared periodic joints."""

    new = np.asarray(new, dtype=float)
    old = np.asarray(old, dtype=float)
    periodic = np.asarray(periodic, dtype=bool)
    if new.ndim != 1 or new.shape != old.shape or periodic.shape != new.shape:
        raise ValueError("joint values and periodic mask must have matching 1-D shapes")
    delta = new - old
    delta = delta.copy()
    delta[periodic] = (delta[periodic] + np.pi) % (2.0 * np.pi) - np.pi
    return delta


def wrist_branch_signature(
    q: np.ndarray,
    *,
    deadband_rad: float = 0.10,
) -> tuple[int, int]:
    """Coarsely identify the PiperX J4/J5 basin without zero chatter."""

    q = np.asarray(q, dtype=float)
    if q.shape != (6,) or np.any(~np.isfinite(q)):
        raise ValueError("PiperX wrist signature requires six finite joints")
    if not np.isfinite(deadband_rad) or deadband_rad < 0:
        raise ValueError("wrist deadband must be finite and non-negative")

    def classify(value: float) -> int:
        if abs(value) <= deadband_rad:
            return 0
        return 1 if value > 0 else -1

    return classify(float(q[3])), classify(float(q[4]))


def trapezoidal_transition_time(
    delta_rad: np.ndarray,
    *,
    velocity_rad_s: float = 1.0,
    acceleration_rad_s2: float = 4.0,
    settle_s: float = 0.1,
    decision_s: float = 0.015,
) -> float:
    """Time required by the slowest joint under a symmetric trapezoid."""

    delta = np.asarray(delta_rad, dtype=float)
    if delta.ndim != 1 or np.any(~np.isfinite(delta)):
        raise ValueError("joint delta must be a finite one-dimensional array")
    parameters = (velocity_rad_s, acceleration_rad_s2, settle_s, decision_s)
    if any(not np.isfinite(value) or value < 0 for value in parameters):
        raise ValueError("execution timing parameters must be finite and non-negative")
    if velocity_rad_s <= 0 or acceleration_rad_s2 <= 0:
        raise ValueError("velocity and acceleration must be positive")
    distance = float(np.max(np.abs(delta), initial=0.0))
    switch_distance = velocity_rad_s**2 / acceleration_rad_s2
    if distance <= switch_distance:
        motion_s = 2.0 * np.sqrt(distance / acceleration_rad_s2)
    else:
        motion_s = (
            2.0 * velocity_rad_s / acceleration_rad_s2
            + (distance - switch_distance) / velocity_rad_s
        )
    return float(motion_s + settle_s + decision_s)


def minimum_jerk_transition(
    q0: np.ndarray,
    q1: np.ndarray,
    frames: int,
) -> np.ndarray:
    """Return a quintic smoothstep whose last row is exactly ``q1``."""

    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    if q0.ndim != 1 or q1.shape != q0.shape:
        raise ValueError("transition endpoints must have matching one-dimensional shapes")
    if np.any(~np.isfinite(q0)) or np.any(~np.isfinite(q1)):
        raise ValueError("transition endpoints must be finite")
    frames = int(frames)
    if frames < 2:
        raise ValueError("minimum-jerk transition needs at least two frames")
    unit = np.linspace(0.0, 1.0, frames)
    blend = 10.0 * unit**3 - 15.0 * unit**4 + 6.0 * unit**5
    result = q0[None, :] + blend[:, None] * (q1 - q0)[None, :]
    result[0] = q0
    result[-1] = q1
    return result


@dataclass(frozen=True)
class CandidateFrame:
    """One strict-pose IK branch available at a source frame."""

    q: np.ndarray
    branch_index: int
    position_error_m: float
    orientation_error_rad: float
    wrist_risk: float = 0.0
    joint_limit_margin_rad: float = np.inf
    wrist_signature_value: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=float)
        if q.ndim != 1 or np.any(~np.isfinite(q)):
            raise ValueError("candidate joints must be a finite one-dimensional array")
        object.__setattr__(self, "q", q.copy())
        values = (
            self.position_error_m,
            self.orientation_error_rad,
            self.wrist_risk,
        )
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("candidate errors and wrist risk must be finite and non-negative")
        if np.isnan(self.joint_limit_margin_rad):
            raise ValueError("candidate joint margin may not be NaN")
        if self.wrist_signature_value is not None:
            signature = tuple(int(value) for value in self.wrist_signature_value)
            if not signature or any(value not in (-1, 0, 1) for value in signature):
                raise ValueError("wrist signature values must be -1, 0, or 1")
            object.__setattr__(self, "wrist_signature_value", signature)

    @property
    def wrist_signature(self) -> tuple[int, ...]:
        if self.wrist_signature_value is not None:
            return self.wrist_signature_value
        return wrist_branch_signature(self.q)


@dataclass(frozen=True)
class RescueScheduleConfig:
    gate: StrictGate = StrictGate()
    velocity_rad_s: float = 1.0
    acceleration_rad_s2: float = 4.0
    settle_s: float = 0.1
    decision_s: float = 0.015
    source_rate_hz: float = 60.0
    dwell_frames: int = 18
    lookahead_frames: int = 120
    early_trigger_frames: int = 12

    def __post_init__(self) -> None:
        if self.velocity_rad_s <= 0 or self.acceleration_rad_s2 <= 0:
            raise ValueError("rescue velocity and acceleration must be positive")
        if self.source_rate_hz <= 0:
            raise ValueError("source rate must be positive")
        if self.settle_s < 0 or self.decision_s < 0:
            raise ValueError("rescue fixed times may not be negative")
        if min(self.dwell_frames, self.lookahead_frames, self.early_trigger_frames) < 0:
            raise ValueError("rescue frame budgets may not be negative")


@dataclass(frozen=True)
class RescueEvent:
    mode: str
    trigger: str
    start_frame: int
    end_frame: int
    transition_frames: int
    from_branch_index: int
    to_branch_index: int
    from_wrist_signature: tuple[int, ...]
    to_wrist_signature: tuple[int, ...]
    recovery_q: np.ndarray
    maximum_joint_delta_rad: float
    wrist_delta_norm_rad: float
    predicted_follow_frames: int
    seam_error_rad: float
    cycle_delay_s: float


@dataclass(frozen=True)
class FollowSchedule:
    command_q: np.ndarray
    source_index: np.ndarray
    execution_time_s: np.ndarray
    state: np.ndarray
    accepted: np.ndarray
    source_command_q: np.ndarray
    source_accepted: np.ndarray
    source_state: np.ndarray
    events: tuple[RescueEvent, ...]
    dropped_source_frames: int
    cycle_delay_s: float


StateValid = Callable[[np.ndarray], bool]
TransitionValid = Callable[[np.ndarray, np.ndarray], bool]


def _always_state_valid(q: np.ndarray) -> bool:
    return True


def _always_transition_valid(previous: np.ndarray, current: np.ndarray) -> bool:
    return True


def _pose_valid(candidate: CandidateFrame, gate: StrictGate) -> bool:
    return gate.pose_accepts(
        candidate.position_error_m,
        candidate.orientation_error_rad,
    )


def _candidate_rank(
    candidate: CandidateFrame,
    previous_q: np.ndarray,
    periodic: np.ndarray,
    gate: StrictGate,
) -> tuple:
    delta = shortest_joint_delta(candidate.q, previous_q, periodic)
    normalized_pose = max(
        candidate.position_error_m / gate.position_tolerance_m,
        candidate.orientation_error_rad / gate.orientation_tolerance_rad,
    )
    return (
        float(np.max(np.abs(delta), initial=0.0)),
        float(np.linalg.norm(_wrist_delta_components(delta))),
        candidate.wrist_risk,
        normalized_pose,
        -candidate.joint_limit_margin_rad,
        int(candidate.branch_index),
    )


def _wrist_delta_components(delta: np.ndarray) -> np.ndarray:
    """Return wrist deltas for one arm or concatenated six-DOF arm blocks."""

    delta = np.asarray(delta, dtype=float)
    if len(delta) >= 6 and len(delta) % 6 == 0:
        return np.concatenate([
            delta[start + 3:start + 6]
            for start in range(0, len(delta), 6)
        ])
    return delta[-min(3, len(delta)):]


def _follow_candidates(
    layer: Sequence[CandidateFrame],
    previous: CandidateFrame,
    periodic: np.ndarray,
    config: RescueScheduleConfig,
    state_valid: StateValid,
    transition_valid: TransitionValid,
) -> list[CandidateFrame]:
    result = []
    for candidate in layer:
        delta = shortest_joint_delta(candidate.q, previous.q, periodic)
        if candidate.wrist_signature != previous.wrist_signature:
            continue
        if not config.gate.accepts(
            candidate.position_error_m,
            candidate.orientation_error_rad,
            delta,
        ):
            continue
        if not state_valid(candidate.q) or not transition_valid(previous.q, candidate.q):
            continue
        result.append(candidate)
    return sorted(
        result,
        key=lambda item: _candidate_rank(
            item, previous.q, periodic, config.gate
        ),
    )


def _continuation_length(
    layers: Sequence[Sequence[CandidateFrame]],
    start: int,
    seed: CandidateFrame,
    periodic: np.ndarray,
    config: RescueScheduleConfig,
    state_valid: StateValid,
    transition_valid: TransitionValid,
) -> int:
    count = 1
    previous = seed
    stop = min(len(layers), start + 1 + config.lookahead_frames)
    for index in range(start + 1, stop):
        choices = _follow_candidates(
            layers[index], previous, periodic, config,
            state_valid, transition_valid,
        )
        if not choices:
            break
        previous = choices[0]
        count += 1
    return count


def _safe_transition_path(
    path: np.ndarray,
    state_valid: StateValid,
    transition_valid: TransitionValid,
) -> bool:
    for q in path[1:]:
        if not state_valid(q):
            return False
    return all(
        transition_valid(previous, current)
        for previous, current in zip(path[:-1], path[1:])
    )


def _transition_frames(
    previous_q: np.ndarray,
    target_q: np.ndarray,
    periodic: np.ndarray,
    config: RescueScheduleConfig,
) -> int:
    delta = shortest_joint_delta(target_q, previous_q, periodic)
    seconds = trapezoidal_transition_time(
        delta,
        velocity_rad_s=config.velocity_rad_s,
        acceleration_rad_s2=config.acceleration_rad_s2,
        settle_s=config.settle_s,
        decision_s=config.decision_s,
    )
    return max(1, int(np.ceil(seconds * config.source_rate_hz)))


def schedule_rescue_v31(
    layers: Sequence[Sequence[CandidateFrame]],
    source_time_s: np.ndarray,
    periodic: np.ndarray,
    config: RescueScheduleConfig = RescueScheduleConfig(),
    *,
    state_valid: StateValid = _always_state_valid,
    transition_valid: TransitionValid = _always_transition_valid,
) -> FollowSchedule:
    """Schedule strict FOLLOW/HOLD plus controlled mode-A/mode-B rescue.

    The input layers contain all pose-valid branches discovered offline or by
    the MuJoCo adapter.  FOLLOW may use only the current wrist basin and the
    0.30-rad guard.  Rescue is the sole path allowed to cross that boundary.
    """

    source_time = np.asarray(source_time_s, dtype=float)
    periodic = np.asarray(periodic, dtype=bool)
    if len(layers) != len(source_time) or not len(layers):
        raise ValueError("candidate layers must match a non-empty source timeline")
    if source_time.ndim != 1 or np.any(~np.isfinite(source_time)):
        raise ValueError("source time must be a finite one-dimensional array")
    if len(source_time) > 1 and np.any(np.diff(source_time) <= 0):
        raise ValueError("source time must be strictly increasing")
    first_q = next((candidate.q for layer in layers for candidate in layer), None)
    if first_q is None:
        raise RuntimeError("frame 0 cannot be anchored: no candidates")
    if periodic.shape != np.asarray(first_q).shape:
        raise ValueError("periodic mask must match candidate joint shape")
    for layer in layers:
        for candidate in layer:
            if candidate.q.shape != periodic.shape:
                raise ValueError("all candidate joints must match the periodic mask")

    anchors = [
        candidate for candidate in layers[0]
        if _pose_valid(candidate, config.gate) and state_valid(candidate.q)
    ]
    if not anchors:
        raise RuntimeError("frame 0 cannot be anchored inside the strict pose gate")
    origin = np.zeros_like(periodic, dtype=float)
    anchors.sort(key=lambda item: _candidate_rank(
        item, origin, periodic, config.gate))
    previous = anchors[0]

    command_rows = [previous.q.copy()]
    source_rows = [0]
    states = ["FOLLOW"]
    accepted_rows = [True]
    source_command = np.full((len(layers), len(periodic)), np.nan)
    source_accepted = np.zeros(len(layers), dtype=bool)
    source_state = np.full(len(layers), "HOLD", dtype="U24")
    source_command[0] = previous.q
    source_accepted[0] = True
    source_state[0] = "FOLLOW"
    events: list[RescueEvent] = []
    dropped = 0
    cycle_delay = 0.0
    last_rescue_frame = -10**9

    def append_mode_b(
        source_frame: int,
        target: CandidateFrame,
        predicted: int,
        trigger: str,
    ) -> None:
        nonlocal previous, cycle_delay, last_rescue_frame
        frames = _transition_frames(previous.q, target.q, periodic, config)
        path = minimum_jerk_transition(previous.q, target.q, frames + 1)
        if not _safe_transition_path(path, state_valid, transition_valid):
            raise RuntimeError("selected mode-B rescue transition is not safe")
        for path_index, q in enumerate(path[1:], start=1):
            command_rows.append(q.copy())
            source_rows.append(source_frame)
            at_end = path_index == frames
            states.append("RESCUE_B_END" if at_end else "RESCUE_B")
            accepted_rows.append(at_end)
        delta = shortest_joint_delta(target.q, previous.q, periodic)
        delay = frames / config.source_rate_hz
        events.append(RescueEvent(
            mode="B", trigger=trigger,
            start_frame=source_frame, end_frame=source_frame,
            transition_frames=frames,
            from_branch_index=previous.branch_index,
            to_branch_index=target.branch_index,
            from_wrist_signature=previous.wrist_signature,
            to_wrist_signature=target.wrist_signature,
            recovery_q=target.q.copy(),
            maximum_joint_delta_rad=float(np.max(np.abs(delta), initial=0.0)),
            wrist_delta_norm_rad=float(np.linalg.norm(
                _wrist_delta_components(delta))),
            predicted_follow_frames=predicted,
            seam_error_rad=float(np.max(np.abs(path[-1] - target.q), initial=0.0)),
            cycle_delay_s=delay,
        ))
        cycle_delay += delay
        source_command[source_frame] = target.q
        source_accepted[source_frame] = True
        source_state[source_frame] = "RESCUE_B_END"
        previous = target
        last_rescue_frame = source_frame

    index = 1
    while index < len(layers):
        follow = _follow_candidates(
            layers[index], previous, periodic, config,
            state_valid, transition_valid,
        )
        dwell_ready = index - last_rescue_frame >= config.dwell_frames

        if follow and config.early_trigger_frames > 0 and dwell_ready:
            normal = follow[0]
            normal_run = _continuation_length(
                layers, index, normal, periodic, config,
                state_valid, transition_valid,
            )
            alternatives = []
            for candidate in layers[index]:
                if candidate is normal or not _pose_valid(candidate, config.gate):
                    continue
                if not state_valid(candidate.q):
                    continue
                if candidate.wrist_signature == previous.wrist_signature:
                    continue
                run = _continuation_length(
                    layers, index, candidate, periodic, config,
                    state_valid, transition_valid,
                )
                if run > normal_run:
                    alternatives.append((candidate, run))
            if normal_run <= config.early_trigger_frames and alternatives:
                alternatives.sort(key=lambda item: (
                    -item[1],
                    _transition_frames(previous.q, item[0].q, periodic, config),
                    _candidate_rank(item[0], previous.q, periodic, config.gate),
                ))
                target, predicted = alternatives[0]
                append_mode_b(index, target, predicted, "early_intercept")
                index += 1
                continue

        if follow:
            selected = follow[0]
            command_rows.append(selected.q.copy())
            source_rows.append(index)
            states.append("FOLLOW")
            accepted_rows.append(True)
            source_command[index] = selected.q
            source_accepted[index] = True
            source_state[index] = "FOLLOW"
            previous = selected
            index += 1
            continue

        if dwell_ready:
            mode_a = []
            stop = min(len(layers), index + 1 + config.lookahead_frames)
            for recovery_frame in range(index + 1, stop):
                available_frames = recovery_frame - index
                for candidate in layers[recovery_frame]:
                    if not _pose_valid(candidate, config.gate):
                        continue
                    required = _transition_frames(
                        previous.q, candidate.q, periodic, config)
                    if available_frames < required:
                        continue
                    path = minimum_jerk_transition(
                        previous.q, candidate.q, available_frames + 2)
                    if not _safe_transition_path(path, state_valid, transition_valid):
                        continue
                    predicted = _continuation_length(
                        layers, recovery_frame, candidate, periodic, config,
                        state_valid, transition_valid,
                    )
                    mode_a.append((
                        candidate, recovery_frame, predicted, path,
                    ))
            if mode_a:
                mode_a.sort(key=lambda item: (
                    -item[2], item[1] - index,
                    _candidate_rank(item[0], previous.q, periodic, config.gate),
                ))
                target, recovery_frame, predicted, path = mode_a[0]
                transition_frames = recovery_frame - index
                for offset, q in enumerate(path[1:]):
                    source_frame = index + offset
                    if source_frame > recovery_frame:
                        break
                    at_end = source_frame == recovery_frame
                    command_rows.append(q.copy())
                    source_rows.append(source_frame)
                    states.append("RESCUE_A_END" if at_end else "RESCUE_A")
                    accepted_rows.append(at_end)
                    source_command[source_frame] = q
                    source_accepted[source_frame] = at_end
                    source_state[source_frame] = (
                        "RESCUE_A_END" if at_end else "RESCUE_A"
                    )
                delta = shortest_joint_delta(target.q, previous.q, periodic)
                events.append(RescueEvent(
                    mode="A", trigger="strict_branch_lost",
                    start_frame=index, end_frame=recovery_frame,
                    transition_frames=transition_frames,
                    from_branch_index=previous.branch_index,
                    to_branch_index=target.branch_index,
                    from_wrist_signature=previous.wrist_signature,
                    to_wrist_signature=target.wrist_signature,
                    recovery_q=target.q.copy(),
                    maximum_joint_delta_rad=float(
                        np.max(np.abs(delta), initial=0.0)),
                    wrist_delta_norm_rad=float(np.linalg.norm(
                        _wrist_delta_components(delta))),
                    predicted_follow_frames=predicted,
                    seam_error_rad=float(
                        np.max(np.abs(path[-1] - target.q), initial=0.0)),
                    cycle_delay_s=0.0,
                ))
                dropped += transition_frames
                previous = target
                last_rescue_frame = recovery_frame
                index = recovery_frame + 1
                continue

            mode_b = [
                candidate for candidate in layers[index]
                if _pose_valid(candidate, config.gate)
                and state_valid(candidate.q)
            ]
            safe_mode_b = []
            for candidate in mode_b:
                frames = _transition_frames(
                    previous.q, candidate.q, periodic, config)
                path = minimum_jerk_transition(
                    previous.q, candidate.q, frames + 1)
                if _safe_transition_path(path, state_valid, transition_valid):
                    safe_mode_b.append(candidate)
            if safe_mode_b:
                safe_mode_b.sort(key=lambda item: _candidate_rank(
                    item, previous.q, periodic, config.gate))
                target = safe_mode_b[0]
                predicted = _continuation_length(
                    layers, index, target, periodic, config,
                    state_valid, transition_valid,
                )
                append_mode_b(index, target, predicted, "strict_branch_lost")
                index += 1
                continue

        command_rows.append(previous.q.copy())
        source_rows.append(index)
        states.append("HOLD")
        accepted_rows.append(False)
        source_command[index] = previous.q
        source_state[index] = "HOLD"
        index += 1

    command_q = np.asarray(command_rows, dtype=float)
    source_index = np.asarray(source_rows, dtype=int)
    state = np.asarray(states, dtype="U24")
    accepted = np.asarray(accepted_rows, dtype=bool)
    execution_time = np.arange(len(command_q), dtype=float) / config.source_rate_hz
    return FollowSchedule(
        command_q=command_q,
        source_index=source_index,
        execution_time_s=execution_time,
        state=state,
        accepted=accepted,
        source_command_q=source_command,
        source_accepted=source_accepted,
        source_state=source_state,
        events=tuple(events),
        dropped_source_frames=int(dropped),
        cycle_delay_s=float(cycle_delay),
    )
