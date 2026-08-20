"""Collision contracts used by the isolated strict bimanual planner.

The callback implementation keeps MuJoCo ownership and geometry details in the
scene adapter while giving the planner one deterministic collision vocabulary.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

import numpy as np


class CollisionClass(str, Enum):
    SELF = "self_collision"
    CROSS_ARM = "cross_arm_collision"
    TABLE = "table_collision"
    BASE = "base_collision"
    CLEARANCE = "clearance_violation"


@dataclass(frozen=True)
class CollisionReport:
    classes: tuple[CollisionClass, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.classes


PairState = tuple[np.ndarray, np.ndarray]
PairEvaluator = Callable[[np.ndarray, np.ndarray], CollisionReport]
TransitionEvaluator = Callable[[PairState, PairState], CollisionReport]


class CallbackCollisionChecker:
    """Adapts model-specific state and swept-transition collision evaluators."""

    def __init__(
        self,
        pair_evaluator: PairEvaluator | None = None,
        transition_evaluator: TransitionEvaluator | None = None,
    ) -> None:
        self._pair_evaluator = pair_evaluator
        self._transition_evaluator = transition_evaluator

    def state(self, left_q: Sequence[float], right_q: Sequence[float]) -> CollisionReport:
        if self._pair_evaluator is None:
            return CollisionReport()
        return self._pair_evaluator(np.asarray(left_q, dtype=float), np.asarray(right_q, dtype=float))

    def transition(self, previous: PairState, current: PairState) -> CollisionReport:
        if self._transition_evaluator is None:
            return CollisionReport()
        return self._transition_evaluator(previous, current)
