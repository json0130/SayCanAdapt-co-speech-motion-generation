"""CanScorer protocol and v1 rule-based implementation."""

from __future__ import annotations

from typing import Protocol

from src.can.checks import check_joint_range, check_pose, check_time_budget
from src.robot.interface import Gesture, RobotState


class CanScorer(Protocol):
    """Scores candidate gestures by how feasible they are on the robot right now."""

    def score(
        self,
        robot_state: RobotState,
        gestures: list[Gesture],
    ) -> dict[str, float]:
        """Return a log-probability per gesture_id.  Higher = more feasible."""
        ...


class RuleBasedCanScorer:
    """v1 Can scorer: three rule-based checks combined in log space.

    score(g) = log p_can(g | state)
             = check_pose(g, state)
             + check_joint_range(g, state)
             + check_time_budget(g, state)

    Each sub-check returns a log-probability in [log(0.01), 0.0]; summing them
    is equivalent to multiplying the three probabilities in linear space.
    """

    def score(
        self,
        robot_state: RobotState,
        gestures: list[Gesture],
    ) -> dict[str, float]:
        """Return a combined log-probability per gesture_id."""
        return {
            g.gesture_id: (
                check_pose(g, robot_state)
                + check_joint_range(g, robot_state)
                + check_time_budget(g, robot_state)
            )
            for g in gestures
        }
