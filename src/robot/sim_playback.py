"""SimPlayback: deterministic robot adapter for development and offline evaluation."""

from __future__ import annotations

from src.robot.interface import ExecutionResult, Gesture, RobotState


class SimPlayback:
    """Deterministic stand-in for a real robot.

    Holds the gesture library and a configurable RobotState. execute() applies
    simple pose-compatibility and time-budget checks to produce a deterministic
    ExecutionResult — no networking, no NAOqi dependency, runs everywhere.

    ASSUMPTION: time-budget failure threshold is duration_s * 0.5. See BUILD_NOTES.md.
    """

    def __init__(
        self,
        library: list[Gesture],
        initial_state: RobotState | None = None,
    ) -> None:
        self._library = library
        self._state: RobotState = initial_state if initial_state is not None else RobotState(
            pose="Stand",
            joint_positions={},
            time_since_last_gesture=10.0,
            time_budget_s=5.0,
        )

    @property
    def gesture_library(self) -> list[Gesture]:
        return self._library

    def get_state(self) -> RobotState:
        return self._state

    def set_state(self, state: RobotState) -> None:
        """Replace the current simulated state (useful for test setup)."""
        self._state = state

    def execute(self, gesture: Gesture) -> ExecutionResult:
        """Return a deterministic result based on pose and time-budget feasibility."""
        if gesture.requires_pose and gesture.requires_pose != self._state.pose:
            return ExecutionResult(
                gesture_id=gesture.gesture_id,
                success=False,
                actual_duration_s=0.0,
                error_message=(
                    f"Pose mismatch: gesture requires '{gesture.requires_pose}', "
                    f"robot is in '{self._state.pose}'"
                ),
            )
        min_budget = gesture.duration_s * 0.5
        if self._state.time_budget_s < min_budget:
            return ExecutionResult(
                gesture_id=gesture.gesture_id,
                success=False,
                actual_duration_s=0.0,
                error_message=(
                    f"Insufficient time budget: {self._state.time_budget_s:.2f}s available, "
                    f"need at least {min_budget:.2f}s"
                ),
            )
        return ExecutionResult(
            gesture_id=gesture.gesture_id,
            success=True,
            actual_duration_s=gesture.duration_s,
        )
