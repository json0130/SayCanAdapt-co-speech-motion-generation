"""Core dataclasses and protocol for the robot abstraction layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


# ASSUMPTION: Gesture uses tuple[str, ...] for joints_used and tags rather than
# list[str] so the frozen dataclass is hashable (lists are unhashable). The
# library loader converts YAML lists to tuples at load time. See BUILD_NOTES.md.
@dataclass(frozen=True)
class Gesture:
    """A discrete gesture from the library. Immutable and hashable."""

    gesture_id: str
    label: str
    description: str
    duration_s: float
    requires_pose: str
    joints_used: tuple[str, ...]
    tags: tuple[str, ...]
    naoqi_animation: str = ""


# ASSUMPTION: RobotState is a regular (mutable) dataclass because joint_positions
# is a dict[str, float], which is unhashable and would break frozen=True. It is
# treated as an immutable snapshot in practice. See BUILD_NOTES.md.
@dataclass
class RobotState:
    """Snapshot of the robot's current physical state."""

    pose: str
    joint_positions: dict[str, float]
    time_since_last_gesture: float
    time_budget_s: float


@dataclass(frozen=True)
class ExecutionResult:
    """Result returned by RobotInterface.execute()."""

    gesture_id: str
    success: bool
    actual_duration_s: float
    error_message: str | None = None


@runtime_checkable
class RobotInterface(Protocol):
    """Protocol for all robot adapters (Pepper hardware, SimPlayback, etc.).

    Everything above this layer talks through this protocol; no code outside
    src/robot/ imports Pepper-specific or simulation-specific types directly.
    """

    @property
    def gesture_library(self) -> list[Gesture]:
        """The full set of gestures available on this robot."""
        ...

    def get_state(self) -> RobotState:
        """Return the robot's current state snapshot."""
        ...

    def execute(self, gesture: Gesture) -> ExecutionResult:
        """Execute a gesture and return the outcome."""
        ...
