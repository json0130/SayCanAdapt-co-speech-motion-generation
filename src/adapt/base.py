"""AdaptScorer protocol — online per-user preference model."""

from __future__ import annotations

from typing import Any, Protocol

from src.robot.interface import Gesture


class AdaptScorer(Protocol):
    """Scores gestures based on what has worked for a specific user in the past.

    Three implementations live in src/adapt/ (bandit, memory, adapter); this
    file defines only the shared interface.

    Cold-start invariant (enforced by tests/test_adapt_cold_start.py):
    With zero history for a user, score() must return near-uniform
    log-probabilities so that the Adapt term contributes nothing on day 1.
    """

    def score(
        self,
        utterance: str,
        dialogue_history: list[str],
        gestures: list[Gesture],
        user_id: str,
    ) -> dict[str, float]:
        """Return a log-probability per gesture_id for this user."""
        ...

    def update(
        self,
        gesture: Gesture,
        utterance: str,
        user_id: str,
        reward: float,
    ) -> None:
        """Incorporate a reward signal into the user model."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Serialise internal state for checkpointing.

        ASSUMPTION: uses dict[str, Any] because the three variants store very
        different structures. See BUILD_NOTES.md.
        """
        ...

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Restore internal state from a checkpoint produced by state_dict()."""
        ...
