"""SayScorer protocol — semantic relevance of gestures to an utterance."""

from __future__ import annotations

from typing import Protocol

from src.robot.interface import Gesture


class SayScorer(Protocol):
    """Scores candidate gestures by how semantically appropriate they are for an utterance.

    The implementation (AnthropicLLMClient etc.) lives in src/say/; this file
    defines only the interface that Selector and tests depend on.
    """

    def score(
        self,
        utterance: str,
        dialogue_history: list[str],
        gestures: list[Gesture],
    ) -> dict[str, float]:
        """Return a log-probability per gesture_id.  Higher = more relevant.

        On internal failure (LLM call error, malformed JSON) implementations
        should return uniform log-probs rather than raise; the Selector also
        wraps this call defensively.  See ASSUMPTION in BUILD_NOTES.md.
        """
        ...
