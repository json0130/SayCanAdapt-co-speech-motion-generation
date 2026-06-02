"""Selector: combines scorer outputs into a gesture selection.

The core scoring rule (from SPEC.md §2):

    score(g) = a · log p_say(g | utterance, context)
             + b · log p_can(g | robot_state, time_budget)
             + c · log p_adapt(g | user_model, history)
             − d · repetition_penalty(g, recent_gestures)

Setting adapt_scorer=None and weights.c=0, weights.d=0 reduces exactly to the
vanilla SayCan formulation.  This is both the B1 baseline and the M0 ablation.
The identity M0 ≡ B1 is enforced by tests/test_selector_identity.py.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections import deque

from src.adapt.base import AdaptScorer
from src.can.scorer import CanScorer
from src.robot.interface import Gesture, RobotState
from src.say.scorer import SayScorer

# ASSUMPTION: standard logging used here until src/pipeline/logging.py exists (Step 7).
_log = logging.getLogger(__name__)


@dataclasses.dataclass
class SelectorWeights:
    """Tunable weights for the four scoring terms."""

    a: float = 1.0  # Say  — LLM semantic relevance
    b: float = 1.0  # Can  — embodiment feasibility
    c: float = 1.0  # Adapt — online user model
    d: float = 0.5  # Repetition penalty


@dataclasses.dataclass
class SelectionRecord:
    """Everything that produced a selection — written verbatim to the per-turn log."""

    gesture: Gesture
    say_scores: dict[str, float]
    can_scores: dict[str, float]
    adapt_scores: dict[str, float]
    combined_scores: dict[str, float]
    say_fallback: bool = False


class Selector:
    """Combines Say, Can, and Adapt scores into a single gesture selection.

    The repetition penalty is computed internally from a rolling window of recent
    selections; it is not a separate scorer because it has no external dependencies
    and no learning component.
    """

    _LOG_FLOOR: float = math.log(1e-10)

    def __init__(
        self,
        say_scorer: SayScorer,
        can_scorer: CanScorer,
        adapt_scorer: AdaptScorer | None = None,
        weights: SelectorWeights | None = None,
        repetition_window: int = 5,
    ) -> None:
        self._say = say_scorer
        self._can = can_scorer
        self._adapt = adapt_scorer
        self._weights = weights if weights is not None else SelectorWeights()
        self._recent: deque[str] = deque(maxlen=repetition_window)

    def select(
        self,
        utterance: str,
        dialogue_history: list[str],
        robot_state: RobotState,
        user_id: str,
        gestures: list[Gesture],
    ) -> SelectionRecord:
        """Score every candidate gesture and return the argmax selection."""
        if not gestures:
            raise ValueError("gestures list must not be empty")

        say_scores, say_fallback = self._score_say(utterance, dialogue_history, gestures)
        can_scores = self._can.score(robot_state, gestures)

        if self._adapt is not None:
            adapt_scores = self._adapt.score(utterance, dialogue_history, gestures, user_id)
        else:
            # Uniform log-probs: constant offset, does not affect argmax.
            uniform = math.log(1.0 / len(gestures))
            adapt_scores = {g.gesture_id: uniform for g in gestures}

        combined: dict[str, float] = {}
        w = self._weights
        for g in gestures:
            gid = g.gesture_id
            rep = self._recent.count(gid)
            combined[gid] = (
                w.a * say_scores.get(gid, self._LOG_FLOOR)
                + w.b * can_scores.get(gid, self._LOG_FLOOR)
                + w.c * adapt_scores.get(gid, self._LOG_FLOOR)
                - w.d * rep
            )

        chosen_id = max(combined, key=lambda k: combined[k])
        chosen = next(g for g in gestures if g.gesture_id == chosen_id)
        self._recent.append(chosen_id)

        return SelectionRecord(
            gesture=chosen,
            say_scores=say_scores,
            can_scores=can_scores,
            adapt_scores=adapt_scores,
            combined_scores=combined,
            say_fallback=say_fallback,
        )

    def observe_reward(
        self,
        gesture: Gesture,
        utterance: str,
        user_id: str,
        reward: float,
    ) -> None:
        """Propagate a reward signal to the Adapt scorer; no-op when adapt is None."""
        if self._adapt is not None:
            self._adapt.update(gesture, utterance, user_id, reward)

    def _score_say(
        self,
        utterance: str,
        dialogue_history: list[str],
        gestures: list[Gesture],
    ) -> tuple[dict[str, float], bool]:
        """Call the Say scorer, catching all failures and returning a fallback flag.

        The pipeline must never crash because the LLM is unavailable or slow.
        Any exception from the scorer falls back to uniform log-probs and sets
        say_fallback=True in the SelectionRecord so the per-turn log records it.
        """
        try:
            scores = self._say.score(utterance, dialogue_history, gestures)
            return scores, False
        except Exception as exc:  # noqa: BLE001  intentional: Say scorer must not crash pipeline
            _log.warning("Say scorer raised an exception, using uniform fallback: %s", exc)
            uniform = math.log(1.0 / len(gestures))
            return {g.gesture_id: uniform for g in gestures}, True
