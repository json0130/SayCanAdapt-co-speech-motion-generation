"""
Guardrail test: M0 (adapt=None, d=0) must produce numerically identical
selections to B1 (vanilla SayCan baseline).

B1 IS our Selector configured with adapt_scorer=None and weights.d=0 — the
two are the same configuration.  This test uses lightweight mock scorers so
it is green from Step 1 onward without any real LLM or Can implementation.

DO NOT weaken this test.  If it fails, fix the Selector code, not the test.
"""

from __future__ import annotations

import math

import pytest

from src.pipeline.selector import SelectionRecord, Selector, SelectorWeights
from src.robot.interface import Gesture, RobotState


# ── helpers ───────────────────────────────────────────────────────────────────

def _gesture(gid: str) -> Gesture:
    return Gesture(
        gesture_id=gid,
        label=gid,
        description=f"Test gesture {gid}",
        duration_s=2.0,
        requires_pose="Stand",
        joints_used=(),
        tags=(),
    )


class _MockSayScorer:
    """Deterministic say scorer for identity tests."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(
        self,
        utterance: str,
        dialogue_history: list[str],
        gestures: list[Gesture],
    ) -> dict[str, float]:
        return {g.gesture_id: self._scores.get(g.gesture_id, math.log(0.5)) for g in gestures}


class _MockCanScorer:
    """Deterministic can scorer for identity tests."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(
        self,
        robot_state: RobotState,
        gestures: list[Gesture],
    ) -> dict[str, float]:
        return {g.gesture_id: self._scores.get(g.gesture_id, math.log(0.5)) for g in gestures}


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def gestures() -> list[Gesture]:
    return [_gesture("g1"), _gesture("g2"), _gesture("g3")]


@pytest.fixture
def robot_state() -> RobotState:
    return RobotState(
        pose="Stand",
        joint_positions={},
        time_since_last_gesture=10.0,
        time_budget_s=10.0,
    )


@pytest.fixture
def scorers() -> tuple[_MockSayScorer, _MockCanScorer]:
    # g1 wins on Say, g2 wins on Can → g1 takes overall with a=b=1
    # g1: log(0.7)+log(0.3) ≈ -1.56   g2: log(0.2)+log(0.6) ≈ -2.12   g3: log(0.1)+log(0.1) ≈ -4.61
    return (
        _MockSayScorer({"g1": math.log(0.7), "g2": math.log(0.2), "g3": math.log(0.1)}),
        _MockCanScorer({"g1": math.log(0.3), "g2": math.log(0.6), "g3": math.log(0.1)}),
    )


# ── core identity test ────────────────────────────────────────────────────────

def test_m0_and_b1_produce_identical_selections(
    gestures: list[Gesture],
    robot_state: RobotState,
    scorers: tuple[_MockSayScorer, _MockCanScorer],
) -> None:
    """M0 ≡ B1: both are Selector(adapt=None, c=0, d=0) — must match exactly."""
    say, can = scorers
    vanilla = SelectorWeights(a=1.0, b=1.0, c=0.0, d=0.0)

    m0 = Selector(say_scorer=say, can_scorer=can, adapt_scorer=None, weights=vanilla)
    b1 = Selector(say_scorer=say, can_scorer=can, adapt_scorer=None, weights=vanilla)

    for i in range(5):
        m0_rec = m0.select("Hello", [], robot_state, "u0", gestures)
        b1_rec = b1.select("Hello", [], robot_state, "u0", gestures)

        assert m0_rec.gesture.gesture_id == b1_rec.gesture.gesture_id, (
            f"Round {i}: M0 chose {m0_rec.gesture.gesture_id!r}, "
            f"B1 chose {b1_rec.gesture.gesture_id!r}. "
            "Identity invariant violated — check Selector for divergent logic."
        )
        assert m0_rec.combined_scores == b1_rec.combined_scores, (
            f"Round {i}: combined scores differ between M0 and B1. "
            "Identity invariant violated."
        )


def test_m0_and_b1_combined_scores_equal_say_plus_can(
    gestures: list[Gesture],
    robot_state: RobotState,
    scorers: tuple[_MockSayScorer, _MockCanScorer],
) -> None:
    """With c=0, d=0, the combined score must equal a*say + b*can exactly."""
    say, can = scorers
    weights = SelectorWeights(a=1.0, b=1.0, c=0.0, d=0.0)
    sel = Selector(say_scorer=say, can_scorer=can, adapt_scorer=None, weights=weights)

    rec = sel.select("Hello", [], robot_state, "u0", gestures)

    for g in gestures:
        gid = g.gesture_id
        expected = rec.say_scores[gid] + rec.can_scores[gid]
        # adapt term: c=0 so it's 0 * uniform = 0; rep term: d=0 so 0 * count = 0
        assert math.isclose(rec.combined_scores[gid], expected, rel_tol=1e-9), (
            f"combined[{gid!r}] = {rec.combined_scores[gid]}, "
            f"expected say+can = {expected}"
        )


def test_adapt_none_with_nonzero_c_same_selection_as_c_zero(
    gestures: list[Gesture],
    robot_state: RobotState,
    scorers: tuple[_MockSayScorer, _MockCanScorer],
) -> None:
    """With adapt=None, changing c is a constant shift that must not affect argmax."""
    say, can = scorers
    c_zero = Selector(
        say_scorer=say, can_scorer=can, adapt_scorer=None,
        weights=SelectorWeights(a=1.0, b=1.0, c=0.0, d=0.0),
    )
    c_one = Selector(
        say_scorer=say, can_scorer=can, adapt_scorer=None,
        weights=SelectorWeights(a=1.0, b=1.0, c=1.0, d=0.0),
    )
    r0 = c_zero.select("Hello", [], robot_state, "u0", gestures)
    r1 = c_one.select("Hello", [], robot_state, "u0", gestures)
    assert r0.gesture.gesture_id == r1.gesture.gesture_id, (
        "With adapt=None, uniform adapt scores form a constant — "
        "argmax must not change when c varies."
    )


def test_selection_is_deterministic_without_repetition_penalty(
    gestures: list[Gesture],
    robot_state: RobotState,
    scorers: tuple[_MockSayScorer, _MockCanScorer],
) -> None:
    """With d=0 there is no stochastic component; same inputs → same output."""
    say, can = scorers
    sel = Selector(
        say_scorer=say, can_scorer=can, adapt_scorer=None,
        weights=SelectorWeights(a=1.0, b=1.0, c=0.0, d=0.0),
    )
    choices = [
        sel.select("Test", [], robot_state, "u0", gestures).gesture.gesture_id
        for _ in range(10)
    ]
    assert len(set(choices)) == 1, (
        f"Expected a single repeated choice with d=0, got: {set(choices)}"
    )


def test_observe_reward_is_noop_without_adapt(
    gestures: list[Gesture],
    robot_state: RobotState,
    scorers: tuple[_MockSayScorer, _MockCanScorer],
) -> None:
    """observe_reward must not raise and must not affect future selections when adapt is None."""
    say, can = scorers
    sel = Selector(say_scorer=say, can_scorer=can, adapt_scorer=None)
    r_before = sel.select("Hello", [], robot_state, "u0", gestures)

    sel.observe_reward(r_before.gesture, "Hello", "u0", 1.0)  # must not raise

    r_after = sel.select("Hello", [], robot_state, "u0", gestures)
    assert r_before.gesture.gesture_id == r_after.gesture.gesture_id, (
        "observe_reward with adapt=None must be a no-op — selection changed unexpectedly."
    )


def test_say_fallback_flag_propagates_on_scorer_exception(
    gestures: list[Gesture],
    robot_state: RobotState,
) -> None:
    """When SayScorer raises, say_fallback must be True in the SelectionRecord."""

    class _BrokenSayScorer:
        def score(
            self,
            utterance: str,
            dialogue_history: list[str],
            gestures: list[Gesture],
        ) -> dict[str, float]:
            raise RuntimeError("simulated LLM outage")

    class _FlatCanScorer:
        def score(
            self,
            robot_state: RobotState,
            gestures: list[Gesture],
        ) -> dict[str, float]:
            return {g.gesture_id: math.log(0.5) for g in gestures}

    sel = Selector(say_scorer=_BrokenSayScorer(), can_scorer=_FlatCanScorer(), adapt_scorer=None)
    rec = sel.select("Hello", [], robot_state, "u0", gestures)

    assert rec.say_fallback is True, "say_fallback must be True when SayScorer raises"
    # Uniform say scores → all say scores equal
    unique_say_vals = set(rec.say_scores.values())
    assert len(unique_say_vals) == 1, "Fallback say scores must be uniform"
