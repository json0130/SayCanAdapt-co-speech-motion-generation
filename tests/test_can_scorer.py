"""Tests for RuleBasedCanScorer in src/can/scorer.py."""

from __future__ import annotations

import math

from src.can.checks import LOG_FLOOR, check_joint_range, check_pose, check_time_budget
from src.can.scorer import CanScorer, RuleBasedCanScorer
from src.robot.interface import Gesture, RobotState


# ── helpers ───────────────────────────────────────────────────────────────────

def _state(
    pose: str = "Stand",
    joints: dict[str, float] | None = None,
    budget: float = 10.0,
) -> RobotState:
    return RobotState(
        pose=pose,
        joint_positions=joints if joints is not None else {},
        time_since_last_gesture=1.0,
        time_budget_s=budget,
    )


def _gesture(
    gid: str,
    requires_pose: str = "Stand",
    joints_used: tuple[str, ...] = (),
    duration_s: float = 2.0,
    tags: tuple[str, ...] = (),
) -> Gesture:
    return Gesture(
        gesture_id=gid,
        label=gid,
        description="test",
        duration_s=duration_s,
        requires_pose=requires_pose,
        joints_used=joints_used,
        tags=tags,
    )


# ── protocol conformance ──────────────────────────────────────────────────────

def test_rule_based_satisfies_can_scorer_protocol() -> None:
    """RuleBasedCanScorer must structurally satisfy the CanScorer Protocol."""
    scorer: CanScorer = RuleBasedCanScorer()  # type: ignore[assignment]  # mypy verifies at check time
    assert hasattr(scorer, "score")


# ── output structure ──────────────────────────────────────────────────────────

def test_scorer_returns_entry_for_every_gesture(gesture_library: list[Gesture]) -> None:
    scorer = RuleBasedCanScorer()
    state = _state()
    result = scorer.score(state, gesture_library)
    assert set(result.keys()) == {g.gesture_id for g in gesture_library}


def test_scorer_returns_empty_dict_for_empty_gesture_list() -> None:
    scorer = RuleBasedCanScorer()
    result = scorer.score(_state(), [])
    assert result == {}


def test_all_scores_are_finite_log_probs(gesture_library: list[Gesture]) -> None:
    scorer = RuleBasedCanScorer()
    for score in scorer.score(_state(), gesture_library).values():
        assert math.isfinite(score), f"Non-finite score: {score}"
        assert score <= 0.0, f"Log-prob > 0: {score}"


# ── combined score correctness ────────────────────────────────────────────────

def test_combined_score_equals_sum_of_checks() -> None:
    """The total score must equal the sum of the three individual check results."""
    scorer = RuleBasedCanScorer()
    g = _gesture("g1", requires_pose="Stand", joints_used=("HeadYaw",), duration_s=3.0)
    state = _state(joints={"HeadYaw": 0.5}, budget=2.0)

    total = scorer.score(state, [g])["g1"]
    expected = (
        check_pose(g, state)
        + check_joint_range(g, state)
        + check_time_budget(g, state)
    )
    assert math.isclose(total, expected, rel_tol=1e-9)


def test_ample_state_gives_zero_score() -> None:
    """Matching pose, neutral joints, large budget → all checks 0.0 → total 0.0."""
    scorer = RuleBasedCanScorer()
    g = _gesture("g1", requires_pose="Stand", joints_used=("HeadYaw",), duration_s=1.0)
    state = _state(pose="Stand", joints={"HeadYaw": 0.0}, budget=10.0)
    assert scorer.score(state, [g])["g1"] == 0.0


# ── pose penalty ──────────────────────────────────────────────────────────────

def test_pose_mismatch_lowers_score() -> None:
    scorer = RuleBasedCanScorer()
    g_stand = _gesture("stand", requires_pose="Stand")
    g_sit = _gesture("sit", requires_pose="Sit")
    state = _state(pose="Stand")
    scores = scorer.score(state, [g_stand, g_sit])
    assert scores["stand"] > scores["sit"], "Standing gesture should score higher in Stand pose"


def test_pose_mismatch_applies_log_floor() -> None:
    scorer = RuleBasedCanScorer()
    g = _gesture("g", requires_pose="Sit", duration_s=0.0)  # only pose check active
    # With empty joint state and zero duration → only pose check contributes
    state = _state(pose="Stand", joints={}, budget=100.0)
    result = scorer.score(state, [g])["g"]
    assert math.isclose(result, LOG_FLOOR, rel_tol=1e-6)


# ── time budget penalty ───────────────────────────────────────────────────────

def test_tight_budget_scores_lower_than_ample_budget() -> None:
    scorer = RuleBasedCanScorer()
    g = _gesture("g", duration_s=4.0)
    ample = scorer.score(_state(budget=10.0), [g])["g"]
    tight = scorer.score(_state(budget=1.0), [g])["g"]  # 1/4 budget
    assert tight < ample


def test_zero_budget_hits_log_floor_contribution() -> None:
    scorer = RuleBasedCanScorer()
    g = _gesture("g", duration_s=2.0)
    # Only time check should be at floor; pose and joint checks are 0.0
    state = _state(budget=0.0, joints={})
    result = scorer.score(state, [g])["g"]
    assert math.isclose(result, LOG_FLOOR, rel_tol=1e-6)


# ── determinism ───────────────────────────────────────────────────────────────

def test_scorer_is_deterministic(gesture_library: list[Gesture]) -> None:
    scorer = RuleBasedCanScorer()
    state = _state()
    r1 = scorer.score(state, gesture_library)
    r2 = scorer.score(state, gesture_library)
    assert r1 == r2


# ── integration with Selector types ──────────────────────────────────────────

def test_scorer_output_usable_by_selector() -> None:
    """Spot-check that the scorer's dict can be consumed by Selector-style code."""
    from src.pipeline.selector import Selector, SelectorWeights
    from src.say.scorer import SayScorer

    class _FlatSay:
        def score(
            self, utterance: str, history: list[str], gestures: list[Gesture]
        ) -> dict[str, float]:
            return {g.gesture_id: math.log(0.5) for g in gestures}

    gestures = [
        _gesture("g1", requires_pose="Stand", duration_s=2.0),
        _gesture("g2", requires_pose="Sit", duration_s=2.0),
    ]
    state = _state(pose="Stand", budget=5.0)

    scorer = RuleBasedCanScorer()
    sel = Selector(
        say_scorer=_FlatSay(),
        can_scorer=scorer,
        adapt_scorer=None,
        weights=SelectorWeights(a=0.0, b=1.0, c=0.0, d=0.0),  # only Can matters
    )
    rec = sel.select("hello", [], state, "u0", gestures)
    # Selector should pick g1 (Stand) over g2 (Sit) because Can score is higher
    assert rec.gesture.gesture_id == "g1"
