"""Unit tests for the three pure Can check functions in src/can/checks.py.

Each check is tested in isolation.  The SPEC requires every pure function to have
at least one unit test; this file provides thorough coverage of all three.
"""

from __future__ import annotations

import math

import pytest

from src.can.checks import LOG_FLOOR, check_joint_range, check_pose, check_time_budget
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
    gid: str = "g1",
    requires_pose: str = "Stand",
    joints_used: tuple[str, ...] = (),
    duration_s: float = 2.0,
) -> Gesture:
    return Gesture(
        gesture_id=gid,
        label=gid,
        description="test",
        duration_s=duration_s,
        requires_pose=requires_pose,
        joints_used=joints_used,
        tags=(),
    )


# ── check_pose ─────────────────────────────────────────────────────────────────

def test_pose_match_returns_zero() -> None:
    assert check_pose(_gesture(requires_pose="Stand"), _state(pose="Stand")) == 0.0


def test_pose_mismatch_returns_log_floor() -> None:
    result = check_pose(_gesture(requires_pose="Sit"), _state(pose="Stand"))
    assert math.isclose(result, LOG_FLOOR, rel_tol=1e-9)


def test_pose_empty_requirement_always_passes() -> None:
    """A gesture with no pose requirement should score 0.0 on any robot pose."""
    for pose in ("Stand", "Sit", "SitRelax", "LyingBack"):
        assert check_pose(_gesture(requires_pose=""), _state(pose=pose)) == 0.0


def test_pose_check_is_case_sensitive() -> None:
    """'stand' ≠ 'Stand' — pose strings must match exactly."""
    result = check_pose(_gesture(requires_pose="stand"), _state(pose="Stand"))
    assert result == LOG_FLOOR


# ── check_joint_range ─────────────────────────────────────────────────────────

def test_joint_range_no_joints_used_returns_zero() -> None:
    assert check_joint_range(_gesture(joints_used=()), _state()) == 0.0


def test_joint_range_empty_robot_state_returns_zero() -> None:
    """When the robot provides no joint data, no penalty is applied."""
    g = _gesture(joints_used=("LShoulderPitch", "RShoulderPitch"))
    assert check_joint_range(g, _state(joints={})) == 0.0


def test_joint_range_neutral_position_returns_zero() -> None:
    g = _gesture(joints_used=("LShoulderPitch",))
    state = _state(joints={"LShoulderPitch": 0.0})
    assert check_joint_range(g, state) == 0.0


def test_joint_range_extreme_position_degrades() -> None:
    """A joint far from neutral (≥ 3.0 rad) should hit the floor."""
    g = _gesture(joints_used=("RShoulderPitch",))
    state = _state(joints={"RShoulderPitch": 3.5})  # 3.5 > 2 × 1.5
    result = check_joint_range(g, state)
    assert result == LOG_FLOOR


def test_joint_range_moderate_deviation_between_zero_and_floor() -> None:
    g = _gesture(joints_used=("HeadYaw",))
    state = _state(joints={"HeadYaw": 0.75})  # deviation = 0.5 × 1.5 → score ≈ -log(10)/2
    result = check_joint_range(g, state)
    expected = -0.5 * math.log(10)
    assert math.isclose(result, expected, rel_tol=1e-6)
    assert LOG_FLOOR < result < 0.0


def test_joint_range_unknown_joint_is_skipped() -> None:
    """Joints in joints_used but absent from robot state are ignored."""
    g = _gesture(joints_used=("UnknownJoint",))
    state = _state(joints={"LShoulderPitch": 0.0})
    assert check_joint_range(g, state) == 0.0


def test_joint_range_only_tracked_joints_contribute() -> None:
    """If only one of several joints is tracked, only that one affects the score."""
    g = _gesture(joints_used=("KnownJoint", "MissingJoint"))
    state = _state(joints={"KnownJoint": 3.5})  # extreme → floor
    result = check_joint_range(g, state)
    assert result == LOG_FLOOR


def test_joint_range_is_symmetric_around_zero() -> None:
    """Positive and negative deviations of equal magnitude should score the same."""
    g = _gesture(joints_used=("J",))
    pos = check_joint_range(g, _state(joints={"J": 1.0}))
    neg = check_joint_range(g, _state(joints={"J": -1.0}))
    assert math.isclose(pos, neg, rel_tol=1e-9)


def test_joint_range_result_clamped_to_floor() -> None:
    """Score must never go below LOG_FLOOR regardless of how extreme the position."""
    g = _gesture(joints_used=("J",))
    state = _state(joints={"J": 1000.0})  # absurdly large
    result = check_joint_range(g, state)
    assert result >= LOG_FLOOR


# ── check_time_budget ─────────────────────────────────────────────────────────

def test_time_budget_ample_returns_zero() -> None:
    g = _gesture(duration_s=2.0)
    assert check_time_budget(g, _state(budget=5.0)) == 0.0


def test_time_budget_exactly_equal_returns_zero() -> None:
    g = _gesture(duration_s=2.0)
    assert check_time_budget(g, _state(budget=2.0)) == 0.0


def test_time_budget_half_returns_log_half() -> None:
    g = _gesture(duration_s=4.0)
    result = check_time_budget(g, _state(budget=2.0))  # ratio = 0.5
    assert math.isclose(result, math.log(0.5), rel_tol=1e-9)


def test_time_budget_near_zero_returns_floor() -> None:
    g = _gesture(duration_s=10.0)
    result = check_time_budget(g, _state(budget=0.05))  # ratio = 0.005 < 0.01
    assert math.isclose(result, LOG_FLOOR, rel_tol=1e-9)


def test_time_budget_zero_returns_floor() -> None:
    g = _gesture(duration_s=2.0)
    result = check_time_budget(g, _state(budget=0.0))
    assert math.isclose(result, LOG_FLOOR, rel_tol=1e-9)


def test_time_budget_is_monotone_in_budget() -> None:
    """A larger time budget should never yield a lower score."""
    g = _gesture(duration_s=3.0)
    budgets = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    scores = [check_time_budget(g, _state(budget=b)) for b in budgets]
    for i in range(len(scores) - 1):
        assert scores[i] <= scores[i + 1], (
            f"Score decreased: budget={budgets[i]:.1f} → {scores[i]:.3f}, "
            f"budget={budgets[i+1]:.1f} → {scores[i+1]:.3f}"
        )


def test_time_budget_result_in_valid_range() -> None:
    g = _gesture(duration_s=2.0)
    for budget in (0.0, 0.5, 1.0, 2.0, 5.0):
        result = check_time_budget(g, _state(budget=budget))
        assert LOG_FLOOR <= result <= 0.0, (
            f"budget={budget}: score={result} outside [{LOG_FLOOR:.2f}, 0.0]"
        )


def test_time_budget_zero_duration_gesture_always_fits() -> None:
    """A zero-duration gesture should score 0.0 regardless of budget."""
    g = _gesture(duration_s=0.0)
    for budget in (0.0, 0.001, 5.0):
        assert check_time_budget(g, _state(budget=budget)) == 0.0
