"""Tests for src/robot/sim_playback.py."""

from __future__ import annotations

from src.robot.interface import ExecutionResult, Gesture, RobotInterface, RobotState
from src.robot.sim_playback import SimPlayback


# ── fixtures / helpers ────────────────────────────────────────────────────────

def _gesture(
    gid: str = "test_wave",
    duration_s: float = 2.0,
    requires_pose: str = "Stand",
) -> Gesture:
    return Gesture(
        gesture_id=gid,
        label="Test Gesture",
        description="A gesture used only in tests.",
        duration_s=duration_s,
        requires_pose=requires_pose,
        joints_used=("RShoulderPitch", "RElbowRoll"),
        tags=("test",),
    )


def _two_gesture_library() -> list[Gesture]:
    return [_gesture("g1"), _gesture("g2")]


# ── protocol conformance ──────────────────────────────────────────────────────

def test_satisfies_robot_interface_protocol() -> None:
    sim = SimPlayback(library=_two_gesture_library())
    assert isinstance(sim, RobotInterface)


# ── construction and state ────────────────────────────────────────────────────

def test_gesture_library_accessible() -> None:
    lib = _two_gesture_library()
    sim = SimPlayback(library=lib)
    assert sim.gesture_library == lib


def test_default_state_is_stand_pose() -> None:
    sim = SimPlayback(library=_two_gesture_library())
    state = sim.get_state()
    assert state.pose == "Stand"


def test_default_state_has_positive_time_budget() -> None:
    state = SimPlayback(library=_two_gesture_library()).get_state()
    assert state.time_budget_s > 0.0


def test_custom_initial_state_is_respected() -> None:
    custom = RobotState(
        pose="Sit",
        joint_positions={"HeadYaw": 0.2},
        time_since_last_gesture=1.0,
        time_budget_s=4.0,
    )
    sim = SimPlayback(library=_two_gesture_library(), initial_state=custom)
    assert sim.get_state().pose == "Sit"
    assert sim.get_state().joint_positions == {"HeadYaw": 0.2}


def test_set_state_updates_state() -> None:
    sim = SimPlayback(library=_two_gesture_library())
    new_state = RobotState(
        pose="Sit",
        joint_positions={},
        time_since_last_gesture=5.0,
        time_budget_s=8.0,
    )
    sim.set_state(new_state)
    assert sim.get_state().pose == "Sit"
    assert sim.get_state().time_budget_s == 8.0


# ── execute() — success ───────────────────────────────────────────────────────

def test_execute_returns_execution_result() -> None:
    sim = SimPlayback(library=_two_gesture_library())
    result = sim.execute(_gesture())
    assert isinstance(result, ExecutionResult)


def test_execute_success_on_matching_pose() -> None:
    sim = SimPlayback(library=_two_gesture_library())
    result = sim.execute(_gesture(requires_pose="Stand"))
    assert result.success is True
    assert result.actual_duration_s == 2.0
    assert result.error_message is None


def test_execute_success_at_exactly_half_budget() -> None:
    # time_budget_s == duration_s * 0.5 is the boundary: should succeed.
    state = RobotState(pose="Stand", joint_positions={}, time_since_last_gesture=0.0, time_budget_s=1.0)
    sim = SimPlayback(library=_two_gesture_library(), initial_state=state)
    result = sim.execute(_gesture(duration_s=2.0))  # min_budget = 1.0
    assert result.success is True


# ── execute() — failures ──────────────────────────────────────────────────────

def test_execute_fails_on_pose_mismatch() -> None:
    sim = SimPlayback(library=_two_gesture_library())  # default pose: Stand
    result = sim.execute(_gesture(requires_pose="Sit"))
    assert result.success is False
    assert result.actual_duration_s == 0.0
    assert result.error_message is not None
    assert "pose" in result.error_message.lower() or "Pose" in result.error_message


def test_execute_fails_below_half_budget() -> None:
    state = RobotState(pose="Stand", joint_positions={}, time_since_last_gesture=0.0, time_budget_s=0.9)
    sim = SimPlayback(library=_two_gesture_library(), initial_state=state)
    result = sim.execute(_gesture(duration_s=2.0))  # min_budget = 1.0; budget < min
    assert result.success is False
    assert "time" in (result.error_message or "").lower()


def test_execute_failure_reports_correct_gesture_id() -> None:
    sim = SimPlayback(library=_two_gesture_library())
    g = _gesture(gid="special_gesture", requires_pose="Sit")
    result = sim.execute(g)
    assert result.gesture_id == "special_gesture"


# ── determinism ───────────────────────────────────────────────────────────────

def test_execute_is_deterministic() -> None:
    sim = SimPlayback(library=_two_gesture_library())
    g = _gesture()
    results = [sim.execute(g) for _ in range(5)]
    assert all(r.success == results[0].success for r in results)
    assert all(r.actual_duration_s == results[0].actual_duration_s for r in results)
