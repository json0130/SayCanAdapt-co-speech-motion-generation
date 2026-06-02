"""Three pure feasibility check functions for the rule-based Can scorer.

Each returns a log-probability in the range [log(0.01), 0.0]:
  - 0.0      → fully feasible
  - log(0.01) → effectively infeasible

The three checks are summed (= multiplied in probability space) by CanScorer.

ASSUMPTION: Joint range check uses the *mean* log-prob across all tracked joints
with a neutral range of ±1.5 rad.  Score degrades linearly from 0.0 at neutral to
log(0.01) at 3.0 rad.  This is a v1 simplification; a proper implementation would
use per-gesture, per-joint starting ranges derived from motion capture data.
See BUILD_NOTES.md.
"""

from __future__ import annotations

import math

from src.robot.interface import Gesture, RobotState

# Score floor for individual checks (≈ probability 0.01 ≈ "effectively infeasible")
LOG_FLOOR: float = math.log(0.01)  # ≈ -4.605

# Distance (radians) from joint neutral (0.0) at which the joint range check
# reaches log(0.1); the floor log(0.01) is reached at 2 × this value.
_JOINT_NEUTRAL_RANGE_RAD: float = 1.5


def check_pose(gesture: Gesture, robot_state: RobotState) -> float:
    """Return 0.0 if pose is compatible, LOG_FLOOR otherwise.

    A gesture with an empty ``requires_pose`` field is always compatible.
    """
    if not gesture.requires_pose or gesture.requires_pose == robot_state.pose:
        return 0.0
    return LOG_FLOOR


def check_joint_range(gesture: Gesture, robot_state: RobotState) -> float:
    """Return the mean log-prob across all tracked joints being near neutral.

    Joints listed in ``gesture.joints_used`` that are not present in
    ``robot_state.joint_positions`` are skipped (assumed in range).  Returns 0.0
    when no joint state information is available.

    Per-joint score degrades linearly:
      0.0       at |position| = 0      (neutral)
      log(0.1)  at |position| = 1.5 rad
      LOG_FLOOR at |position| ≥ 3.0 rad (floored)
    """
    if not gesture.joints_used or not robot_state.joint_positions:
        return 0.0

    per_joint: list[float] = []
    for joint in gesture.joints_used:
        pos = robot_state.joint_positions.get(joint)
        if pos is None:
            continue  # joint not tracked → assume in range
        deviation = abs(pos) / _JOINT_NEUTRAL_RANGE_RAD
        # -deviation * log(10) gives 0.0 at deviation=0, log(0.1) at deviation=1
        per_joint.append(max(LOG_FLOOR, -deviation * math.log(10)))

    return sum(per_joint) / len(per_joint) if per_joint else 0.0


def check_time_budget(gesture: Gesture, robot_state: RobotState) -> float:
    """Return 0.0 when the time budget comfortably covers the gesture duration.

    When ``time_budget_s < duration_s`` the score degrades proportionally:
      budget / duration = 1.0  →  0.0
      budget / duration = 0.1  →  log(0.1)  ≈ -2.303
      budget / duration ≤ 0.01 →  LOG_FLOOR ≈ -4.605
    """
    duration = gesture.duration_s
    if duration <= 0.0:
        return 0.0  # zero-duration gesture always fits

    ratio = min(1.0, robot_state.time_budget_s / duration)
    return math.log(max(0.01, ratio))
