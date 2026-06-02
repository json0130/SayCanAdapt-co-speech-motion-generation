"""Loads the gesture library from a YAML file into typed Gesture instances."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.robot.interface import Gesture


def load_gesture_library(path: str | Path) -> list[Gesture]:
    """Parse a gesture-library YAML file and return a list of Gesture objects.

    The YAML must have a top-level ``gestures`` list.  Each entry must supply
    at minimum: ``id``, ``label``, ``description``, ``duration_s``.  Optional
    fields default to sensible values when absent.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the YAML is structurally invalid or a required field is missing.
    """
    resolved = Path(path)
    raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))

    if not isinstance(raw, dict) or "gestures" not in raw:
        raise ValueError(
            f"Library YAML at {str(path)!r} must have a top-level 'gestures' key"
        )

    gestures: list[Gesture] = []
    for i, entry in enumerate(raw["gestures"]):
        try:
            gestures.append(
                Gesture(
                    gesture_id=str(entry["id"]),
                    label=str(entry["label"]),
                    description=str(entry["description"]),
                    duration_s=float(entry["duration_s"]),
                    requires_pose=str(entry.get("requires_pose", "Stand")),
                    joints_used=tuple(str(j) for j in entry.get("joints_used", [])),
                    tags=tuple(str(t) for t in entry.get("tags", [])),
                    naoqi_animation=str(entry.get("naoqi_animation", "")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid gesture entry at index {i} in {str(path)!r}: {exc}"
            ) from exc

    return gestures
