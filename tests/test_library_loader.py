"""Tests for src/robot/library_loader.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.robot.library_loader import load_gesture_library

LIBRARY_PATH = Path(__file__).parent.parent / "data" / "library" / "pepper_v1.yaml"


# ── happy-path ────────────────────────────────────────────────────────────────

def test_library_loads_without_error() -> None:
    gestures = load_gesture_library(LIBRARY_PATH)
    assert len(gestures) > 0


def test_library_has_expected_count() -> None:
    gestures = load_gesture_library(LIBRARY_PATH)
    # Target is ~30; allow minor edits to the YAML without breaking the test.
    assert 28 <= len(gestures) <= 40, f"Expected ~30 gestures, got {len(gestures)}"


def test_gesture_ids_are_unique() -> None:
    gestures = load_gesture_library(LIBRARY_PATH)
    ids = [g.gesture_id for g in gestures]
    assert len(ids) == len(set(ids)), "Duplicate gesture IDs found"


def test_all_gestures_have_positive_duration() -> None:
    for g in load_gesture_library(LIBRARY_PATH):
        assert g.duration_s > 0.0, f"{g.gesture_id!r} has non-positive duration"


def test_all_gestures_have_nonempty_label_and_description() -> None:
    for g in load_gesture_library(LIBRARY_PATH):
        assert g.label.strip(), f"{g.gesture_id!r} has empty label"
        assert g.description.strip(), f"{g.gesture_id!r} has empty description"


def test_joints_used_and_tags_are_tuples() -> None:
    for g in load_gesture_library(LIBRARY_PATH):
        assert isinstance(g.joints_used, tuple), f"{g.gesture_id!r}: joints_used must be tuple"
        assert isinstance(g.tags, tuple), f"{g.gesture_id!r}: tags must be tuple"


def test_all_required_categories_present() -> None:
    all_tags: set[str] = set()
    for g in load_gesture_library(LIBRARY_PATH):
        all_tags.update(g.tags)
    for required in ("greeting", "beat", "affect", "rest", "deictic", "discourse", "iconic"):
        assert required in all_tags, f"No gesture with tag '{required}' found in library"


def test_gestures_are_frozen_and_hashable() -> None:
    gestures = load_gesture_library(LIBRARY_PATH)
    # Frozen dataclasses with only hashable fields must be usable as dict keys.
    lookup = {g: i for i, g in enumerate(gestures)}
    assert len(lookup) == len(gestures)


# ── error cases ───────────────────────────────────────────────────────────────

def test_load_nonexistent_path_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_gesture_library("/nonexistent/path/library.yaml")


def test_load_yaml_missing_gestures_key_raises_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_gestures: []", encoding="utf-8")
    with pytest.raises(ValueError, match="gestures"):
        load_gesture_library(bad)


def test_load_entry_missing_required_field_raises_value_error(tmp_path: Path) -> None:
    # 'description' is required but absent
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "gestures:\n  - id: g1\n    label: G1\n    duration_s: 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="index 0"):
        load_gesture_library(bad)
