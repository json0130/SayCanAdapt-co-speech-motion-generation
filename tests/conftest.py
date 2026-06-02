"""Shared pytest fixtures for the SayCan-Gesture test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.robot.interface import Gesture
from src.robot.library_loader import load_gesture_library

_LIBRARY_PATH = Path(__file__).parent.parent / "data" / "library" / "pepper_v1.yaml"
_MAPPING_PATH = Path(__file__).parent.parent / "data" / "library" / "dataset_to_pepper.yaml"


@pytest.fixture(scope="session")
def gesture_library() -> list[Gesture]:
    """The full Pepper gesture library loaded from pepper_v1.yaml."""
    return load_gesture_library(_LIBRARY_PATH)


@pytest.fixture(scope="session")
def gesture_ids(gesture_library: list[Gesture]) -> list[str]:
    """Just the gesture IDs, for use in data-layer tests."""
    return [g.gesture_id for g in gesture_library]


@pytest.fixture(scope="session")
def known_tags(gesture_library: list[Gesture]) -> list[str]:
    """Sorted list of all unique tags present in the library."""
    all_tags: set[str] = set()
    for g in gesture_library:
        all_tags.update(g.tags)
    return sorted(all_tags)


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh seeded Generator for each test (seed 42)."""
    return np.random.default_rng(42)


@pytest.fixture
def library_path() -> Path:
    return _LIBRARY_PATH


@pytest.fixture
def mapping_path() -> Path:
    return _MAPPING_PATH
