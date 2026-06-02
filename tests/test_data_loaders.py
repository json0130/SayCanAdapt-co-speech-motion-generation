"""Tests for src/data/loaders.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.loaders import (
    UtteranceGesturePair,
    load_dataset_to_pepper_map,
    make_synthetic_dataset,
)


# ── make_synthetic_dataset ────────────────────────────────────────────────────

def test_synthetic_dataset_is_reproducible(gesture_ids: list[str]) -> None:
    """Same seed must produce byte-identical datasets."""
    d1 = make_synthetic_dataset(np.random.default_rng(42), gesture_ids)
    d2 = make_synthetic_dataset(np.random.default_rng(42), gesture_ids)
    assert d1 == d2


def test_synthetic_dataset_differs_with_different_seeds(gesture_ids: list[str]) -> None:
    d1 = make_synthetic_dataset(np.random.default_rng(1), gesture_ids)
    d2 = make_synthetic_dataset(np.random.default_rng(2), gesture_ids)
    assert d1 != d2


def test_synthetic_dataset_total_count(gesture_ids: list[str]) -> None:
    pairs = make_synthetic_dataset(
        np.random.default_rng(0), gesture_ids, num_speakers=5, utterances_per_speaker=10
    )
    assert len(pairs) == 50


def test_synthetic_dataset_speaker_count(gesture_ids: list[str]) -> None:
    pairs = make_synthetic_dataset(
        np.random.default_rng(0), gesture_ids, num_speakers=8, utterances_per_speaker=10
    )
    assert len({p.speaker_id for p in pairs}) == 8


def test_synthetic_dataset_all_gesture_ids_valid(gesture_ids: list[str]) -> None:
    pairs = make_synthetic_dataset(np.random.default_rng(0), gesture_ids)
    gesture_id_set = set(gesture_ids)
    for p in pairs:
        assert p.gesture_id in gesture_id_set, (
            f"pair {p.pair_id!r} has gesture_id {p.gesture_id!r} not in gesture_ids"
        )


def test_synthetic_dataset_pair_ids_are_unique(gesture_ids: list[str]) -> None:
    pairs = make_synthetic_dataset(np.random.default_rng(0), gesture_ids)
    ids = [p.pair_id for p in pairs]
    assert len(ids) == len(set(ids))


def test_synthetic_dataset_returns_utterance_gesture_pairs(gesture_ids: list[str]) -> None:
    pairs = make_synthetic_dataset(np.random.default_rng(0), gesture_ids, num_speakers=2)
    assert all(isinstance(p, UtteranceGesturePair) for p in pairs)
    assert all(p.utterance.strip() for p in pairs), "All utterances must be non-empty"


def test_synthetic_dataset_dataset_field(gesture_ids: list[str]) -> None:
    pairs = make_synthetic_dataset(
        np.random.default_rng(0), gesture_ids, dataset_name="test_ds"
    )
    assert all(p.dataset == "test_ds" for p in pairs)


def test_synthetic_dataset_empty_gesture_ids_raises() -> None:
    with pytest.raises(ValueError, match="gesture_ids"):
        make_synthetic_dataset(np.random.default_rng(0), [])


def test_synthetic_dataset_zero_speakers_raises(gesture_ids: list[str]) -> None:
    with pytest.raises(ValueError, match="num_speakers"):
        make_synthetic_dataset(np.random.default_rng(0), gesture_ids, num_speakers=0)


# ── load_dataset_to_pepper_map ─────────────────────────────────────────────────

def test_mapping_loads_without_error(mapping_path: Path) -> None:
    result = load_dataset_to_pepper_map(mapping_path)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_mapping_contains_beat2(mapping_path: Path) -> None:
    result = load_dataset_to_pepper_map(mapping_path)
    assert "beat2" in result, "beat2 mapping must be present"


def test_mapping_beat2_keys_are_strings(mapping_path: Path) -> None:
    result = load_dataset_to_pepper_map(mapping_path)
    for key, val in result["beat2"].items():
        assert isinstance(key, str)
        assert val is None or isinstance(val, str)


def test_mapping_null_values_are_none(mapping_path: Path) -> None:
    """Gesture labels with pepper_id: null must appear as None (not the string 'null')."""
    result = load_dataset_to_pepper_map(mapping_path)
    for dataset_map in result.values():
        for label, pepper_id in dataset_map.items():
            assert pepper_id is None or isinstance(pepper_id, str), (
                f"pepper_id for label '{label}' must be None or str, got {type(pepper_id)}"
            )


def test_mapping_beat_maps_to_a_pepper_id(mapping_path: Path) -> None:
    result = load_dataset_to_pepper_map(mapping_path)
    beat2_map = result.get("beat2", {})
    assert "beat" in beat2_map, "beat2 must have a 'beat' category mapping"
    assert beat2_map["beat"] is not None, "beat2.beat must map to a non-null Pepper gesture"


def test_mapping_nonexistent_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset_to_pepper_map(tmp_path / "no_such_file.yaml")


def test_mapping_malformed_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- this\n- is\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level mapping"):
        load_dataset_to_pepper_map(bad)
