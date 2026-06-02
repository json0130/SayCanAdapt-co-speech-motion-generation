"""Tests for speaker-disjoint splits — the SPEC §6 non-negotiable.

The test test_no_speaker_in_multiple_splits is the primary guardrail.
Do not weaken it.  If it fails, fix the split logic, not the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.loaders import UtteranceGesturePair, make_synthetic_dataset
from src.data.splits import DatasetSplit, assert_speaker_disjoint, make_speaker_disjoint_splits


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_pairs(
    num_speakers: int = 15,
    utterances_per_speaker: int = 20,
    seed: int = 0,
) -> list[UtteranceGesturePair]:
    rng = np.random.default_rng(seed)
    gesture_ids = ["wave_right", "nod_yes", "beat_small", "point_forward", "shrug"]
    return make_synthetic_dataset(rng, gesture_ids, num_speakers, utterances_per_speaker)


# ── THE primary guardrail ─────────────────────────────────────────────────────

def test_no_speaker_in_multiple_splits() -> None:
    """SPEC §6: no speaker may appear in more than one split.  This must always pass."""
    pairs = _make_pairs(num_speakers=20)
    split = make_speaker_disjoint_splits(pairs, np.random.default_rng(99))

    train_spk = {p.speaker_id for p in split.train}
    val_spk = {p.speaker_id for p in split.val}
    test_spk = {p.speaker_id for p in split.test}

    assert not (train_spk & val_spk), f"Speaker overlap train∩val: {train_spk & val_spk}"
    assert not (train_spk & test_spk), f"Speaker overlap train∩test: {train_spk & test_spk}"
    assert not (val_spk & test_spk), f"Speaker overlap val∩test: {val_spk & test_spk}"


def test_assert_speaker_disjoint_passes_on_valid_split() -> None:
    """assert_speaker_disjoint must not raise on a correctly produced split."""
    split = make_speaker_disjoint_splits(_make_pairs(), np.random.default_rng(7))
    assert_speaker_disjoint(split)  # must not raise


def test_assert_speaker_disjoint_raises_on_violation() -> None:
    """assert_speaker_disjoint must raise AssertionError when a speaker is in two splits."""
    # Manually construct a bad split with shared speakers
    bad_pair = UtteranceGesturePair(
        pair_id="x", speaker_id="shared_spk", utterance="hello",
        gesture_id="wave_right", dataset="synthetic", dataset_gesture_label="greeting",
    )
    bad_split = DatasetSplit(
        train=[bad_pair],
        val=[bad_pair],  # same speaker → violation
        test=[
            UtteranceGesturePair(
                pair_id="y", speaker_id="other_spk", utterance="bye",
                gesture_id="nod_yes", dataset="synthetic", dataset_gesture_label="greeting",
            )
        ],
    )
    with pytest.raises(AssertionError, match="shared_spk"):
        assert_speaker_disjoint(bad_split)


# ── coverage and totals ───────────────────────────────────────────────────────

def test_all_pairs_appear_in_exactly_one_split() -> None:
    """No pair should be lost or duplicated across splits."""
    pairs = _make_pairs()
    split = make_speaker_disjoint_splits(pairs, np.random.default_rng(1))
    all_ids = (
        [p.pair_id for p in split.train]
        + [p.pair_id for p in split.val]
        + [p.pair_id for p in split.test]
    )
    assert len(all_ids) == len(pairs), "Some pairs were lost in the split"
    assert len(set(all_ids)) == len(pairs), "Some pairs were duplicated in the split"


def test_all_speakers_assigned_to_exactly_one_split() -> None:
    pairs = _make_pairs()
    split = make_speaker_disjoint_splits(pairs, np.random.default_rng(2))

    all_speakers_in = (
        {p.speaker_id for p in split.train}
        | {p.speaker_id for p in split.val}
        | {p.speaker_id for p in split.test}
    )
    original_speakers = {p.speaker_id for p in pairs}
    assert all_speakers_in == original_speakers


# ── approximate fractions ─────────────────────────────────────────────────────

def test_split_fractions_are_approximately_correct() -> None:
    """Train should be largest; test and val should be present."""
    pairs = _make_pairs(num_speakers=30, utterances_per_speaker=10)
    split = make_speaker_disjoint_splits(pairs, np.random.default_rng(3))
    n = len(pairs)
    assert len(split.train) > len(split.val), "train should be larger than val"
    assert len(split.train) > len(split.test), "train should be larger than test"
    assert len(split.val) > 0, "val split must not be empty"
    assert len(split.test) > 0, "test split must not be empty"
    # train should be roughly 70% ± 15%
    train_frac = len(split.train) / n
    assert 0.50 <= train_frac <= 0.90, f"train_frac={train_frac:.2f} out of expected range"


# ── reproducibility ───────────────────────────────────────────────────────────

def test_split_is_reproducible_with_same_seed() -> None:
    pairs = _make_pairs(seed=77)
    split_a = make_speaker_disjoint_splits(pairs, np.random.default_rng(55))
    split_b = make_speaker_disjoint_splits(pairs, np.random.default_rng(55))

    assert [p.pair_id for p in split_a.train] == [p.pair_id for p in split_b.train]
    assert [p.pair_id for p in split_a.val] == [p.pair_id for p in split_b.val]
    assert [p.pair_id for p in split_a.test] == [p.pair_id for p in split_b.test]


def test_different_seeds_can_give_different_splits() -> None:
    pairs = _make_pairs(num_speakers=20)
    split_a = make_speaker_disjoint_splits(pairs, np.random.default_rng(1))
    split_b = make_speaker_disjoint_splits(pairs, np.random.default_rng(2))
    # Different seeds should (very likely) produce different assignments
    ids_a = {p.pair_id for p in split_a.train}
    ids_b = {p.pair_id for p in split_b.train}
    assert ids_a != ids_b, "Different seeds produced identical splits (extremely unlikely)"


# ── error cases ───────────────────────────────────────────────────────────────

def test_empty_pairs_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        make_speaker_disjoint_splits([], np.random.default_rng(0))


def test_too_few_speakers_raises_value_error() -> None:
    pairs = _make_pairs(num_speakers=2, utterances_per_speaker=5)
    with pytest.raises(ValueError, match="3"):
        make_speaker_disjoint_splits(pairs, np.random.default_rng(0))


def test_invalid_fractions_raise_value_error() -> None:
    pairs = _make_pairs()
    with pytest.raises(ValueError):
        make_speaker_disjoint_splits(pairs, np.random.default_rng(0), train_frac=0.9, val_frac=0.2)
