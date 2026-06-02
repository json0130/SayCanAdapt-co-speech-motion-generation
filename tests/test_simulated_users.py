"""Tests for src/data/simulated_users.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.data.loaders import make_synthetic_dataset
from src.data.simulated_users import (
    UserPreferenceProfile,
    UserSession,
    compute_preference_reward,
    generate_user_profiles,
    generate_user_sessions,
)


# ── generate_user_profiles ────────────────────────────────────────────────────

def test_generate_profiles_count(known_tags: list[str]) -> None:
    profiles = generate_user_profiles(np.random.default_rng(0), known_tags, num_users=10)
    assert len(profiles) == 10


def test_profile_tag_weights_cover_all_known_tags(known_tags: list[str]) -> None:
    profiles = generate_user_profiles(np.random.default_rng(0), known_tags, num_users=5)
    for p in profiles:
        assert set(p.tag_weights.keys()) == set(known_tags)


def test_profile_tag_weights_sum_to_one(known_tags: list[str]) -> None:
    profiles = generate_user_profiles(np.random.default_rng(0), known_tags, num_users=5)
    for p in profiles:
        total = sum(p.tag_weights.values())
        assert math.isclose(total, 1.0, rel_tol=1e-6), (
            f"Tag weights for {p.user_id!r} sum to {total:.6f}, expected 1.0"
        )


def test_profile_tag_weights_are_nonnegative(known_tags: list[str]) -> None:
    profiles = generate_user_profiles(np.random.default_rng(0), known_tags, num_users=10)
    for p in profiles:
        for tag, w in p.tag_weights.items():
            assert w >= 0.0, f"Negative weight {w} for tag '{tag}' in {p.user_id!r}"


def test_profiles_are_reproducible(known_tags: list[str]) -> None:
    p1 = generate_user_profiles(np.random.default_rng(42), known_tags, num_users=5)
    p2 = generate_user_profiles(np.random.default_rng(42), known_tags, num_users=5)
    for a, b in zip(p1, p2):
        assert a.user_id == b.user_id
        assert a.tag_weights == b.tag_weights


def test_profiles_differ_across_users(known_tags: list[str]) -> None:
    """Users should have different preference distributions (not all identical)."""
    profiles = generate_user_profiles(np.random.default_rng(0), known_tags, num_users=5)
    weight_sets = [tuple(sorted(p.tag_weights.items())) for p in profiles]
    assert len(set(weight_sets)) > 1, "All users have identical preferences — no personalization"


def test_profile_user_ids_are_unique(known_tags: list[str]) -> None:
    profiles = generate_user_profiles(np.random.default_rng(0), known_tags, num_users=10)
    ids = [p.user_id for p in profiles]
    assert len(ids) == len(set(ids))


def test_empty_known_tags_raises() -> None:
    with pytest.raises(ValueError, match="known_tags"):
        generate_user_profiles(np.random.default_rng(0), [], num_users=5)


def test_zero_users_raises(known_tags: list[str]) -> None:
    with pytest.raises(ValueError, match="num_users"):
        generate_user_profiles(np.random.default_rng(0), known_tags, num_users=0)


# ── generate_user_sessions ────────────────────────────────────────────────────

def _make_profile_and_pairs(
    known_tags: list[str], gesture_ids: list[str]
) -> tuple[UserPreferenceProfile, list]:
    profile = UserPreferenceProfile(
        user_id="test_user",
        tag_weights={t: 1.0 / len(known_tags) for t in known_tags},
    )
    pairs = make_synthetic_dataset(
        np.random.default_rng(1), gesture_ids, num_speakers=5, utterances_per_speaker=20
    )
    return profile, pairs


def test_generate_sessions_count(
    known_tags: list[str], gesture_ids: list[str]
) -> None:
    profile, pairs = _make_profile_and_pairs(known_tags, gesture_ids)
    sessions = generate_user_sessions(
        np.random.default_rng(7), profile, pairs, num_sessions=10, utterances_per_session=5
    )
    assert len(sessions) == 10


def test_generate_sessions_pairs_per_session(
    known_tags: list[str], gesture_ids: list[str]
) -> None:
    profile, pairs = _make_profile_and_pairs(known_tags, gesture_ids)
    sessions = generate_user_sessions(
        np.random.default_rng(7), profile, pairs, num_sessions=5, utterances_per_session=15
    )
    for s in sessions:
        assert len(s.pairs) == 15


def test_generate_sessions_user_id_propagated(
    known_tags: list[str], gesture_ids: list[str]
) -> None:
    profile, pairs = _make_profile_and_pairs(known_tags, gesture_ids)
    sessions = generate_user_sessions(np.random.default_rng(0), profile, pairs)
    for s in sessions:
        assert s.user_id == profile.user_id


def test_generate_sessions_session_indices(
    known_tags: list[str], gesture_ids: list[str]
) -> None:
    profile, pairs = _make_profile_and_pairs(known_tags, gesture_ids)
    sessions = generate_user_sessions(
        np.random.default_rng(0), profile, pairs, num_sessions=5
    )
    indices = [s.session_idx for s in sessions]
    assert indices == list(range(5))


def test_generate_sessions_reproducible(
    known_tags: list[str], gesture_ids: list[str]
) -> None:
    profile, pairs = _make_profile_and_pairs(known_tags, gesture_ids)
    s1 = generate_user_sessions(np.random.default_rng(99), profile, pairs)
    s2 = generate_user_sessions(np.random.default_rng(99), profile, pairs)
    for a, b in zip(s1, s2):
        assert [p.pair_id for p in a.pairs] == [p.pair_id for p in b.pairs]


def test_generate_sessions_empty_pairs_raises(known_tags: list[str]) -> None:
    profile = UserPreferenceProfile(user_id="u0", tag_weights={"beat": 1.0})
    with pytest.raises(ValueError, match="available_pairs"):
        generate_user_sessions(np.random.default_rng(0), profile, [])


# ── compute_preference_reward ─────────────────────────────────────────────────

def test_preference_reward_uniform_profile_returns_base_reward(known_tags: list[str]) -> None:
    """A profile with uniform weights should yield reward close to base_reward."""
    profile = UserPreferenceProfile(
        user_id="u0",
        tag_weights={t: 1.0 / len(known_tags) for t in known_tags},
    )
    for base in (0.0, 0.5, 1.0):
        result = compute_preference_reward(profile, ("beat",), base)
        assert math.isclose(result, base, rel_tol=1e-6), (
            f"Uniform profile: expected {base}, got {result}"
        )


def test_preference_reward_liked_tag_boosts_reward(known_tags: list[str]) -> None:
    """A user who strongly prefers 'beat' should get a higher reward for beat gestures."""
    n = len(known_tags)
    loved_weights = {t: 0.01 / (n - 1) for t in known_tags}
    loved_weights["beat"] = 0.99  # extremely likes beat
    profile = UserPreferenceProfile(user_id="u0", tag_weights=loved_weights)

    uniform_profile = UserPreferenceProfile(
        user_id="u1", tag_weights={t: 1.0 / n for t in known_tags}
    )
    r_loved = compute_preference_reward(profile, ("beat",), 0.5)
    r_uniform = compute_preference_reward(uniform_profile, ("beat",), 0.5)
    assert r_loved >= r_uniform, "A user who loves 'beat' should score at least as high"


def test_preference_reward_disliked_tag_reduces_reward(known_tags: list[str]) -> None:
    """A user who dislikes 'beat' should get a lower reward for beat gestures."""
    n = len(known_tags)
    hated_weights = {t: (1.0 - 0.001) / (n - 1) for t in known_tags if t != "beat"}
    hated_weights["beat"] = 0.001
    profile = UserPreferenceProfile(user_id="u0", tag_weights=hated_weights)

    uniform_profile = UserPreferenceProfile(
        user_id="u1", tag_weights={t: 1.0 / n for t in known_tags}
    )
    r_hated = compute_preference_reward(profile, ("beat",), 0.5)
    r_uniform = compute_preference_reward(uniform_profile, ("beat",), 0.5)
    assert r_hated <= r_uniform, "A user who dislikes 'beat' should score lower"


def test_preference_reward_clamped_to_unit_interval(known_tags: list[str]) -> None:
    """Reward must always be in [0, 1] regardless of profile or base reward."""
    n = len(known_tags)
    extreme_weights = {t: 0.0 for t in known_tags}
    extreme_weights[known_tags[0]] = 1.0
    profile = UserPreferenceProfile(user_id="u0", tag_weights=extreme_weights)

    for base in (0.0, 0.5, 1.0):
        r = compute_preference_reward(profile, tuple(known_tags), base)
        assert 0.0 <= r <= 1.0, f"Reward {r} out of [0, 1] for base={base}"


def test_preference_reward_empty_tags_returns_base_reward(known_tags: list[str]) -> None:
    profile = UserPreferenceProfile(
        user_id="u0", tag_weights={t: 1.0 / len(known_tags) for t in known_tags}
    )
    assert compute_preference_reward(profile, (), 0.7) == 0.7
