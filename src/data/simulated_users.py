"""Synthetic user generation for the simulated long-term evaluation protocol.

SPEC §7: Generate N synthetic users, each with a preference profile (a probability
distribution over gesture tags) and M utterances drawn from the dataset per session.
The preference profile creates per-user variation so the Adapt term has a learnable
signal; without it every user would look identical.

All randomness flows through the caller-supplied numpy.random.Generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data.loaders import UtteranceGesturePair


@dataclass
class UserPreferenceProfile:
    """Per-user preference distribution over gesture tags.

    ``tag_weights`` is a probability distribution (values sum to 1.0, all >= 0)
    over gesture tags.  A high weight for a tag means the user tends to respond
    positively to gestures of that type.  Generated via Dirichlet sampling so
    each simulated user has distinct preferences.
    """

    user_id: str
    tag_weights: dict[str, float]


@dataclass
class UserSession:
    """One session of utterance-gesture pairs for a specific simulated user.

    Sessions are indexed from 0.  The same pair may appear in multiple sessions
    (sampling with replacement) to simulate repeated long-term interactions.
    """

    user_id: str
    session_idx: int
    pairs: list[UtteranceGesturePair] = field(default_factory=list)


def generate_user_profiles(
    rng: np.random.Generator,
    known_tags: list[str],
    num_users: int = 20,
    concentration: float = 0.5,
) -> list[UserPreferenceProfile]:
    """Sample per-user preference distributions using a Dirichlet distribution.

    Parameters
    ----------
    rng:
        Seeded Generator; owned by the caller.
    known_tags:
        The gesture tags present in the library (e.g. ["greeting", "beat", ...]).
        Must not be empty.
    num_users:
        Number of synthetic users to generate.
    concentration:
        Dirichlet concentration parameter alpha (same for all tags).
        < 1 → spiky (strong individual preferences); = 1 → uniform Dirichlet;
        > 1 → flatter distributions.  Default 0.5 gives realistic variation.
    """
    if not known_tags:
        raise ValueError("known_tags must not be empty")
    if num_users < 1:
        raise ValueError("num_users must be >= 1")
    if concentration <= 0.0:
        raise ValueError(f"concentration must be positive, got {concentration}")

    alpha = [concentration] * len(known_tags)
    profiles: list[UserPreferenceProfile] = []
    for i in range(num_users):
        weights_arr = rng.dirichlet(alpha)
        tag_weights = {tag: float(w) for tag, w in zip(known_tags, weights_arr)}
        profiles.append(UserPreferenceProfile(user_id=f"sim_user_{i:03d}", tag_weights=tag_weights))
    return profiles


def generate_user_sessions(
    rng: np.random.Generator,
    profile: UserPreferenceProfile,
    available_pairs: list[UtteranceGesturePair],
    num_sessions: int = 10,
    utterances_per_session: int = 50,
) -> list[UserSession]:
    """Sample utterances from *available_pairs* to fill a user's session sequence.

    Sampling is with replacement so users can encounter the same utterance across
    different sessions, simulating long-term repeated interactions.

    Parameters
    ----------
    rng:
        Seeded Generator; owned by the caller.
    profile:
        The user whose sessions are being generated.
    available_pairs:
        Pool of utterance-gesture pairs to draw from (typically the training split).
    num_sessions:
        How many sessions the user has.
    utterances_per_session:
        Number of utterance-gesture pairs per session.

    Raises
    ------
    ValueError
        If *available_pairs* is empty.
    """
    if not available_pairs:
        raise ValueError("available_pairs must not be empty")
    if num_sessions < 1:
        raise ValueError("num_sessions must be >= 1")
    if utterances_per_session < 1:
        raise ValueError("utterances_per_session must be >= 1")

    total = num_sessions * utterances_per_session
    indices: list[int] = rng.integers(0, len(available_pairs), size=total).tolist()

    sessions: list[UserSession] = []
    for sess_idx in range(num_sessions):
        start = sess_idx * utterances_per_session
        end = start + utterances_per_session
        sess_pairs = [available_pairs[indices[k]] for k in range(start, end)]
        sessions.append(
            UserSession(
                user_id=profile.user_id,
                session_idx=sess_idx,
                pairs=sess_pairs,
            )
        )
    return sessions


def compute_preference_reward(
    profile: UserPreferenceProfile,
    gesture_tags: tuple[str, ...],
    base_reward: float,
) -> float:
    """Scale *base_reward* by the user's preference for the gesture's tags.

    Used by SimulatedEngagementFeedback (Step 6) to personalise the reward signal.

    Formula: ``base_reward * (user_pref / uniform_pref)``, clamped to [0, 1].
    - When a tag's weight equals the uniform baseline (1/N tags), scale = 1 →
      reward equals base_reward.
    - A tag the user likes (weight >> 1/N) boosts the reward toward 1.0.
    - A tag the user dislikes (weight << 1/N) suppresses the reward toward 0.0.

    ASSUMPTION: we use the *maximum* preference across all of the gesture's tags,
    so a gesture with at least one liked tag gets the benefit of the doubt.
    See BUILD_NOTES.md.

    Parameters
    ----------
    profile:
        The user's preference profile.
    gesture_tags:
        Tags of the selected Pepper gesture.
    base_reward:
        Raw reward from the ground-truth comparison (0.0, 0.5, or 1.0).
    """
    if not gesture_tags or not profile.tag_weights:
        return base_reward

    n = len(profile.tag_weights)
    uniform = 1.0 / n  # expected weight per tag in a neutral profile
    user_pref = max(profile.tag_weights.get(tag, 0.0) for tag in gesture_tags)
    scale = user_pref / uniform if uniform > 0.0 else 1.0
    return min(1.0, max(0.0, base_reward * scale))
