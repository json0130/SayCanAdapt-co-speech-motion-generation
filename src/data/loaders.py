"""Dataset loaders and the UtteranceGesturePair type.

Real dataset loading is stubbed for BEAT2 and TED-Gesture (they return empty lists
with a warning when the data directory is absent).  For unit tests and offline
development, use make_synthetic_dataset(), which is fully self-contained and
deterministic via the caller-supplied numpy.random.Generator.

SPEC §7: every dataset gesture is mapped to a Pepper library entry via
data/library/dataset_to_pepper.yaml; unmapped entries (pepper_id=None) are excluded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_log = logging.getLogger(__name__)

# Template utterances for synthetic data generation, keyed by broad gesture tag.
# Each entry represents a plausible sentence that would accompany that gesture type.
_TEMPLATES: list[tuple[str, str]] = [
    # greeting
    ("greeting", "Hello, nice to meet you."),
    ("greeting", "Good morning! How are you today?"),
    ("greeting", "Welcome, I'm glad you're here."),
    ("greeting", "Hi there! Let's get started."),
    ("greeting", "Good day! It's wonderful to see you."),
    # deictic
    ("deictic", "Look over there — that's what I mean."),
    ("deictic", "Can you see that particular spot?"),
    ("deictic", "I'm referring to this specific area."),
    ("deictic", "That, right there, is the key element."),
    ("deictic", "Over in that direction, you can see it."),
    # beat
    ("beat", "The key point is this."),
    ("beat", "As I was saying, we need to consider this carefully."),
    ("beat", "Furthermore, the evidence clearly suggests otherwise."),
    ("beat", "In addition to that, we must also factor in..."),
    ("beat", "Let me emphasize this crucial detail once more."),
    # iconic
    ("iconic", "It was this big — absolutely enormous."),
    ("iconic", "Just a tiny little thing, about this much."),
    ("iconic", "Spread across a wide area like this."),
    ("iconic", "About this quantity, roughly speaking."),
    ("iconic", "Imagine something this size, held in your hands."),
    # affect
    ("affect", "I completely agree with what you're saying."),
    ("affect", "No, that's not quite right, actually."),
    ("affect", "I'm not entirely sure about that one."),
    ("affect", "Absolutely, that's exactly what I think too."),
    ("affect", "Hmm, let me think about that for a moment."),
    # discourse
    ("discourse", "On one hand, we have this strong argument."),
    ("discourse", "First, let me explain the broader context here."),
    ("discourse", "Second, the implications become quite clear."),
    ("discourse", "To summarize everything we've covered today."),
    ("discourse", "In conclusion, the results clearly demonstrate..."),
    # rest
    ("rest", "Take a moment to consider what we've discussed."),
    ("rest", "Let that sink in for just a second."),
    ("rest", "Processing everything we've just covered together."),
    ("rest", "A brief pause here before we continue."),
    ("rest", "Thinking carefully about the next step forward."),
]


@dataclass(frozen=True)
class UtteranceGesturePair:
    """A single (utterance, gesture) pair after mapping to the Pepper library."""

    pair_id: str
    speaker_id: str
    utterance: str
    gesture_id: str           # Pepper library gesture_id (post-mapping)
    dataset: str              # "beat2", "ted_gesture", "synthetic", …
    dataset_gesture_label: str  # original category label before mapping (for reference)


def load_dataset_to_pepper_map(path: str | Path) -> dict[str, dict[str, str | None]]:
    """Parse data/library/dataset_to_pepper.yaml into a nested mapping dict.

    Returns
    -------
    dict mapping ``dataset_name → {dataset_gesture_label → pepper_id | None}``
    """
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"dataset_to_pepper.yaml at {str(path)!r} must be a top-level mapping"
        )
    result: dict[str, dict[str, str | None]] = {}
    for dataset_name, label_map in raw.items():
        if label_map is None:
            result[str(dataset_name)] = {}
            continue
        if not isinstance(label_map, dict):
            raise ValueError(
                f"Mapping for dataset '{dataset_name}' must be a dict, got {type(label_map)}"
            )
        result[str(dataset_name)] = {
            str(k): (str(v) if v is not None else None)
            for k, v in label_map.items()
        }
    return result


def make_synthetic_dataset(
    rng: np.random.Generator,
    gesture_ids: list[str],
    num_speakers: int = 10,
    utterances_per_speaker: int = 50,
    dataset_name: str = "synthetic",
) -> list[UtteranceGesturePair]:
    """Generate a fully reproducible synthetic dataset for testing and development.

    All randomness flows through *rng* so the result is byte-identical given the
    same generator state.

    Parameters
    ----------
    rng:
        A seeded numpy Generator.  The caller owns this generator; do not create
        one inside this function.
    gesture_ids:
        The Pepper gesture IDs to sample from uniformly as ground-truth gestures.
    num_speakers:
        Number of distinct synthetic speakers (each becomes a separate split unit).
    utterances_per_speaker:
        How many utterance-gesture pairs each speaker contributes.
    dataset_name:
        Recorded in each pair's ``dataset`` field.
    """
    if not gesture_ids:
        raise ValueError("gesture_ids must not be empty")
    if num_speakers < 1:
        raise ValueError("num_speakers must be >= 1")
    if utterances_per_speaker < 1:
        raise ValueError("utterances_per_speaker must be >= 1")

    n_templates = len(_TEMPLATES)
    n_gestures = len(gesture_ids)
    pairs: list[UtteranceGesturePair] = []

    for spk_idx in range(num_speakers):
        speaker_id = f"speaker_{spk_idx:03d}"
        tmpl_idxs: list[int] = rng.integers(
            0, n_templates, size=utterances_per_speaker
        ).tolist()
        gest_idxs: list[int] = rng.integers(
            0, n_gestures, size=utterances_per_speaker
        ).tolist()

        for utt_idx in range(utterances_per_speaker):
            tag, utterance = _TEMPLATES[tmpl_idxs[utt_idx]]
            gesture_id = gesture_ids[gest_idxs[utt_idx]]
            pairs.append(
                UtteranceGesturePair(
                    pair_id=f"{dataset_name}_{speaker_id}_{utt_idx:04d}",
                    speaker_id=speaker_id,
                    utterance=utterance,
                    gesture_id=gesture_id,
                    dataset=dataset_name,
                    dataset_gesture_label=tag,
                )
            )

    return pairs


def load_beat2(
    data_dir: str | Path,
    mapping: dict[str, str | None],
) -> list[UtteranceGesturePair]:
    """Load the BEAT2 dataset and apply the Pepper gesture mapping.

    Returns an empty list with a warning if *data_dir* does not exist or the
    loader has not yet been implemented.  Unmapped labels (None values in *mapping*)
    are silently excluded.
    """
    resolved = Path(data_dir)
    if not resolved.exists():
        _log.warning(
            "BEAT2 data directory not found at '%s' — returning empty list", resolved
        )
        return []
    # TODO: parse BEAT2 JSON/CSV files from resolved and apply mapping
    _log.warning("BEAT2 file-level parsing not yet implemented — returning empty list")
    return []


def load_ted_gesture(
    data_dir: str | Path,
    mapping: dict[str, str | None],
) -> list[UtteranceGesturePair]:
    """Load the TED-Gesture dataset and apply the Pepper gesture mapping.

    Returns an empty list with a warning if *data_dir* does not exist.
    """
    resolved = Path(data_dir)
    if not resolved.exists():
        _log.warning(
            "TED-Gesture data directory not found at '%s' — returning empty list", resolved
        )
        return []
    _log.warning("TED-Gesture file-level parsing not yet implemented — returning empty list")
    return []
