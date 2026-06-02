"""Speaker-disjoint train/val/test splits.

SPEC §6 non-negotiable: no speaker may appear in more than one split.
The unit test in tests/test_data_splits.py enforces this.  Do not relax it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.loaders import UtteranceGesturePair


@dataclass
class DatasetSplit:
    """Container for speaker-disjoint train / val / test partitions."""

    train: list[UtteranceGesturePair]
    val: list[UtteranceGesturePair]
    test: list[UtteranceGesturePair]

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


def make_speaker_disjoint_splits(
    pairs: list[UtteranceGesturePair],
    rng: np.random.Generator,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> DatasetSplit:
    """Partition *pairs* so each speaker appears in exactly one split.

    Speakers are shuffled with *rng* (for reproducibility) and then assigned to
    train / val / test in proportion to *train_frac* / *val_frac* /
    (1 - train_frac - val_frac).

    Parameters
    ----------
    pairs:
        Full list of utterance-gesture pairs.  Must not be empty.
    rng:
        A seeded numpy Generator owned by the caller.
    train_frac:
        Fraction of *speakers* (not pairs) to assign to train.
    val_frac:
        Fraction of *speakers* to assign to val.  The remainder goes to test.

    Raises
    ------
    ValueError
        If *pairs* is empty, fractions are invalid, or there are fewer than 3
        distinct speakers (cannot make all three splits non-empty).
    """
    if not pairs:
        raise ValueError("pairs must not be empty")
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac must be < 1.0, got {train_frac + val_frac:.4f}"
        )

    speaker_list: list[str] = sorted({p.speaker_id for p in pairs})
    n = len(speaker_list)
    if n < 3:
        raise ValueError(
            f"Need at least 3 distinct speakers for a three-way split, got {n}. "
            "Add more speakers or use a different split strategy."
        )

    # Shuffle speakers reproducibly
    perm = rng.permutation(n)
    shuffled: list[str] = [speaker_list[int(i)] for i in perm]

    # Integer speaker counts (floor); test gets the remainder to avoid losing any
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    n_test = n - n_train - n_val

    # If rounding left no test speakers, shave one from val
    if n_test < 1:
        n_val -= 1
        n_test = n - n_train - n_val
    if n_val < 1 or n_test < 1:
        raise ValueError(
            f"Cannot satisfy train_frac={train_frac}, val_frac={val_frac} "
            f"with only {n} speakers while keeping all three splits non-empty."
        )

    train_spk: set[str] = set(shuffled[:n_train])
    val_spk: set[str] = set(shuffled[n_train : n_train + n_val])
    test_spk: set[str] = set(shuffled[n_train + n_val :])

    train = [p for p in pairs if p.speaker_id in train_spk]
    val = [p for p in pairs if p.speaker_id in val_spk]
    test = [p for p in pairs if p.speaker_id in test_spk]

    return DatasetSplit(train=train, val=val, test=test)


def assert_speaker_disjoint(split: DatasetSplit) -> None:
    """Raise AssertionError if any speaker appears in more than one split.

    Call this in tests and in the eval runner to catch regressions early.
    """
    train_spk = {p.speaker_id for p in split.train}
    val_spk = {p.speaker_id for p in split.val}
    test_spk = {p.speaker_id for p in split.test}

    tv = train_spk & val_spk
    tt = train_spk & test_spk
    vt = val_spk & test_spk

    violations: list[str] = []
    if tv:
        violations.append(f"train∩val: {sorted(tv)}")
    if tt:
        violations.append(f"train∩test: {sorted(tt)}")
    if vt:
        violations.append(f"val∩test: {sorted(vt)}")

    if violations:
        raise AssertionError(
            "Speaker-disjoint split violated — speakers appear in multiple splits: "
            + "; ".join(violations)
        )
