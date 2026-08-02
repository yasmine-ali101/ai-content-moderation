"""Tests for label normalisation.

The two models in the cascade emit incompatible label vocabularies, one uses
category names with inconsistent casing, the other uses `LABEL_0`/`LABEL_1`.
Getting this mapping wrong silently mislabels everything downstream.
"""

import pytest

from moderation import labels


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("neutral", labels.NEUTRAL),
        ("Neutral", labels.NEUTRAL),
        ("  NEUTRAL  ", labels.NEUTRAL),
        ("LABEL_0", labels.NEUTRAL),
        ("label_0", labels.NEUTRAL),
        ("offensive", labels.OFFENSIVE),
        ("racism", labels.RACISM),
        ("sexism", labels.SEXISM),
        ("religious discrimination", labels.RELIGIOUS),
        ("religious_discrimination", labels.RELIGIOUS),
        ("Religious-Discrimination", labels.RELIGIOUS),
        ("LABEL_1", labels.EXTREMISM),
    ],
)
def test_known_labels_map_to_the_canonical_vocabulary(raw, expected):
    assert labels.normalise(raw) == expected


def test_unknown_labels_fail_closed_to_offensive_not_neutral():
    """An unrecognised label means a model flagged *something*.

    Defaulting to Neutral would let it through, the more dangerous of the two
    possible errors for a moderation system.
    """
    assert labels.normalise("some_unseen_category") == labels.OFFENSIVE
    assert labels.normalise("LABEL_7") == labels.OFFENSIVE


def test_every_normalised_label_is_in_the_declared_vocabulary():
    for raw in ["neutral", "offensive", "racism", "sexism", "religion", "LABEL_1", "junk"]:
        assert labels.normalise(raw) in labels.CATEGORIES


def test_is_harmful_is_true_for_everything_except_neutral():
    assert not labels.is_harmful(labels.NEUTRAL)
    for category in labels.CATEGORIES:
        if category != labels.NEUTRAL:
            assert labels.is_harmful(category)
