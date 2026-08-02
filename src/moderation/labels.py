"""Category vocabulary and the mapping from raw model labels onto it.

The two models in the cascade disagree about what a label even is. The dialect
model emits human-readable category names with inconsistent casing and spacing;
the toxicity model emits `LABEL_0` / `LABEL_1`. Normalising both into one
vocabulary here, rather than string-matching at each call site, is what keeps
the pipeline logic readable and testable.
"""

from __future__ import annotations

NEUTRAL = "Neutral"
OFFENSIVE = "Offensive"
RACISM = "Racism"
SEXISM = "Sexism"
RELIGIOUS = "Religious Discrimination"
EXTREMISM = "Extremism"

CATEGORIES = (NEUTRAL, OFFENSIVE, RACISM, SEXISM, RELIGIOUS, EXTREMISM)

# Raw label (lowercased, stripped) -> canonical category.
_ALIASES = {
    "neutral": NEUTRAL,
    "normal": NEUTRAL,
    "none": NEUTRAL,
    "not_hate": NEUTRAL,
    "label_0": NEUTRAL,
    "offensive": OFFENSIVE,
    "abusive": OFFENSIVE,
    "hate": OFFENSIVE,
    "racism": RACISM,
    "racist": RACISM,
    "nationality": RACISM,
    "ethnicity": RACISM,
    "sexism": SEXISM,
    "sexist": SEXISM,
    "gender": SEXISM,
    "misogyny": SEXISM,
    "religious discrimination": RELIGIOUS,
    "religious_discrimination": RELIGIOUS,
    "religion": RELIGIOUS,
    "religious": RELIGIOUS,
    "extremism": EXTREMISM,
    "violence": EXTREMISM,
    "threat": EXTREMISM,
    "label_1": EXTREMISM,
}


def normalise(raw_label: str) -> str:
    """Map a raw model label onto the canonical vocabulary.

    Unknown labels fall back to `Offensive` rather than `Neutral`: if a model
    flagged something under a name we do not recognise, treating that as clean
    is the more dangerous of the two errors.
    """
    key = str(raw_label).strip().lower().replace("-", "_")
    if key in _ALIASES:
        return _ALIASES[key]
    key_spaced = key.replace("_", " ")
    if key_spaced in _ALIASES:
        return _ALIASES[key_spaced]
    return OFFENSIVE


def is_harmful(category: str) -> bool:
    return category != NEUTRAL
