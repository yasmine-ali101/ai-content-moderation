"""Tests for the cascade's routing logic.

Both classifiers are stubbed, so these run in milliseconds with no model
download — the routing decisions are what matter here, not model quality.
"""

import pytest

from moderation import labels
from moderation.config import settings
from moderation.pipeline import ModerationPipeline


class StubClassifier:
    """Returns a canned label/score, recording what it was asked about."""

    def __init__(self, label: str, score: float = 0.99):
        self.label = label
        self.score = score
        self.calls: list = []

    def __call__(self, text):
        self.calls.append(text)
        if isinstance(text, list):
            return [{"label": self.label, "score": self.score} for _ in text]
        return [{"label": self.label, "score": self.score}]


def build(hate_label, hate_score=0.99, toxic_label="LABEL_0", toxic_score=0.99):
    hate = StubClassifier(hate_label, hate_score)
    toxic = StubClassifier(toxic_label, toxic_score)
    return ModerationPipeline(hate_clf=hate, toxicity_clf=toxic), hate, toxic


def test_a_confident_harmful_verdict_stops_at_stage_one():
    """Stage 1's category detail is the reason for using it; don't dilute it."""
    pipeline, hate, toxic = build("Sexism")

    verdict = pipeline.moderate("...")

    assert verdict.category == labels.SEXISM
    assert verdict.stage == "hate"
    assert toxic.calls == []  # stage 2 never consulted


def test_a_neutral_verdict_escalates_to_the_toxicity_stage():
    """The recall gap the cascade exists to close."""
    pipeline, hate, toxic = build("neutral", toxic_label="LABEL_1", toxic_score=0.95)

    verdict = pipeline.moderate("implicitly hostile text")

    assert verdict.category == labels.EXTREMISM
    assert verdict.stage == "toxicity"
    assert verdict.escalated is True
    assert len(toxic.calls) == 1


def test_a_weak_toxicity_signal_does_not_overturn_a_neutral_verdict():
    """The notebook accepted any LABEL_1, including a 0.51 coin flip.

    A broad multilingual model misreads dialect; requiring it to clear a
    threshold is what stops that becoming a false positive.
    """
    pipeline, _, _ = build("neutral", toxic_label="LABEL_1", toxic_score=0.55)

    verdict = pipeline.moderate("perfectly fine text")

    assert verdict.category == labels.NEUTRAL
    assert verdict.action == "allow"


def test_a_toxicity_signal_above_the_threshold_does_overturn_it():
    pipeline, _, _ = build("neutral", toxic_label="LABEL_1", toxic_score=0.95)

    assert pipeline.moderate("...").category == labels.EXTREMISM


def test_clean_text_is_allowed_and_reported_as_neutral():
    pipeline, _, _ = build("neutral", toxic_label="LABEL_0")

    verdict = pipeline.moderate("الجو حلو النهاردة")

    assert verdict.category == labels.NEUTRAL
    assert verdict.action == "allow"
    assert verdict.is_harmful is False


def test_offensive_text_is_masked_and_keeps_the_original():
    pipeline, _, _ = build("Offensive")

    verdict = pipeline.moderate("انت غبي")

    assert verdict.action == "mask"
    assert verdict.masked_text is not None
    assert verdict.text == "انت غبي"  # original preserved for audit


def test_sexism_is_flagged_rather_than_masked():
    """Sexism is usually carried by sentence structure, not a removable word.

    Masking would produce a mutilated sentence that still says the same thing.
    """
    pipeline, _, _ = build("Sexism")

    verdict = pipeline.moderate("الستات مكانهم المطبخ")

    assert verdict.action == "flag"
    assert verdict.masked_text is None


def test_masking_only_redacts_tokens_above_the_mask_threshold():
    hate = StubClassifier("Offensive", score=settings.mask_threshold - 0.1)
    pipeline = ModerationPipeline(hate_clf=hate, toxicity_clf=StubClassifier("LABEL_0"))

    assert pipeline.mask("كلمة تانية") == "كلمة تانية"  # below threshold -> untouched


def test_masking_redacts_confident_tokens():
    hate = StubClassifier("Offensive", score=0.99)
    pipeline = ModerationPipeline(hate_clf=hate, toxicity_clf=StubClassifier("LABEL_0"))

    masked = pipeline.mask("كلمة تانية")

    assert "*" in masked
    assert "كلمة" not in masked


def test_masking_an_empty_string_is_a_no_op():
    pipeline, _, _ = build("Offensive")

    assert pipeline.mask("") == ""
    assert pipeline.mask("   ").strip() == ""


def test_every_verdict_carries_an_arabic_explanation():
    for label in ("Offensive", "Sexism", "Racism", "neutral"):
        pipeline, _, _ = build(label)
        verdict = pipeline.moderate("...")
        assert verdict.explanation
        # Explanations are addressed to the audience being moderated.
        assert any("؀" <= ch <= "ۿ" for ch in verdict.explanation)


def test_verdict_serialises_to_a_json_safe_dict():
    pipeline, _, _ = build("Racism")

    payload = pipeline.moderate("...").as_dict()

    assert payload["category"] == labels.RACISM
    assert payload["action"] == "mask"
    assert isinstance(payload["confidence"], float)


def test_batch_moderation_returns_one_verdict_per_input():
    pipeline, _, _ = build("neutral", toxic_label="LABEL_0")

    verdicts = pipeline.moderate_batch(["a", "b", "c"])

    assert len(verdicts) == 3
