"""The two-stage moderation cascade.

    text ──> Stage 1: dialect hate-speech classifier (MARBERTv2)
               │
               ├── harmful category  ──────────────> verdict (with category)
               │
               └── "Neutral" ──> Stage 2: multilingual toxicity classifier
                                    │
                                    ├── toxic  ──> verdict: Extremism
                                    └── clean  ──> verdict: Neutral

Why a cascade rather than one model, or an ensemble vote:

The notebook's own finding was that the dialect model "detects hate/offensive
speech well but if the sentence does not contain a hate *word*, it's not
detected." That is a recall gap on implicit hostility, not a precision problem.
Stacking a broad toxicity model behind it targets exactly that gap: stage 1
keeps its precise category labels, and stage 2 only ever gets consulted on text
stage 1 already believes is clean. An ensemble vote would instead dilute stage 1's
category information, which is the thing that makes the output actionable.

Two changes from the notebook's version:

1. **Low-confidence neutrals escalate.** The notebook escalated only on a hard
   `label == "neutral"` check, so a 0.34-confidence neutral was treated as
   settled. Confidence is now part of the routing decision.
2. **Stage 2 needs to clear a threshold to overturn stage 1.** The notebook
   accepted any `LABEL_1`, including a 0.51 coin-flip, which is how a broad
   multilingual model produces false positives on dialect it misreads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import labels
from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class Verdict:
    """The result of moderating one piece of text."""

    text: str
    category: str
    confidence: float
    stage: str                      # "hate" | "toxicity"
    action: str                     # "allow" | "flag" | "mask"
    explanation: str = ""
    masked_text: str | None = None
    escalated: bool = False         # did stage 1's neutral get re-examined?

    @property
    def is_harmful(self) -> bool:
        return labels.is_harmful(self.category)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "stage": self.stage,
            "action": self.action,
            "escalated": self.escalated,
            "explanation": self.explanation,
            "masked_text": self.masked_text,
        }


# User-facing rationale, in Arabic, matching the audience being moderated.
EXPLANATIONS = {
    labels.RACISM: (
        "هذا الكلام يحتوي على تمييز عنصري، وهذا غير مقبول لأنه يقلل من قيمة الناس "
        "بناءً على عرقهم أو جنسيتهم."
    ),
    labels.SEXISM: (
        "هذا الكلام يتضمن تمييزًا على أساس الجنس، وهو غير مقبول لأن الجميع يستحقون "
        "المساواة والاحترام."
    ),
    labels.RELIGIOUS: (
        "هذا الكلام يسيء إلى معتقدات دينية، وهذا غير مقبول لأن حرية المعتقد حق للجميع."
    ),
    labels.OFFENSIVE: (
        "هذا الكلام مسيء وقد يجرح مشاعر الآخرين. استخدام لغة محترمة يساعد في التواصل "
        "بشكل أفضل."
    ),
    labels.EXTREMISM: (
        "هذا الكلام يتضمن أفكارًا متطرفة، وهذا غير مقبول لأنه يشجع على العنف وعدم التسامح."
    ),
    labels.NEUTRAL: "الجملة سليمة ولا تحتوي على إساءة.",
}


class ModerationPipeline:
    """Loads both classifiers lazily and routes text through the cascade."""

    def __init__(self, hate_clf=None, toxicity_clf=None) -> None:
        self._hate_clf = hate_clf
        self._toxicity_clf = toxicity_clf

    # --- model loading -----------------------------------------------------

    @property
    def hate_clf(self):
        if self._hate_clf is None:
            self._hate_clf = self._load(settings.hate_model)
        return self._hate_clf

    @property
    def toxicity_clf(self):
        if self._toxicity_clf is None:
            self._toxicity_clf = self._load(settings.toxicity_model)
        return self._toxicity_clf

    @staticmethod
    def _load(model_name: str):
        from transformers import pipeline as hf_pipeline

        logger.info("Loading %s", model_name)
        return hf_pipeline(
            "text-classification", model=model_name, device=settings.device
        )

    # --- cascade -----------------------------------------------------------

    def classify(self, text: str) -> tuple[str, float, str, bool]:
        """Run the cascade. Returns `(category, confidence, stage, escalated)`."""
        raw = self.hate_clf(text)[0]
        category = labels.normalise(raw["label"])
        confidence = float(raw["score"])

        # Stage 1 found something. Trust its category, that detail is the whole
        # reason for using a dialect-specific model.
        if category != labels.NEUTRAL:
            return category, confidence, "hate", False

        # Stage 1 said neutral. Escalate when it is confident *or* not, but
        # record whether the neutral was shaky, since that is the case the
        # notebook's version silently accepted.
        escalated = True
        low_confidence = confidence < settings.neutral_confidence_floor

        toxic = self.toxicity_clf(text)[0]
        toxic_category = labels.normalise(toxic["label"])
        toxic_score = float(toxic["score"])

        if toxic_category != labels.NEUTRAL and toxic_score >= settings.toxicity_threshold:
            return toxic_category, toxic_score, "toxicity", escalated

        # Neither stage found harm. Report the *lower* of the two confidences so
        # a shaky neutral does not look certain downstream.
        reported = min(confidence, toxic_score) if low_confidence else confidence
        return labels.NEUTRAL, reported, "toxicity", escalated

    # --- masking -----------------------------------------------------------

    def mask(self, text: str) -> str:
        """Redact the individual tokens the hate model scores as harmful.

        Word-level classification is a blunt instrument, these models were
        trained on sentences, and a single word out of context is a distribution
        they never saw. The threshold is therefore set high (0.80): over-masking
        a clean word is more visible and more irritating than leaving one through,
        given the sentence is already flagged.
        """
        words = text.split()
        if not words:
            return text

        results = self.hate_clf(words)
        masked = []
        for word, result in zip(words, results):
            category = labels.normalise(result["label"])
            score = float(result["score"])
            if category != labels.NEUTRAL and score >= settings.mask_threshold:
                masked.append("*" * max(len(word), 2))
            else:
                masked.append(word)
        return " ".join(masked)

    # --- public API --------------------------------------------------------

    def moderate(self, text: str) -> Verdict:
        """Classify `text` and decide what to do about it."""
        category, confidence, stage, escalated = self.classify(text)

        if category == labels.NEUTRAL:
            return Verdict(
                text=text, category=category, confidence=confidence, stage=stage,
                action="allow", explanation=EXPLANATIONS[labels.NEUTRAL],
                escalated=escalated,
            )

        explanation = EXPLANATIONS.get(category, EXPLANATIONS[labels.OFFENSIVE])

        if category in settings.mask_categories:
            return Verdict(
                text=text, category=category, confidence=confidence, stage=stage,
                action="mask", explanation=explanation, masked_text=self.mask(text),
                escalated=escalated,
            )

        # Sexism and religious discrimination are usually carried by the
        # sentence's structure, not by a single removable word, so masking would
        # produce a mutilated sentence that still says the same thing. Flag instead.
        return Verdict(
            text=text, category=category, confidence=confidence, stage=stage,
            action="flag", explanation=explanation, escalated=escalated,
        )

    def moderate_batch(self, texts: list[str]) -> list[Verdict]:
        return [self.moderate(t) for t in texts]
