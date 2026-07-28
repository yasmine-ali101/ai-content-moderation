"""Configuration for the moderation cascade."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Fine-tuned on Egyptian-dialect Arabic hate speech; emits the fine-grained
# categories (Racism, Sexism, Religious Discrimination, Offensive, Neutral).
HATE_MODEL = os.getenv(
    "HATE_MODEL", "IbrahimAmin/marbertv2-finetuned-egyptian-hate-speech-classification"
)

# Broad multilingual toxicity model. Used as the second stage: it catches
# hostility the dialect model calls neutral, at the cost of no category detail.
TOXICITY_MODEL = os.getenv("TOXICITY_MODEL", "akhooli/xlm-r-large-arabic-toxic")


@dataclass(frozen=True)
class Settings:
    hate_model: str = HATE_MODEL
    toxicity_model: str = TOXICITY_MODEL

    # Below this confidence the hate model's own "neutral" call is not trusted,
    # and the text is escalated to the toxicity stage regardless.
    neutral_confidence_floor: float = float(os.getenv("NEUTRAL_CONFIDENCE_FLOOR", "0.85"))

    # Minimum toxicity score required to overturn a "neutral" verdict.
    toxicity_threshold: float = float(os.getenv("TOXICITY_THRESHOLD", "0.60"))

    # Word-level masking confidence. Set high: masking a clean word is a visible,
    # annoying error for the user, so the bar to redact should exceed the bar to flag.
    mask_threshold: float = float(os.getenv("MASK_THRESHOLD", "0.80"))

    device: int = int(os.getenv("MODERATION_DEVICE", "-1"))  # -1 = CPU

    # Categories whose text gets redacted rather than merely flagged.
    mask_categories: tuple[str, ...] = ("Offensive", "Racism", "Extremism")


settings = Settings()
