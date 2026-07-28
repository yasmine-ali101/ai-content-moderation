"""Arabic (Egyptian-dialect) content moderation via a two-stage classifier cascade."""

from .labels import CATEGORIES, is_harmful, normalise
from .pipeline import ModerationPipeline, Verdict

__version__ = "0.1.0"
__all__ = ["ModerationPipeline", "Verdict", "CATEGORIES", "normalise", "is_harmful"]
