"""Evaluation for the moderation cascade.

Two levels, because they answer different questions:

**Binary (harmful vs neutral)** is the deployment-relevant one. A moderation
system's job is first to decide whether to act at all; getting `Racism` vs
`Offensive` wrong is a much cheaper mistake than letting abuse through.

**Multi-class** measures whether the category labels are trustworthy enough to
show a user or route to a specific policy.

Both are reported, because a system can be strong at one and weak at the other,
and a single accuracy number hides that.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import labels


@dataclass
class Example:
    text: str
    label: str
    note: str | None = None


def load_eval_set(path: str | Path) -> list[Example]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Example(text=item["text"], label=item["label"], note=item.get("note"))
        for item in payload["examples"]
    ]


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass
class BinaryScores:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @property
    def accuracy(self) -> float:
        total = self.true_positive + self.false_positive + self.true_negative + self.false_negative
        return _safe_divide(self.true_positive + self.true_negative, total)

    @property
    def precision(self) -> float:
        return _safe_divide(self.true_positive, self.true_positive + self.false_positive)

    @property
    def recall(self) -> float:
        return _safe_divide(self.true_positive, self.true_positive + self.false_negative)

    @property
    def f1(self) -> float:
        return _safe_divide(2 * self.precision * self.recall, self.precision + self.recall)

    def as_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
        }


def score_binary(examples: list[Example], predictions: list[str]) -> BinaryScores:
    """Harmful vs neutral. 'Harmful' is the positive class."""
    scores = BinaryScores()
    for example, predicted in zip(examples, predictions):
        actual_harmful = labels.is_harmful(example.label)
        predicted_harmful = labels.is_harmful(predicted)
        if actual_harmful and predicted_harmful:
            scores.true_positive += 1
        elif not actual_harmful and predicted_harmful:
            scores.false_positive += 1
        elif not actual_harmful and not predicted_harmful:
            scores.true_negative += 1
        else:
            scores.false_negative += 1
    return scores


def score_multiclass(examples: list[Example], predictions: list[str]) -> dict:
    """Exact-match accuracy plus per-category precision/recall/F1."""
    correct = sum(1 for e, p in zip(examples, predictions) if e.label == p)
    accuracy = _safe_divide(correct, len(examples))

    per_category = {}
    for category in labels.CATEGORIES:
        tp = sum(1 for e, p in zip(examples, predictions) if e.label == category and p == category)
        fp = sum(1 for e, p in zip(examples, predictions) if e.label != category and p == category)
        fn = sum(1 for e, p in zip(examples, predictions) if e.label == category and p != category)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        support = sum(1 for e in examples if e.label == category)
        if support == 0 and tp + fp == 0:
            continue
        per_category[category] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(_safe_divide(2 * precision * recall, precision + recall), 4),
            "support": support,
        }

    macro_f1 = _safe_divide(
        sum(v["f1"] for v in per_category.values()), len(per_category)
    )
    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(examples),
        "macro_f1": round(macro_f1, 4),
        "per_category": per_category,
    }


def confusion_pairs(examples: list[Example], predictions: list[str]) -> list[dict]:
    """Every misclassification, so failures can be read rather than inferred."""
    return [
        {"text": e.text, "expected": e.label, "predicted": p, "note": e.note}
        for e, p in zip(examples, predictions)
        if e.label != p
    ]
