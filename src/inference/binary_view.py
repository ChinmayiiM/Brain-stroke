"""Binary STROKE / NO STROKE presentation of the 3-class model output.

The trained model and the Phase 5 prediction are unchanged: this module only
aggregates the model's real class probabilities into two groups for display.

    P(STROKE)    = sum of probabilities of the configured positive classes
    P(NO STROKE) = sum of probabilities of the configured negative classes   (= 1 − P(STROKE))
    label        = STROKE if P(STROKE) >= 0.5 else NO STROKE
    confidence   = probability of the shown label

This is NOT the same as taking the largest of the three class probabilities:
e.g. Haemorrhagic 0.30, Ischemic 0.25, Normal 0.45 → the 3-class argmax is
Normal, but P(STROKE) = 0.55, so the binary view shows STROKE (55 %).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.inference.errors import ClassMappingError

if TYPE_CHECKING:  # type hint only — keeps this module free of the TensorFlow import
    from src.inference.predictor import PredictionResult


@dataclass(frozen=True)
class BinaryViewConfig:
    positive_label: str
    negative_label: str
    positive_classes: tuple[str, ...]
    negative_classes: tuple[str, ...]

    @classmethod
    def from_config(cls, config: dict) -> "BinaryViewConfig":
        view = config["inference"]["binary_view"]
        return cls(
            positive_label=str(view["positive_label"]),
            negative_label=str(view["negative_label"]),
            positive_classes=tuple(view["positive_classes"]),
            negative_classes=tuple(view["negative_classes"]),
        )

    def validate(self, class_names: tuple[str, ...]) -> None:
        """Every model class must belong to exactly one group."""
        grouped = list(self.positive_classes) + list(self.negative_classes)
        if sorted(grouped) != sorted(class_names) or len(set(grouped)) != len(grouped):
            raise ClassMappingError(
                f"binary_view groups {grouped} must partition the model classes {list(class_names)} exactly"
            )


@dataclass(frozen=True)
class BinaryResult:
    label: str
    confidence: float
    positive_probability: float
    negative_probability: float
    is_positive: bool


def to_binary(result: PredictionResult, view: BinaryViewConfig) -> BinaryResult:
    """Aggregate a real 3-class :class:`PredictionResult` into the binary view."""
    view.validate(tuple(result.probabilities))
    positive = sum(result.probabilities[c] for c in view.positive_classes)
    negative = sum(result.probabilities[c] for c in view.negative_classes)
    is_positive = positive >= 0.5
    return BinaryResult(
        label=view.positive_label if is_positive else view.negative_label,
        confidence=positive if is_positive else negative,
        positive_probability=positive,
        negative_probability=negative,
        is_positive=is_positive,
    )
