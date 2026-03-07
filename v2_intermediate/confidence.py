"""
v2_intermediate/confidence.py — 3-component weighted confidence scorer.

Formula
-------
  confidence = w1 * attention_concentration
             + w2 * localization_score
             + w3 * anchor_validation_score

  Default weights: 0.20 / 0.40 / 0.40  (must sum to 1.0)

Why three components?
- attention_concentration: how focused the model was when reading the source
- localization_score: does the model read from the right position?
- anchor_validation_score: is the context causally necessary?

Thresholds
----------
  >= 0.80 : HIGH
  >= 0.55 : MEDIUM
  >= 0.30 : LOW
  <  0.30 : VERY LOW
"""

from __future__ import annotations

from shared.schemas import LocalizationResult, AnchorValidationResult, ConfidenceBreakdown


_THRESHOLDS = [
    (0.80, "HIGH      — extraction is reliable"),
    (0.55, "MEDIUM    — extraction is plausible; consider manual review"),
    (0.30, "LOW       — extraction is uncertain; manual review recommended"),
    (0.00, "VERY LOW  — extraction should not be trusted"),
]


def _interpret(confidence: float) -> str:
    for threshold, label in _THRESHOLDS:
        if confidence >= threshold:
            return label
    return _THRESHOLDS[-1][1]


class WeightedConfidenceScorer:
    """
    3-component weighted confidence scorer.

    Parameters
    ----------
    attention_weight : float
        Weight for the gradient attribution concentration score.
    localization_weight : float
        Weight for the localization (value substitution) score.
    anchor_weight : float
        Weight for the anchor validation (context corruption) score.
        Weights must sum to 1.0.
    """

    def __init__(
        self,
        attention_weight: float = 0.20,
        localization_weight: float = 0.40,
        anchor_weight: float = 0.40,
    ) -> None:
        total = attention_weight + localization_weight + anchor_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.4f}. "
                f"({attention_weight} + {localization_weight} + {anchor_weight})"
            )
        self.attention_weight = attention_weight
        self.localization_weight = localization_weight
        self.anchor_weight = anchor_weight

    def compute(
        self,
        attention_concentration: float,
        localization: LocalizationResult,
        anchor_validation: AnchorValidationResult,
    ) -> ConfidenceBreakdown:
        """
        Compute 3-component weighted confidence.

        Parameters
        ----------
        attention_concentration : float
            Score from GradientAnchorFinder.find() — how focused the model was.
        localization : LocalizationResult
        anchor_validation : AnchorValidationResult
        """
        L = localization.score
        A = anchor_validation.score
        C = float(attention_concentration)

        combined = (
            self.attention_weight * C
            + self.localization_weight * L
            + self.anchor_weight * A
        )
        combined = max(0.0, min(1.0, combined))

        return ConfidenceBreakdown(
            localization_score=L,
            anchor_validation_score=A,
            combined_confidence=combined,
            interpretation=_interpret(combined),
            attention_score=C,
        )
