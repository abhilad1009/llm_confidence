"""
v1_basic/confidence.py — 2-component confidence scorer.

Formula
-------
Both localization (L) and anchor-validation (A) must be high for the
extraction to be trustworthy:

  confidence = geometric_mean(L, A) = sqrt(L × A)

The geometric mean returns 0 if *either* component is 0, which is more
conservative than the arithmetic mean when one signal is absent.

Thresholds
----------
  >= 0.80 : HIGH
  >= 0.55 : MEDIUM
  >= 0.30 : LOW
  <  0.30 : VERY LOW
"""

from __future__ import annotations

import math

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


class ConfidenceScorer:
    """
    Combines localization + anchor-validation scores via geometric mean.

    Parameters
    ----------
    use_geometric_mean : bool
        If True (default), use sqrt(L * A).  Otherwise weighted arithmetic.
    loc_weight / anchor_weight : float
        Used only when use_geometric_mean=False.
    """

    def __init__(
        self,
        use_geometric_mean: bool = True,
        loc_weight: float = 0.5,
        anchor_weight: float = 0.5,
    ) -> None:
        self.use_geometric_mean = use_geometric_mean
        self.loc_weight = loc_weight
        self.anchor_weight = anchor_weight

    def compute(
        self,
        localization: LocalizationResult,
        anchor_validation: AnchorValidationResult,
    ) -> ConfidenceBreakdown:
        L = localization.score
        A = anchor_validation.score

        if self.use_geometric_mean:
            combined = math.sqrt(L * A)
        else:
            total_w = self.loc_weight + self.anchor_weight
            combined = (L * self.loc_weight + A * self.anchor_weight) / total_w

        combined = self._smooth(combined, localization.n_trials + anchor_validation.n_trials)

        return ConfidenceBreakdown(
            localization_score=L,
            anchor_validation_score=A,
            combined_confidence=combined,
            interpretation=_interpret(combined),
            # attention_score, entropy_score, mlm_perplexity, confidence_interval
            # remain None for v1
        )

    @staticmethod
    def _smooth(score: float, n_total: int) -> float:
        """Laplace-style smoothing: pull extreme values toward 0.5 for small samples."""
        k = max(0, 4 - n_total // 5)
        return (score * n_total + 0.5 * k) / (n_total + k)
