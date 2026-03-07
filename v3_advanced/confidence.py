"""
v3_advanced/confidence.py — 4-component probabilistic confidence scorer.

Formula
-------
  confidence = w1 * attention_concentration
             + w2 * localization_score
             + w3 * anchor_validation_score
             + w4 * entropy_score

  Default weights: 0.15 / 0.35 / 0.35 / 0.15   (must sum to 1.0)

When MLM score is available it modulates the anchor_validation component:
  effective_anchor = 0.7 * anchor_validation_score + 0.3 * mlm_score

Bootstrap confidence interval
------------------------------
Given the individual trial results in LocalizationResult and
AnchorValidationResult, we bootstrap the combined confidence score:
  • Resample localization trials with replacement N times.
  • Resample anchor_validation trials with replacement N times.
  • Each resample produces a combined_score → build a distribution.
  • Report [2.5th, 97.5th] percentile as the 95 % CI.

This gives a honest uncertainty estimate at no additional LLM cost.
"""

from __future__ import annotations

import math
import random
from typing import Optional

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


def _bootstrap_ci(
    loc: LocalizationResult,
    anchor_val: AnchorValidationResult,
    w_loc: float,
    w_anchor: float,
    n_bootstrap: int = 500,
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap 95 % confidence interval on the combined score.

    Resamples individual trial outcomes with replacement and recomputes
    the weighted average.  Returns (lower_bound, upper_bound).
    """
    loc_successes = [t.success for t in loc.trials]
    anchor_failures = [t.extraction_failed for t in anchor_val.trials]

    if not loc_successes and not anchor_failures:
        return (0.0, 1.0)

    bootstrap_scores: list[float] = []
    for _ in range(n_bootstrap):
        if loc_successes:
            loc_sample = random.choices(loc_successes, k=len(loc_successes))
            l_score = sum(loc_sample) / len(loc_sample)
        else:
            l_score = 0.0

        if anchor_failures:
            anchor_sample = random.choices(anchor_failures, k=len(anchor_failures))
            a_score = sum(anchor_sample) / len(anchor_sample)
        else:
            a_score = 0.0

        total_w = w_loc + w_anchor
        bootstrap_scores.append((l_score * w_loc + a_score * w_anchor) / total_w)

    bootstrap_scores.sort()
    alpha = 1.0 - ci
    lo_idx = int(math.floor(alpha / 2 * n_bootstrap))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    lo = bootstrap_scores[max(0, lo_idx)]
    hi = bootstrap_scores[min(len(bootstrap_scores) - 1, hi_idx)]
    return (lo, hi)


class AdvancedConfidenceScorer:
    """
    4-component probabilistic confidence scorer with bootstrap CI.

    Parameters
    ----------
    attention_weight / localization_weight / anchor_weight / entropy_weight : float
        Weights must sum to 1.0.
    use_mlm_modulation : bool
        When True and an MLM score is provided, the effective anchor score
        is a weighted blend of anchor_validation_score and mlm_score.
    n_bootstrap : int
        Number of bootstrap resamples for confidence interval (0 = skip CI).
    """

    def __init__(
        self,
        attention_weight: float = 0.15,
        localization_weight: float = 0.35,
        anchor_weight: float = 0.35,
        entropy_weight: float = 0.15,
        use_mlm_modulation: bool = True,
        n_bootstrap: int = 500,
    ) -> None:
        total = attention_weight + localization_weight + anchor_weight + entropy_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.4f}. "
                f"({attention_weight} + {localization_weight} + "
                f"{anchor_weight} + {entropy_weight})"
            )
        self.attention_weight = attention_weight
        self.localization_weight = localization_weight
        self.anchor_weight = anchor_weight
        self.entropy_weight = entropy_weight
        self.use_mlm_modulation = use_mlm_modulation
        self.n_bootstrap = n_bootstrap

    def compute(
        self,
        attention_concentration: float,
        localization: LocalizationResult,
        anchor_validation: AnchorValidationResult,
        entropy_score: float,
        mlm_score: Optional[float] = None,
        mlm_perplexity: Optional[float] = None,
    ) -> ConfidenceBreakdown:
        """
        Compute 4-component confidence with optional MLM modulation and bootstrap CI.

        Parameters
        ----------
        attention_concentration : float  (from gradient anchor finder)
        localization : LocalizationResult
        anchor_validation : AnchorValidationResult
        entropy_score : float            (from entropy_scorer)
        mlm_score : float, optional      (from mlm_scorer, [0,1])
        mlm_perplexity : float, optional (raw perplexity for display)
        """
        L = localization.score
        A = anchor_validation.score
        C = float(attention_concentration)
        E = float(entropy_score)

        # Optional MLM modulation of anchor component
        if self.use_mlm_modulation and mlm_score is not None:
            A_effective = 0.7 * A + 0.3 * mlm_score
        else:
            A_effective = A

        combined = (
            self.attention_weight * C
            + self.localization_weight * L
            + self.anchor_weight * A_effective
            + self.entropy_weight * E
        )
        combined = max(0.0, min(1.0, combined))

        # Bootstrap CI
        ci: Optional[tuple[float, float]] = None
        if self.n_bootstrap > 0:
            ci = _bootstrap_ci(
                localization, anchor_validation,
                self.localization_weight, self.anchor_weight,
                n_bootstrap=self.n_bootstrap,
            )

        return ConfidenceBreakdown(
            localization_score=L,
            anchor_validation_score=A,
            combined_confidence=combined,
            interpretation=_interpret(combined),
            attention_score=C,
            entropy_score=E,
            mlm_perplexity=mlm_perplexity,
            confidence_interval=ci,
        )
