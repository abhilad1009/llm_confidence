"""
v3_advanced/anchor_finder.py — Gradient attribution + MLM perplexity re-scoring.

Algorithm
---------
1. Run gradient attribution (same as v2) to identify the top-K token span
   and compute attention_concentration.
2. Compute the MLM pseudo-perplexity of the identified anchor text.
3. Return the anchor annotated with both scores.
   Both scores are available to the v3 confidence scorer.

The MLM re-scoring does not change *which* anchor is selected (gradient
attribution is still authoritative), but it contributes the mlm_perplexity
field to AnchorResult for use in the 4-component confidence formula.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from shared.schemas import AnchorResult, TokenAttribution
from shared.models import HFExtractionModel
from v2_intermediate.anchor_finder import GradientAnchorFinder, _attention_concentration
from v3_advanced.mlm_scorer import MLMAnchorScorer


class AdvancedAnchorFinder:
    """
    Gradient attribution anchor finder augmented with MLM perplexity scoring.

    Parameters
    ----------
    model : HFExtractionModel (or AdvancedHFExtractionModel)
    mlm_scorer : MLMAnchorScorer, optional
        If None, MLM scoring is skipped (mlm_perplexity = None).
    top_k_fraction : float
    context_chars : int
    min_anchor_chars : int
    """

    def __init__(
        self,
        model: HFExtractionModel,
        mlm_scorer: Optional[MLMAnchorScorer] = None,
        top_k_fraction: float = 0.30,
        context_chars: int = 60,
        min_anchor_chars: int = 40,
    ) -> None:
        self._gradient_finder = GradientAnchorFinder(
            model,
            top_k_fraction=top_k_fraction,
            context_chars=context_chars,
            min_anchor_chars=min_anchor_chars,
        )
        self._mlm_scorer = mlm_scorer

    # ── Public API ────────────────────────────────────────────────────────────

    def find(
        self,
        source_text: str,
        field: str,
        extracted_value: str,
    ) -> tuple[AnchorResult, float, Optional[float]]:
        """
        Find anchor using gradient attribution, then score with MLM.

        Returns
        -------
        anchor : AnchorResult
            AnchorResult.mlm_perplexity is populated if MLM scorer is available.
        concentration : float
            Attention concentration score for the confidence scorer.
        mlm_score : float or None
            Normalised MLM score in [0, 1] (higher = more natural anchor).
            None if mlm_scorer was not provided.
        """
        anchor, concentration = self._gradient_finder.find(
            source_text, field, extracted_value
        )

        mlm_score: Optional[float] = None
        if self._mlm_scorer is not None:
            try:
                mlm_score = self._mlm_scorer.compute_mlm_score(anchor.text)
                # Store raw perplexity in the anchor for display
                anchor.mlm_perplexity = self._mlm_scorer.compute_perplexity(anchor.text)
            except Exception as exc:
                print(f"[AdvancedAnchorFinder] MLM scoring failed ({exc}); skipping.")

        return anchor, concentration, mlm_score
