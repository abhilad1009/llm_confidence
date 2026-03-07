"""
benchmark/baselines/attention_entropy.py — Attention-entropy confidence baseline.

Uses the gradient attribution concentration from v2_intermediate/anchor_finder.py
as a standalone confidence signal. This isolates the attention component to show
how much value the perturbation steps add.

Confidence = 1 − normalised_entropy(attribution_scores)
           = anchor_result.concentration  (already computed in GradientAnchorFinder)
"""

from __future__ import annotations

from benchmark.baselines.base import BaseConfidenceMethod
from benchmark.schemas import BenchmarkExample, MethodResult


class AttentionEntropyConfidence(BaseConfidenceMethod):
    """
    Confidence from gradient attribution concentration (open-weight models only).

    Args:
        model: HFExtractionModel instance (must support forward_with_grad()).
        correctness_fn: Callable(predicted, gold) -> bool.
    """

    name = "attention_entropy"

    def __init__(self, model, correctness_fn=None) -> None:
        self._model = model
        self._correctness_fn = correctness_fn or _default_correct
        # Lazy-import to avoid circular deps
        self._finder = None

    def _get_finder(self):
        if self._finder is None:
            from v2_intermediate.anchor_finder import GradientAnchorFinder
            self._finder = GradientAnchorFinder(self._model)
        return self._finder

    def predict(self, example: BenchmarkExample) -> MethodResult:
        # Step 1: greedy extraction
        value = self._model.extract(example.source_text, example.query)

        if not value or value.strip() == "":
            return MethodResult(
                example_id=example.id,
                method=self.name,
                confidence=0.0,
                extracted_value=value,
                is_correct=False,
            )

        # Step 2: gradient attribution → concentration score
        finder = self._get_finder()
        try:
            anchor_result, concentration = finder.find(
                example.source_text, example.query, value
            )
        except Exception:
            concentration = 0.0
            anchor_result = None

        confidence = float(concentration)
        is_correct = self._correctness_fn(value, example.gold_answer)

        token_attrs = None
        if anchor_result and anchor_result.token_attributions:
            token_attrs = [
                {
                    "token": ta.token,
                    "score": ta.score,
                    "char_start": ta.char_start,
                    "char_end": ta.char_end,
                }
                for ta in anchor_result.token_attributions
            ]

        return MethodResult(
            example_id=example.id,
            method=self.name,
            confidence=confidence,
            extracted_value=value,
            is_correct=is_correct,
            token_attributions=token_attrs,
        )


def _default_correct(predicted: str, gold: str) -> bool:
    return predicted.strip().lower() == gold.strip().lower()
