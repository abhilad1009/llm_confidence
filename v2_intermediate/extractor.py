"""
v2_intermediate/extractor.py — IntermediateConfidenceExtractor.

Uses gradient × embedding attribution (||grad · emb||₂) to identify the
anchor span, then runs the same perturbation tests as v1.  The confidence
score is a 3-component weighted combination:

  confidence = 0.20 * attention_concentration
             + 0.40 * localization_score
             + 0.40 * anchor_validation_score

Requires an open-weight HuggingFace model with gradient support.

Pipeline
--------
  1. model.extract()                   → raw value string
  2. GradientAnchorFinder.find()       → anchor + attention_concentration
  3. Localizer.test()                  → localization score
  4. AnchorValidator.validate()        → anchor validation score
  5. WeightedConfidenceScorer.compute() → 3-component confidence
"""

from __future__ import annotations

from typing import Optional

from shared.models import HFExtractionModel
from shared.localizer import Localizer
from shared.anchor_validator import AnchorValidator
from shared.value_generator import ValueGenerator
from shared.schemas import ExtractionResult

from v2_intermediate.anchor_finder import GradientAnchorFinder
from v2_intermediate.confidence import WeightedConfidenceScorer


class IntermediateConfidenceExtractor:
    """
    LLM confidence extractor — intermediate (gradient attribution) version.

    Parameters
    ----------
    model_name : str
        HuggingFace model name.  Default: 'Qwen/Qwen3-0.6B'
    device : str, optional
        'cpu', 'cuda', or 'mps'.  Auto-detected if None.
    n_trials : int
        Perturbation trials for each of stages 2 and 3.
    top_k_fraction : float
        Fraction of source tokens used to define the anchor span.
    attention_weight / localization_weight / anchor_weight : float
        Confidence formula weights (must sum to 1.0).
    corruption_rate : float
        Fraction of anchor context words corrupted in stage 3.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
        n_trials: int = 10,
        top_k_fraction: float = 0.30,
        attention_weight: float = 0.20,
        localization_weight: float = 0.40,
        anchor_weight: float = 0.40,
        corruption_rate: float = 0.65,
    ) -> None:
        self.n_trials = n_trials

        self._model = HFExtractionModel(
            model_name=model_name,
            device=device,
            load_in_4bit=load_in_4bit,
        )

        self._value_gen = ValueGenerator()
        self._anchor_finder = GradientAnchorFinder(
            self._model, top_k_fraction=top_k_fraction
        )
        self._localizer = Localizer(self._model, self._value_gen, n_trials=n_trials)
        self._anchor_validator = AnchorValidator(
            self._model, n_trials=n_trials, corruption_rate=corruption_rate
        )
        self._confidence_scorer = WeightedConfidenceScorer(
            attention_weight=attention_weight,
            localization_weight=localization_weight,
            anchor_weight=anchor_weight,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, text: str, field: str) -> ExtractionResult:
        """
        Extract *field* from *text* with gradient-attributed confidence.

        Returns
        -------
        ExtractionResult
        """
        print(f"\n{'='*60}")
        print(f" [v2_intermediate] Extracting: {field!r}")
        print(f"{'='*60}")

        print("[Stage 1] Running LLM extraction …")
        value = self._model.extract(text, field)
        print(f"          Raw extracted value: {value!r}")

        print("[Stage 2] Finding anchor (gradient attribution) …")
        try:
            anchor, concentration = self._anchor_finder.find(text, field, value)
            print(f"          Attention concentration: {concentration:.2%}")
        except Exception as exc:
            print(f"          Gradient pass failed ({exc}) — cannot proceed with v2.")
            raise

        preview = anchor.text[:60].replace("\n", " ")
        print(f"          Anchor: «{preview}{'…' if len(anchor.text) > 60 else ''}»")

        print(f"[Stage 3] Running {self.n_trials} localization trials …")
        localization = self._localizer.test(
            full_text=text,
            field=field,
            original_value=value,
            anchor_text=anchor.text,
            anchor_start=anchor.start_char,
            anchor_end=anchor.end_char,
        )
        print(f"          Localization score: {localization.score:.2%} "
              f"({localization.n_success}/{localization.n_trials})")

        print(f"[Stage 4] Running {self.n_trials} anchor-validation trials …")
        anchor_validation = self._anchor_validator.validate(
            full_text=text,
            field=field,
            original_value=value,
            anchor_text=anchor.text,
            anchor_start=anchor.start_char,
            anchor_end=anchor.end_char,
        )
        print(f"          Anchor-validation score: {anchor_validation.score:.2%} "
              f"({anchor_validation.n_failures}/{anchor_validation.n_trials} failures)")

        breakdown = self._confidence_scorer.compute(
            attention_concentration=concentration,
            localization=localization,
            anchor_validation=anchor_validation,
        )
        print(f"[Stage 5] Confidence: {breakdown.combined_confidence:.2%}  "
              f"→  {breakdown.interpretation}")

        return ExtractionResult(
            field=field,
            raw_text=text,
            value=value,
            anchor=anchor,
            localization=localization,
            anchor_validation=anchor_validation,
            confidence=breakdown.combined_confidence,
            confidence_breakdown=breakdown,
        )

    def extract_fields(self, text: str, fields: list[str]) -> dict[str, ExtractionResult]:
        """Extract multiple fields from the same text."""
        return {field: self.extract(text, field) for field in fields}
