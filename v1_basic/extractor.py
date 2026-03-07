"""
v1_basic/extractor.py — BasicConfidenceExtractor.

The simplest version of the pipeline.  Uses only text-based heuristics for
anchor finding — no gradients, no model internals.  Works with any LLM that
exposes a simple (text, field) -> str extraction interface.

Pipeline
--------
  1. model.extract(text, field)         → raw value string
  2. HeuristicAnchorFinder.find()       → anchor via value-match or sentence masking
  3. Localizer.test()                   → localization score
  4. AnchorValidator.validate()         → anchor validation score
  5. ConfidenceScorer.compute()         → sqrt(loc * anchor) confidence
"""

from __future__ import annotations

from typing import Optional

from shared.models import HFExtractionModel
from shared.localizer import Localizer
from shared.anchor_validator import AnchorValidator
from shared.value_generator import ValueGenerator
from shared.schemas import ExtractionResult

from v1_basic.anchor_finder import HeuristicAnchorFinder
from v1_basic.confidence import ConfidenceScorer


class BasicConfidenceExtractor:
    """
    LLM confidence extractor — basic (text heuristic) version.

    Parameters
    ----------
    model_name : str
        HuggingFace model name.  Default: 'Qwen/Qwen3-0.6B'
    device : str, optional
        'cpu', 'cuda', or 'mps'.  Auto-detected if None.
    n_trials : int
        Perturbation trials for each of stages 2 and 3.  Recommended: 10–20.
    context_chars : int
        Characters of padding around the detected anchor span.
    corruption_rate : float
        Fraction of anchor context words corrupted in stage 3.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
        n_trials: int = 10,
        context_chars: int = 80,
        corruption_rate: float = 0.65,
    ) -> None:
        self.n_trials = n_trials

        self._model = HFExtractionModel(
            model_name=model_name,
            device=device,
            load_in_4bit=load_in_4bit,
        )

        self._value_gen = ValueGenerator()
        self._anchor_finder = HeuristicAnchorFinder(
            self._model, context_chars=context_chars
        )
        self._localizer = Localizer(self._model, self._value_gen, n_trials=n_trials)
        self._anchor_validator = AnchorValidator(
            self._model, n_trials=n_trials, corruption_rate=corruption_rate
        )
        self._confidence_scorer = ConfidenceScorer()

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, text: str, field: str) -> ExtractionResult:
        """
        Extract *field* from *text* with confidence measurement.

        Returns
        -------
        ExtractionResult
            Contains value, anchor, localization score, anchor-validation score,
            and a final confidence measure.
        """
        print(f"\n{'='*60}")
        print(f" [v1_basic] Extracting: {field!r}")
        print(f"{'='*60}")

        print("[Stage 1] Running LLM extraction …")
        value = self._model.extract(text, field)
        print(f"          Raw extracted value: {value!r}")

        print("[Stage 2] Finding anchor (heuristic) …")
        anchor = self._anchor_finder.find(text, field, value)
        print(f"          Method: {anchor.method}")
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

        breakdown = self._confidence_scorer.compute(localization, anchor_validation)
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
