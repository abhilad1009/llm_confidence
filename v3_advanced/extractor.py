"""
v3_advanced/extractor.py — AdvancedConfidenceExtractor.

Combines all signals into a probabilistic confidence estimate:

  1. model.extract_with_logits()         → value + per-token logits
  2. AdvancedAnchorFinder.find()         → anchor + attention_concentration + mlm_score
  3. Localizer.test()                    → localization score
  4. AnchorValidator.validate()          → anchor validation score
  5. compute_generation_entropy(logits)  → entropy_score
  6. AdvancedConfidenceScorer.compute()  → 4-component confidence + CI

Confidence formula (default weights):
  confidence = 0.15 * attention_concentration
             + 0.35 * localization_score
             + 0.35 * anchor_validation_score   (MLM-modulated)
             + 0.15 * entropy_score

  Plus a bootstrap 95 % confidence interval from trial-level resampling.
"""

from __future__ import annotations

from typing import Optional

from shared.models import HFExtractionModel
from shared.localizer import Localizer
from shared.anchor_validator import AnchorValidator
from shared.value_generator import ValueGenerator
from shared.schemas import ExtractionResult

from v3_advanced.models import AdvancedHFExtractionModel
from v3_advanced.anchor_finder import AdvancedAnchorFinder
from v3_advanced.entropy_scorer import compute_generation_entropy
from v3_advanced.mlm_scorer import MLMAnchorScorer
from v3_advanced.confidence import AdvancedConfidenceScorer


class AdvancedConfidenceExtractor:
    """
    LLM confidence extractor — advanced (probabilistic) version.

    Parameters
    ----------
    model_name : str
        HuggingFace model name.  Default: 'Qwen/Qwen3-0.6B'
    mlm_model_name : str
        MLM model for anchor perplexity.  Default: 'distilbert-base-uncased'
        Set to None to disable MLM scoring.
    device : str, optional
    n_trials : int
        Perturbation trials per stage.
    top_k_fraction : float
    attention_weight / localization_weight / anchor_weight / entropy_weight : float
        Confidence formula weights (must sum to 1.0).
    corruption_rate : float
    n_bootstrap : int
        Bootstrap resamples for CI.  Set to 0 to disable.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        mlm_model_name: Optional[str] = "distilbert-base-uncased",
        device: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
        n_trials: int = 10,
        top_k_fraction: float = 0.30,
        attention_weight: float = 0.15,
        localization_weight: float = 0.35,
        anchor_weight: float = 0.35,
        entropy_weight: float = 0.15,
        corruption_rate: float = 0.65,
        n_bootstrap: int = 500,
    ) -> None:
        self.n_trials = n_trials

        self._model = AdvancedHFExtractionModel(
            model_name=model_name,
            device=device,
            load_in_4bit=load_in_4bit,
        )

        mlm_scorer: Optional[MLMAnchorScorer] = None
        if mlm_model_name:
            mlm_scorer = MLMAnchorScorer(
                model_name=mlm_model_name,
                device=device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
            )

        self._value_gen = ValueGenerator()
        self._anchor_finder = AdvancedAnchorFinder(
            self._model,
            mlm_scorer=mlm_scorer,
            top_k_fraction=top_k_fraction,
        )
        self._localizer = Localizer(self._model, self._value_gen, n_trials=n_trials)
        self._anchor_validator = AnchorValidator(
            self._model, n_trials=n_trials, corruption_rate=corruption_rate
        )
        self._confidence_scorer = AdvancedConfidenceScorer(
            attention_weight=attention_weight,
            localization_weight=localization_weight,
            anchor_weight=anchor_weight,
            entropy_weight=entropy_weight,
            n_bootstrap=n_bootstrap,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, text: str, field: str) -> ExtractionResult:
        """
        Extract *field* from *text* with full probabilistic confidence.

        Returns
        -------
        ExtractionResult
            confidence_breakdown includes attention_score, entropy_score,
            mlm_perplexity, and confidence_interval (bootstrap 95% CI).
        """
        print(f"\n{'='*60}")
        print(f" [v3_advanced] Extracting: {field!r}")
        print(f"{'='*60}")

        print("[Stage 1] Running LLM extraction (capturing logits) …")
        value, logits_list = self._model.extract_with_logits(text, field)
        print(f"          Raw extracted value: {value!r}")
        print(f"          Captured logits for {len(logits_list)} generated tokens.")

        print("[Stage 2] Finding anchor (gradient + MLM) …")
        try:
            anchor, concentration, mlm_score = self._anchor_finder.find(text, field, value)
        except Exception as exc:
            print(f"          Gradient pass failed ({exc}) — cannot proceed with v3.")
            raise

        print(f"          Attention concentration: {concentration:.2%}")
        if mlm_score is not None:
            print(f"          MLM anchor score: {mlm_score:.2%}")
        if anchor.mlm_perplexity is not None:
            print(f"          MLM perplexity: {anchor.mlm_perplexity:.2f}")
        preview = anchor.text[:60].replace("\n", " ")
        print(f"          Anchor: «{preview}{'…' if len(anchor.text) > 60 else ''}»")

        print("[Stage 2b] Computing logit entropy …")
        entropy_score = compute_generation_entropy(logits_list)
        print(f"          Entropy score: {entropy_score:.2%} "
              f"({'decisive' if entropy_score >= 0.7 else 'uncertain'})")

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
            entropy_score=entropy_score,
            mlm_score=mlm_score,
            mlm_perplexity=anchor.mlm_perplexity,
        )

        ci_str = ""
        if breakdown.confidence_interval:
            lo, hi = breakdown.confidence_interval
            ci_str = f"  95% CI: [{lo:.0%}, {hi:.0%}]"

        print(f"[Stage 5] Confidence: {breakdown.combined_confidence:.2%}  "
              f"→  {breakdown.interpretation}{ci_str}")

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
