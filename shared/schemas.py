"""
shared/schemas.py — Data structures for the LLM Confidence Extractor pipeline.

All three versions (basic, intermediate, advanced) use these types.
ConfidenceBreakdown carries optional fields for v2/v3 signals so v1 results
remain fully compatible without changes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


# ──────────────────────────────────────────────
# Stage 1 – Anchor
# ──────────────────────────────────────────────

@dataclass
class TokenAttribution:
    """Attribution score for a single input token."""
    token: str
    score: float          # Normalised [0, 1]
    char_start: int
    char_end: int


@dataclass
class AnchorResult:
    """
    The minimal passage in the source text that the model attends to
    when producing the extracted value.
    """
    text: str                                   # Anchor substring
    start_char: int                             # Inclusive offset in original text
    end_char: int                               # Exclusive offset in original text
    token_attributions: list[TokenAttribution]  # Per-token scores (empty for v1)
    method: str                                 # 'value_match' | 'sentence_mask' | 'gradient' | 'gradient_mlm' | 'whole_text'
    top_token_fraction: float = 0.30            # Fraction of tokens deemed high-signal (v2/v3)
    mlm_perplexity: Optional[float] = None      # v3: anchor perplexity from MLM scorer


# ──────────────────────────────────────────────
# Stage 2 – Localization
# ──────────────────────────────────────────────

@dataclass
class LocalizationTrial:
    """Result of a single value-replacement trial."""
    injected_value: str     # What was placed in the anchor
    extracted_value: str    # What the model returned
    success: bool           # extracted_value ≈ injected_value


@dataclass
class LocalizationResult:
    """
    Measures whether the model reads from the right *location*.

    Protocol: replace the original value with N random alternatives
    inside the anchor, re-run extraction, check the model tracks the
    substitution.  A high score means the model truly reads from that
    position.
    """
    score: float                        # Success rate [0, 1]
    n_trials: int
    n_success: int
    trials: list[LocalizationTrial]


# ──────────────────────────────────────────────
# Stage 3 – Anchor Validation
# ──────────────────────────────────────────────

@dataclass
class AnchorValidationTrial:
    """Result of a single context-corruption trial."""
    corrupted_anchor: str   # Anchor with context words swapped
    extracted_value: str    # What the model returned
    extraction_failed: bool # True if model could no longer extract correctly


@dataclass
class AnchorValidationResult:
    """
    Confirms the anchor context is *causally necessary*.

    Protocol: keep the value in place but replace surrounding anchor
    words with random words.  If the anchor context is what the model
    uses, extraction should break.  High failure rate → anchor is real.
    """
    score: float                            # Failure rate [0, 1]  (higher is better)
    n_trials: int
    n_failures: int
    trials: list[AnchorValidationTrial]


# ──────────────────────────────────────────────
# Final Result
# ──────────────────────────────────────────────

@dataclass
class ConfidenceBreakdown:
    localization_score: float       # Stage-2 score
    anchor_validation_score: float  # Stage-3 score
    combined_confidence: float      # Final combined score
    interpretation: str             # Human-readable verdict

    # v2 / v3 — gradient attribution concentration (None for v1)
    attention_score: Optional[float] = None

    # v3 — logit entropy confidence (1 = low entropy = model was decisive)
    entropy_score: Optional[float] = None

    # v3 — MLM pseudo-perplexity of the anchor text (lower = more natural)
    mlm_perplexity: Optional[float] = None

    # v3 — bootstrap 95 % confidence interval on combined_confidence
    confidence_interval: Optional[tuple[float, float]] = None


@dataclass
class ExtractionResult:
    """Top-level result returned by every version's extractor."""
    field: str
    raw_text: str
    value: Any
    anchor: AnchorResult
    localization: LocalizationResult
    anchor_validation: AnchorValidationResult
    confidence: float               # [0, 1]
    confidence_breakdown: ConfidenceBreakdown

    def __str__(self) -> str:
        bd = self.confidence_breakdown
        lines = [
            f"Field         : {self.field}",
            f"Value         : {self.value}",
            f"Confidence    : {self.confidence:.2%}",
        ]
        if bd.attention_score is not None:
            lines.append(f"  Attention   : {bd.attention_score:.2%}")
        lines.append(
            f"  Localization: {self.localization.score:.2%} "
            f"({self.localization.n_success}/{self.localization.n_trials} trials)"
        )
        lines.append(
            f"  Anchor valid: {self.anchor_validation.score:.2%} "
            f"({self.anchor_validation.n_failures}/{self.anchor_validation.n_trials} failures)"
        )
        if bd.entropy_score is not None:
            lines.append(f"  Entropy conf: {bd.entropy_score:.2%}")
        if bd.mlm_perplexity is not None:
            lines.append(f"  MLM perp.   : {bd.mlm_perplexity:.2f}")
        if bd.confidence_interval is not None:
            lo, hi = bd.confidence_interval
            lines.append(f"  95% CI      : [{lo:.2%}, {hi:.2%}]")
        anchor_preview = self.anchor.text[:80].replace("\n", " ")
        lines.append(
            f"Anchor [{self.anchor.method}]: "
            f"«{anchor_preview}{'…' if len(self.anchor.text) > 80 else ''}»"
        )
        lines.append(f"Verdict       : {bd.interpretation}")
        return "\n".join(lines)
