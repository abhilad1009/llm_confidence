"""
v2_intermediate/anchor_finder.py — Gradient attribution anchor finding.

Algorithm
---------
1. Run a differentiable forward pass (model.forward_with_grad).
2. Compute per-token attribution  ||grad · emb||₂.
3. Restrict to tokens inside the source-text region of the prompt.
4. Select top-K tokens (top_k_fraction of source tokens).
5. Take the bounding span, expand to word boundaries + context padding.

Raises ValueError if gradients are unavailable (e.g. quantised model without
bitsandbytes, or inference-only backends).  Unlike v1, there is no heuristic
fallback here — failure is surfaced explicitly so the caller can decide.

Attention concentration
-----------------------
The concentration of attribution scores (Gini coefficient) is returned as a
proxy for how "focused" the model was.  High concentration → the model read
from a small, specific region.  This feeds into the 3-component confidence
scorer as the attention_score component.
"""

from __future__ import annotations

import re
import numpy as np
from typing import Optional

from shared.schemas import AnchorResult, TokenAttribution
from shared.models import HFExtractionModel


# ──────────────────────────────────────────────────────────────────────────────
# Attribution concentration metric
# ──────────────────────────────────────────────────────────────────────────────

def _attention_concentration(attr_scores: np.ndarray) -> float:
    """
    Compute 1 − normalised_entropy of attribution scores.

    Returns a value in [0, 1]:
      • 1.0  →  fully concentrated (single spike)
      • 0.0  →  perfectly uniform (diffuse attention)
    """
    scores = np.abs(attr_scores)
    total = scores.sum()
    if total < 1e-9:
        return 0.0
    p = scores / total
    # Normalised Shannon entropy
    entropy = -np.sum(p * np.log(p + 1e-12)) / (np.log(len(p) + 1e-12) + 1e-12)
    return float(1.0 - entropy)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt offset helper
# ──────────────────────────────────────────────────────────────────────────────

def _prompt_text_offsets(prompt: str, source_text: str) -> Optional[tuple[int, int]]:
    idx = prompt.find(source_text)
    return (idx, idx + len(source_text)) if idx >= 0 else None


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class GradientAnchorFinder:
    """
    Anchor finder using gradient × embedding attribution.

    Parameters
    ----------
    model : HFExtractionModel
    top_k_fraction : float
        Fraction of source-text tokens considered high-signal (default 0.30).
    context_chars : int
        Characters of padding added on each side of the raw attribution span.
    min_anchor_chars : int
        Minimum anchor length.
    """

    def __init__(
        self,
        model: HFExtractionModel,
        top_k_fraction: float = 0.30,
        context_chars: int = 60,
        min_anchor_chars: int = 40,
    ) -> None:
        self.model = model
        self.top_k_fraction = top_k_fraction
        self.context_chars = context_chars
        self.min_anchor_chars = min_anchor_chars

    # ── Public API ────────────────────────────────────────────────────────────

    def find(
        self,
        source_text: str,
        field: str,
        extracted_value: str,
    ) -> tuple[AnchorResult, float]:
        """
        Find anchor via gradient attribution.

        Returns
        -------
        anchor : AnchorResult
        concentration : float
            Attention concentration score in [0, 1] for the confidence scorer.

        Raises
        ------
        ValueError
            If the gradient pass fails (e.g. model doesn't support inputs_embeds).
        """
        prompt = self.model.build_prompt(source_text, field)
        attr_scores, offset_mapping = self.model.forward_with_grad(
            source_text, field, extracted_value
        )

        text_span = _prompt_text_offsets(prompt, source_text)

        token_attrs: list[TokenAttribution] = []
        for i, (tok_char_start, tok_char_end) in enumerate(offset_mapping):
            if i >= len(attr_scores):
                break
            score = float(attr_scores[i])

            in_source = True
            if text_span is not None:
                in_source = tok_char_start >= text_span[0] and tok_char_end <= text_span[1]

            if in_source:
                offset = text_span[0] if text_span else 0
                src_start = max(0, tok_char_start - offset)
                src_end = max(0, tok_char_end - offset)
                token_attrs.append(TokenAttribution(
                    token=source_text[src_start:src_end],
                    score=score,
                    char_start=src_start,
                    char_end=src_end,
                ))

        if not token_attrs:
            raise ValueError("No tokens mapped to source text — check tokeniser offsets.")

        # Compute concentration on all source-token attributions
        all_scores = np.array([t.score for t in token_attrs])
        concentration = _attention_concentration(all_scores)

        # Select top-K tokens by attribution
        k = max(1, int(len(token_attrs) * self.top_k_fraction))
        top_tokens = sorted(token_attrs, key=lambda t: t.score, reverse=True)[:k]

        span_start = min(t.char_start for t in top_tokens)
        span_end = max(t.char_end for t in top_tokens)

        anchor_start, anchor_end = self._expand_to_words(source_text, span_start, span_end)

        anchor = AnchorResult(
            text=source_text[anchor_start:anchor_end],
            start_char=anchor_start,
            end_char=anchor_end,
            token_attributions=token_attrs,
            method="gradient",
            top_token_fraction=self.top_k_fraction,
        )
        return anchor, concentration

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _expand_to_words(self, text: str, start: int, end: int) -> tuple[int, int]:
        start = max(0, start - self.context_chars)
        end = min(len(text), end + self.context_chars)

        while start > 0 and text[start - 1] not in " \n\t":
            start -= 1
        while end < len(text) and text[end] not in " \n\t":
            end += 1

        if (end - start) < self.min_anchor_chars:
            extra = (self.min_anchor_chars - (end - start)) // 2
            start = max(0, start - extra)
            end = min(len(text), end + extra)

        return start, end
