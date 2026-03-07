"""
v1_basic/anchor_finder.py — Heuristic anchor finding (no model internals).

Algorithm
---------
Primary:  Find the sentence(s) that directly contain the extracted value via
          character-level search.  Expand to word boundaries + context padding.

Fallback: When the value is not found literally in the text (paraphrase,
          numeric rounding, etc.), run sentence masking — mask each sentence
          one at a time, re-run extraction, and score each sentence by how
          much extraction degrades when it is absent.  The most important
          sentence(s) form the anchor.

This approach requires only the model's .extract() method, making it
compatible with any LLM including black-box API models.
"""

from __future__ import annotations

import re
from typing import Optional

from shared.schemas import AnchorResult, TokenAttribution
from shared.models import HFExtractionModel


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Return list of (sentence, start, end) triples."""
    sentences = []
    for m in re.finditer(r"[^.!?\n]*[.!?\n]?", text):
        s = m.group().strip()
        if s:
            sentences.append((s, m.start(), m.end()))
    return sentences


def _word_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / (len(set_a | set_b) + 1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class HeuristicAnchorFinder:
    """
    Text-based anchor finder — no gradients, no model internals.

    Parameters
    ----------
    model : HFExtractionModel
        Used only via .extract() for sentence-masking fallback.
    context_chars : int
        Characters of padding added on each side of the found span.
    min_anchor_chars : int
        Minimum anchor length; short anchors are extended.
    """

    def __init__(
        self,
        model: HFExtractionModel,
        context_chars: int = 80,
        min_anchor_chars: int = 40,
    ) -> None:
        self.model = model
        self.context_chars = context_chars
        self.min_anchor_chars = min_anchor_chars

    # ── Public API ────────────────────────────────────────────────────────────

    def find(self, source_text: str, field: str, extracted_value: str) -> AnchorResult:
        """
        Find anchor for *extracted_value* in *source_text*.

        Tries direct value-match first; falls back to sentence masking.
        """
        # Primary: sentence(s) directly containing the value
        result = self._value_match_anchor(source_text, extracted_value)
        if result is not None:
            return result

        # Fallback: sentence masking
        return self._sentence_mask_anchor(source_text, field, extracted_value)

    # ── Primary: direct value match ───────────────────────────────────────────

    def _value_match_anchor(
        self, source_text: str, extracted_value: str
    ) -> Optional[AnchorResult]:
        """
        Find the sentence(s) that literally contain the extracted value.
        Returns None if the value cannot be found.
        """
        value_lower = extracted_value.lower().strip()
        text_lower = source_text.lower()

        # Exact character match
        idx = text_lower.find(value_lower)
        if idx < 0:
            # Try stripping punctuation
            stripped = re.sub(r"[^\w\s]", "", value_lower)
            idx = text_lower.find(stripped)
            if idx < 0:
                return None

        # Expand to sentence boundaries then add context padding
        value_end = idx + len(value_lower)
        anchor_start, anchor_end = self._expand_to_words(
            source_text, idx, value_end
        )

        # Build token_attributions as sentence-level scores (all 1.0 for value sentence)
        token_attrs = [TokenAttribution(
            token=source_text[idx:value_end],
            score=1.0,
            char_start=idx,
            char_end=value_end,
        )]

        return AnchorResult(
            text=source_text[anchor_start:anchor_end],
            start_char=anchor_start,
            end_char=anchor_end,
            token_attributions=token_attrs,
            method="value_match",
        )

    # ── Fallback: sentence masking ────────────────────────────────────────────

    def _sentence_mask_anchor(
        self, source_text: str, field: str, extracted_value: str
    ) -> AnchorResult:
        """
        Score each sentence by degradation when masked.
        The most important sentence(s) form the anchor.
        """
        sentences = _split_sentences(source_text)
        if not sentences:
            return self._whole_text_anchor(source_text)

        scores: list[float] = []
        for sent, s_start, s_end in sentences:
            masked = source_text[:s_start] + " " * (s_end - s_start) + source_text[s_end:]
            try:
                result = self.model.extract(masked, field)
                sim = _word_similarity(result, extracted_value)
                scores.append(1.0 - sim)
            except Exception:
                scores.append(0.0)

        # Boost sentences that directly contain the value
        for i, (sent, s_start, s_end) in enumerate(sentences):
            if extracted_value.lower() in sent.lower():
                scores[i] = max(scores[i], 0.9)

        max_s = max(scores) if max(scores) > 0 else 1.0
        norm_scores = [s / max_s for s in scores]

        top_indices = [i for i, s in enumerate(norm_scores) if s > 0.5]
        if not top_indices:
            top_indices = [norm_scores.index(max(norm_scores))]

        span_start = sentences[top_indices[0]][1]
        span_end = sentences[top_indices[-1]][2]
        anchor_start, anchor_end = self._expand_to_words(source_text, span_start, span_end)

        token_attrs = [
            TokenAttribution(
                token=sent[:30],
                score=norm_scores[i],
                char_start=s_start,
                char_end=s_end,
            )
            for i, (sent, s_start, s_end) in enumerate(sentences)
        ]

        return AnchorResult(
            text=source_text[anchor_start:anchor_end],
            start_char=anchor_start,
            end_char=anchor_end,
            token_attributions=token_attrs,
            method="sentence_mask",
        )

    def _whole_text_anchor(self, source_text: str) -> AnchorResult:
        return AnchorResult(
            text=source_text,
            start_char=0,
            end_char=len(source_text),
            token_attributions=[],
            method="whole_text",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _expand_to_words(self, text: str, start: int, end: int) -> tuple[int, int]:
        """Add context padding and snap to word boundaries."""
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
