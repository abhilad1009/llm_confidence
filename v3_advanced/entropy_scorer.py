"""
v3_advanced/entropy_scorer.py — Logit entropy as a confidence signal.

Intuition
---------
When the model generates each output token it produces a probability
distribution over the vocabulary.  If the distribution is sharply peaked
(low entropy), the model was "decisive".  If it is flat (high entropy),
the model was uncertain.

We compute the average normalised Shannon entropy across all generated tokens,
then invert it so that the final score is:

  entropy_score = 1 − mean_normalised_entropy     ∈ [0, 1]

  1.0  →  model was fully decisive (very low entropy) — high confidence signal
  0.0  →  model was maximally uncertain (uniform distribution) — low confidence

This score complements localization and anchor tests because it captures
*generative* uncertainty — how unsure the model was about what to write —
rather than *location* uncertainty.
"""

from __future__ import annotations

import math
import torch
from typing import Sequence


def compute_generation_entropy(logits_list: Sequence[torch.Tensor]) -> float:
    """
    Compute the entropy-based confidence score from per-token generation logits.

    Parameters
    ----------
    logits_list : list of Tensor, shape (vocab_size,) each
        Raw (un-normalised) logits for each generated token, as returned by
        AdvancedHFExtractionModel.extract_with_logits().

    Returns
    -------
    float
        entropy_score in [0, 1]: 1 = confident (low entropy), 0 = uncertain.
        Returns 0.5 (neutral) if logits_list is empty.
    """
    if not logits_list:
        return 0.5  # neutral — no data

    token_entropies: list[float] = []
    for logits in logits_list:
        logits = logits.float()
        # Softmax to probabilities
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-12)
        h = -(probs * log_probs).sum().item()

        # Normalise by the maximum possible entropy: log(vocab_size)
        max_h = math.log(logits.shape[-1])
        normalised = h / max_h if max_h > 0 else 0.0
        token_entropies.append(normalised)

    mean_normalised_entropy = sum(token_entropies) / len(token_entropies)

    # Invert: high entropy → low confidence score
    return 1.0 - mean_normalised_entropy


def entropy_to_confidence_label(entropy_score: float) -> str:
    """Human-readable label for the entropy confidence score."""
    if entropy_score >= 0.85:
        return "decisive"
    if entropy_score >= 0.65:
        return "moderate"
    if entropy_score >= 0.45:
        return "uncertain"
    return "very uncertain"
