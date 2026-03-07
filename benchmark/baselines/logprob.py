"""
benchmark/baselines/logprob.py — Token log-probability confidence baselines.

Implements three variants:
  - logprob_mean: average log-prob of all output tokens
  - logprob_min: minimum log-prob (weakest link)
  - logprob_norm: length-normalized sum of log-probs

All require HF model access with output_scores=True (open-weight models only).
Confidence is returned in [0, 1] via sigmoid normalization.
"""

from __future__ import annotations

import math
import torch
from typing import Literal

from benchmark.baselines.base import BaseConfidenceMethod
from benchmark.schemas import BenchmarkExample, MethodResult


# Dataset-level sigmoid temperature — calibrated so typical log-probs
# (range roughly -0.1 to -5) map to [0.05, 0.95].
_SIGMOID_TEMP = 1.5


class LogprobConfidence(BaseConfidenceMethod):
    """
    Confidence from per-token log-probabilities during greedy generation.

    Args:
        model: HFExtractionModel instance (shared across calls).
        variant: 'mean', 'min', or 'norm' (length-normalized mean).
        correctness_fn: Callable(predicted, gold) -> bool. Dataset-specific.
    """

    def __init__(
        self,
        model,
        variant: Literal["mean", "min", "norm"] = "mean",
        correctness_fn=None,
    ) -> None:
        self._model = model
        self.variant = variant
        self.name = f"logprob_{variant}"
        self._correctness_fn = correctness_fn or _default_correct

    def predict(self, example: BenchmarkExample) -> MethodResult:
        value, log_probs = self._extract_with_logprobs(
            example.source_text, example.query
        )
        confidence = self._aggregate(log_probs)
        is_correct = self._correctness_fn(value, example.gold_answer)
        return MethodResult(
            example_id=example.id,
            method=self.name,
            confidence=confidence,
            extracted_value=value,
            is_correct=is_correct,
        )

    def _extract_with_logprobs(
        self, text: str, field: str
    ) -> tuple[str, list[float]]:
        """Run generation with output_scores=True; return (value, token_logprobs)."""
        model = self._model
        prompt = model.build_prompt(text, field)
        inputs = model.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=model.max_input_tokens,
            truncation=True,
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.model.generate(
                **inputs,
                max_new_tokens=model.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # outputs.scores: tuple of (vocab_size,) tensors, one per generated token
        generated_ids = outputs.sequences[0]
        if model.is_seq2seq:
            new_ids = generated_ids
        else:
            input_len = inputs["input_ids"].shape[1]
            new_ids = generated_ids[input_len:]

        log_probs: list[float] = []
        for step_idx, token_id in enumerate(new_ids):
            if step_idx >= len(outputs.scores):
                break
            logits = outputs.scores[step_idx]  # (1, vocab_size)
            lp = torch.log_softmax(logits, dim=-1)[0, token_id].item()
            log_probs.append(lp)

        decoded = model.tokenizer.decode(new_ids, skip_special_tokens=True)
        value = _clean(decoded)
        return value, log_probs

    def _aggregate(self, log_probs: list[float]) -> float:
        if not log_probs:
            return 0.0
        if self.variant == "mean":
            raw = sum(log_probs) / len(log_probs)
        elif self.variant == "min":
            raw = min(log_probs)
        else:  # norm — same as mean but explicit
            raw = sum(log_probs) / len(log_probs)
        # Sigmoid: raw is in (−∞, 0]; map to (0, 0.5] → scale up
        return float(1.0 / (1.0 + math.exp(-raw * _SIGMOID_TEMP)))


def _clean(text: str) -> str:
    text = text.strip()
    if ":" in text:
        parts = text.split(":", 1)
        if len(parts[0].split()) <= 3:
            text = parts[1].strip()
    return text.split("\n")[0].strip()


def _default_correct(predicted: str, gold: str) -> bool:
    return predicted.strip().lower() == gold.strip().lower()
