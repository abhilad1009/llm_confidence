"""
benchmark/baselines/self_consistency.py — Self-consistency confidence baseline.

Sample N outputs at temperature T; confidence = fraction of outputs matching
the majority answer (Wang et al., 2023).
"""

from __future__ import annotations

import collections
import torch

from benchmark.baselines.base import BaseConfidenceMethod
from benchmark.schemas import BenchmarkExample, MethodResult


class SelfConsistencyConfidence(BaseConfidenceMethod):
    """
    Self-consistency: sample N outputs, return majority-vote agreement ratio.

    Args:
        model: HFExtractionModel instance.
        n_samples: Number of samples per example (default 10).
        temperature: Sampling temperature (default 0.7).
        correctness_fn: Callable(predicted, gold) -> bool.
    """

    name = "self_consistency"

    def __init__(
        self,
        model,
        n_samples: int = 10,
        temperature: float = 0.7,
        correctness_fn=None,
    ) -> None:
        self._model = model
        self.n_samples = n_samples
        self.temperature = temperature
        self._correctness_fn = correctness_fn or _default_correct

    def predict(self, example: BenchmarkExample) -> MethodResult:
        samples = self._sample(example.source_text, example.query)
        if not samples:
            return MethodResult(
                example_id=example.id,
                method=self.name,
                confidence=0.0,
                extracted_value="",
                is_correct=False,
            )

        counts: dict[str, int] = collections.Counter(
            _normalize(s) for s in samples
        )
        majority, majority_count = counts.most_common(1)[0]
        confidence = majority_count / len(samples)

        # Use the majority answer as the predicted value
        is_correct = self._correctness_fn(majority, example.gold_answer)
        return MethodResult(
            example_id=example.id,
            method=self.name,
            confidence=confidence,
            extracted_value=majority,
            is_correct=is_correct,
            metadata={"n_samples": len(samples), "counts": dict(counts)},
        )

    def _sample(self, text: str, field: str) -> list[str]:
        model = self._model
        prompt = model.build_prompt(text, field)
        inputs = model.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=model.max_input_tokens,
            truncation=True,
            padding=True,
        ).to(model.device)

        samples: list[str] = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                out_ids = model.model.generate(
                    **inputs,
                    max_new_tokens=model.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.95,
                )
                if model.is_seq2seq:
                    new_ids = out_ids[0]
                else:
                    input_len = inputs["input_ids"].shape[1]
                    new_ids = out_ids[0][input_len:]
                decoded = model.tokenizer.decode(new_ids, skip_special_tokens=True)
                samples.append(_clean(decoded))
        return samples


def _clean(text: str) -> str:
    text = text.strip()
    if ":" in text:
        parts = text.split(":", 1)
        if len(parts[0].split()) <= 3:
            text = parts[1].strip()
    return text.split("\n")[0].strip()


def _normalize(text: str) -> str:
    return text.strip().lower()


def _default_correct(predicted: str, gold: str) -> bool:
    return predicted.strip().lower() == gold.strip().lower()
